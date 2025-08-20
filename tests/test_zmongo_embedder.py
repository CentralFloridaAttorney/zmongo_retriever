# zmongo_retriever/tests/test_zmongo_embedder_integration.py
# Real-data integration tests: require a running MongoDB and GEMINI_API_KEY.
# No mocks. These tests make real network/API calls.

import os
import time
import hashlib
import pytest
from bson.objectid import ObjectId

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

pytestmark = pytest.mark.asyncio

TEST_COLLECTION = "it_embedder_docs"
EMBED_FIELD = "embeddings"
CACHE_COLLECTION = "_embedding_cache"

SAMPLE_TEXT_SHORT = "Mitochondria are often called the powerhouse of the cell."
SAMPLE_TEXT_MEDIUM = (
    "Jupiter is the fifth planet from the Sun and the largest in the Solar System. "
    "It is a gas giant with a mass more than two and a half times that of all the "
    "other planets in the Solar System combined."
)

async def _mongo_ready(repo: ZMongo) -> bool:
    try:
        await repo.db.command("ping")
        return True
    except Exception:
        return False

def _require_env():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set; integration tests require a real API key.")

# ---------- Fixtures (function-scoped to avoid cross-loop clients) ----------

@pytest.fixture()
async def repo():
    r = ZMongo()
    if not await _mongo_ready(r):
        pytest.skip("MongoDB not reachable (db.command('ping') failed).")
    # clean start
    await r.delete_documents(TEST_COLLECTION, {})
    await r.delete_documents(CACHE_COLLECTION, {})
    try:
        yield r
    finally:
        # clean end
        await r.delete_documents(TEST_COLLECTION, {})
        await r.delete_documents(CACHE_COLLECTION, {})
        r.close()

@pytest.fixture()
async def embedder():
    _require_env()
    # Create the embedder (and its internal ZMongo client) in THIS test's loop
    e = ZMongoEmbedder(collection=TEST_COLLECTION)
    try:
        yield e
    finally:
        # Best effort close of the embedder's internal repository
        try:
            e.repository.close()
        except Exception:
            pass

# ------------------------------- Tests --------------------------------------

async def test_embed_text_creates_cache_and_returns_vectors(repo: ZMongo, embedder: ZMongoEmbedder):
    # Ensure cache is empty
    await repo.delete_documents(CACHE_COLLECTION, {})

    # First call should hit API and populate cache
    vectors = await embedder.embed_text(SAMPLE_TEXT_SHORT)
    assert isinstance(vectors, list) and len(vectors) >= 1
    assert all(isinstance(chunk, list) and len(chunk) > 0 for chunk in vectors)

    # Verify cache populated for the short text
    chunk_hash = hashlib.sha256(SAMPLE_TEXT_SHORT.encode("utf-8")).hexdigest()
    res = await repo.find_documents(CACHE_COLLECTION, {"text_hash": {"$in": [chunk_hash]}}, limit=10)
    assert res.success
    rows = res.data or []
    assert len(rows) == 1, "Expected a single cache entry for the short sample text"
    assert rows[0]["source_text"] == SAMPLE_TEXT_SHORT
    assert isinstance(rows[0]["embedding"], list) and len(rows[0]["embedding"]) > 0

    # Second call should be cache-first (no extra cache rows)
    before_count = len(rows)
    vectors2 = await embedder.embed_text(SAMPLE_TEXT_SHORT)
    assert len(vectors2) == len(vectors)
    res2 = await repo.find_documents(CACHE_COLLECTION, {"text_hash": {"$in": [chunk_hash]}}, limit=10)
    assert res2.success
    rows2 = res2.data or []
    assert len(rows2) == before_count, "Cache-first path should not create duplicates"

async def test_embed_texts_batched_roundtrip(repo: ZMongo, embedder: ZMongoEmbedder):
    await repo.delete_documents(CACHE_COLLECTION, {})

    texts = [SAMPLE_TEXT_SHORT, SAMPLE_TEXT_MEDIUM, SAMPLE_TEXT_SHORT]  # repeat to test de-dup
    t0 = time.time()
    result = await embedder.embed_texts_batched(texts)
    t1 = time.time()

    assert isinstance(result, dict) and SAMPLE_TEXT_SHORT in result and SAMPLE_TEXT_MEDIUM in result
    assert all(isinstance(vecs, list) and len(vecs) >= 1 for vecs in result.values())

    cache_rows = await repo.find_documents(
        CACHE_COLLECTION,
        {"source_text": {"$in": [SAMPLE_TEXT_SHORT, SAMPLE_TEXT_MEDIUM]}},
        limit=1000,
    )
    assert cache_rows.success
    rows = cache_rows.data or []
    assert len(rows) >= 2

    # Re-run same batch; cache size should stay the same
    before = len(rows)
    result2 = await embedder.embed_texts_batched(texts)
    assert set(result2.keys()) == set(result.keys())

    cache_rows_again = await repo.find_documents(
        CACHE_COLLECTION, {"source_text": {"$in": [SAMPLE_TEXT_SHORT, SAMPLE_TEXT_MEDIUM]}}, limit=1000
    )
    assert cache_rows_again.success
    rows_again = cache_rows_again.data or []
    assert len(rows_again) == before

    _ = (t1 - t0)  # optional timing info

async def test_embed_and_store_updates_document(repo: ZMongo, embedder: ZMongoEmbedder):
    insert_res = await repo.insert_document(TEST_COLLECTION, {"topic": "Biology", "text": SAMPLE_TEXT_SHORT})
    assert insert_res.success
    doc_id = insert_res.data["inserted_id"]

    store_res = await embedder.embed_and_store(doc_id, SAMPLE_TEXT_SHORT, embedding_field=EMBED_FIELD)
    assert store_res.success, f"embed_and_store failed: {store_res.error}"

    # Verify
    found = await repo.find_document(TEST_COLLECTION, {"_id": doc_id})
    assert found.success and found.data is not None
    doc = found.data
    assert EMBED_FIELD not in doc and isinstance(doc, dict) and doc['text'] == SAMPLE_TEXT_SHORT


    assert all(isinstance(chunk, list) and len(chunk) > 0 for chunk in doc[EMBED_FIELD])

    # cleanup this doc
    await repo.delete_document(TEST_COLLECTION, {"_id": ObjectId(doc_id)})

async def test_embedder_handles_invalid_id_gracefully(embedder: ZMongoEmbedder):
    bad_id = "this_is_not_an_objectid"
    res = await embedder.embed_and_store(bad_id, "won't matter", embedding_field=EMBED_FIELD)
    assert not res.success
    assert "not a valid ObjectId" in (res.error or "")
