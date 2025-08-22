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

# ... (imports and other fixtures)

@pytest.fixture()
async def embedder(repo: ZMongo):  # <-- FIX: Accept 'repo' as a parameter
    _require_env()
    # Create the embedder in THIS test's loop, using the injected 'repo'
    e = ZMongoEmbedder(repository=repo, collection=TEST_COLLECTION) # <-- FIX: Use the 'repo' variable
    try:
        yield e
    finally:
        # Best effort close of the embedder's internal repository
        try:
            e.repository.close()
        except Exception:
            pass

# ... (other tests)

# ------------------------------- Tests --------------------------------------

async def test_embed_text_creates_cache_and_returns_vectors(repo: ZMongo, embedder: ZMongoEmbedder):
    # Ensure cache is empty to start
    await repo.delete_documents(CACHE_COLLECTION, {})

    # 1. First call should hit the API and populate the cache
    vectors = await embedder.embed_text(SAMPLE_TEXT_SHORT)
    assert isinstance(vectors, list) and len(vectors) == 1
    assert isinstance(vectors[0], list) and len(vectors[0]) > 0
    first_vector = vectors[0]

    # 2. Verify cache was populated by using the embedder's public API.
    cached_vector = await embedder.get_embedding_vector(SAMPLE_TEXT_SHORT)
    assert cached_vector is not None, "get_embedding_vector should find the entry in the cache"
    assert cached_vector == first_vector, "The cached vector must be identical to the one just created"

    # 3. Second call should be a cache hit and not create new cache entries

    # FIX: Extract the integer from the SafeResult object
    count_res_before = await repo.count_documents(CACHE_COLLECTION, {})
    assert count_res_before.success, "Failed to count documents before the second call"
    count_before = count_res_before.data['count']
    assert count_before > 0, "Cache should have at least one entry"

    # This call should be a cache hit
    vectors2 = await embedder.embed_text(SAMPLE_TEXT_SHORT)
    assert len(vectors2) == len(vectors)

    # FIX: Extract the integer from the SafeResult object again for the final check
    count_res_after = await repo.count_documents(CACHE_COLLECTION, {})
    assert count_res_after.success, "Failed to count documents after the second call"
    count_after = count_res_after.data['count']
    assert count_after == count_before, "Cache-first path should not create duplicate cache entries"
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

# ... (inside the test_embed_and_store_updates_document function)

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
    # FIX: The embedding field SHOULD be in the document after the operation.
    assert EMBED_FIELD in doc and isinstance(doc, dict) and doc['text'] == SAMPLE_TEXT_SHORT

    assert all(isinstance(chunk, list) and len(chunk) > 0 for chunk in doc[EMBED_FIELD])

    # cleanup this doc
    await repo.delete_document(TEST_COLLECTION, {"_id": ObjectId(doc_id)})


async def test_embedder_handles_invalid_id_gracefully(embedder: ZMongoEmbedder):
    bad_id = "this_is_not_an_objectid"
    res = await embedder.embed_and_store(bad_id, "won't matter", embedding_field=EMBED_FIELD)
    assert not res.success
    assert "not a valid ObjectId" in (res.error or "")
