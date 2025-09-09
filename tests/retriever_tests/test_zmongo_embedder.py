import os
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from bson import ObjectId
from dotenv import load_dotenv

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    CHUNK_STYLE_FIXED,
    CHUNK_STYLE_SENTENCE,
    CHUNK_STYLE_PARAGRAPH,
    field_name,
)

# --- Test configuration ---
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "embedder_test_coll"

pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set for live embedder tests",
)

# -------- Fixtures --------

@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def repo():
    z = ZMongo()
    await z.db.drop_collection(COLLECTION_NAME) # Clean slate for tests
    yield z
    z.close()

@pytest_asyncio.fixture
def embedder(repo: ZMongo) -> ZEmbedder:
    """
    FIX: Embedder is now stateless regarding the collection.
    It uses the provided repository for DB operations.
    """
    return ZEmbedder(repository=repo, gemini_api_key=GEMINI_API_KEY)

# -------- Tests --------

@pytest.mark.asyncio
async def test_get_embedding_returns_vectors(embedder: ZEmbedder):
    """
    FIX: Renamed test to reflect the new method name `get_embedding`.
    """
    text = (
        "Artificial intelligence is transforming legal practice. "
        "AI assists with review, drafting, and research."
    )
    # FIX: Changed call from embed_text to get_embedding
    vecs_fixed = await embedder.get_embedding(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=160,
        overlap=8,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )
    vecs_sentence = await embedder.get_embedding(
        text,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=220,
        overlap=0,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )
    vecs_paragraph = await embedder.get_embedding(
        text,
        chunk_style=CHUNK_STYLE_PARAGRAPH,
        chunk_size=1000,
        overlap=0,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )

    assert len(vecs_fixed) >= 1
    assert len(vecs_sentence) >= 1
    assert len(vecs_paragraph) >= 1
    assert len(vecs_fixed[0]) == 768
    assert len(vecs_sentence[0]) == 768
    assert len(vecs_paragraph[0]) == 768

@pytest.mark.asyncio
async def test_embed_texts_batched_roundtrip(embedder: ZEmbedder):
    texts = [
        "The mitochondrion is the powerhouse of the cell.",
        "Jupiter is the fifth planet from the Sun.",
    ]
    results = await embedder.embed_texts_batched(
        texts,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=180,
        overlap=0,
        embedding_style="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    )
    assert set(results.keys()) == set(texts)
    for text, chunks in results.items():
        assert isinstance(chunks, list) and len(chunks) >= 1
        assert isinstance(chunks[0], list) and len(chunks[0]) == 768

@pytest.mark.asyncio
async def test_embed_and_store_uses_field_naming_and_persists(repo: ZMongo, embedder: ZEmbedder):
    """
    Ensures embeddings are saved to the document under the expected field name.
    """
    doc_id = ObjectId()
    base_field = "text"
    embedding_style = "RETRIEVAL_DOCUMENT"
    chunk_style = "sentence"
    target_field = field_name(base_field, embedding_style, chunk_style)
    text = "Python is a dynamic, high-level programming language used for many applications."

    ins = await repo.insert_document(COLLECTION_NAME, {"_id": doc_id, base_field: text})
    assert ins.success

    # FIX: Pass the collection name to the embed_and_store method.
    res = await embedder.embed_and_store(
        collection=COLLECTION_NAME,
        document_id=doc_id,
        text=text,
        embedding_field=target_field,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=200,
        overlap=0,
        embedding_style=embedding_style,
        output_dimensionality=768,
    )
    assert res.success, res.error

    got = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert got.success and got.data is not None
    doc = got.data
    assert target_field in doc
    assert isinstance(doc[target_field], list) and len(doc[target_field]) >= 1
    assert isinstance(doc[target_field][0], list) and len(doc[target_field][0]) == 768

    # Re-run to test idempotency
    res2 = await embedder.embed_and_store(
        collection=COLLECTION_NAME,
        document_id=doc_id,
        text=text,
        embedding_field=target_field,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=200,
        overlap=0,
        embedding_style=embedding_style,
        output_dimensionality=768,
    )
    assert res2.success, res2.error

    got2 = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert got2.success and got2.data is not None
    doc2 = got2.data
    assert target_field in doc2
    assert isinstance(doc2[target_field], list) and len(doc2[target_field]) >= 1

@pytest.mark.asyncio
async def test_cache_consistency_does_not_remove_previous(repo: ZMongo, embedder: ZEmbedder):
    """
    A lightweight check that calling get_embedding repeatedly yields stable shapes.
    """
    text = "Caching test text. Repeat embeddings should come from cache if available."
    first = await embedder.get_embedding(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=180,
        overlap=10,
        embedding_style="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    )
    second = await embedder.get_embedding(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=180,
        overlap=10,
        embedding_style="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    )
    assert len(first) >= 1 and len(second) >= 1 and len(first[0]) == len(second[0]) == 768
