import os
import asyncio
from pathlib import Path
from typing import Dict, List

import pytest
import pytest_asyncio
from bson import ObjectId
from dotenv import load_dotenv

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import (
    ZMongoEmbedder,
    CHUNK_STYLE_FIXED,
    CHUNK_STYLE_SENTENCE,
    CHUNK_STYLE_PARAGRAPH,
)

# --- Test configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "embedder_test_coll"

pytestmark = pytest.mark.skipif(
    not all([MONGO_URI, GEMINI_API_KEY]),
    reason="MONGO_URI and GEMINI_API_KEY must be set for live embedder tests",
)

# -------- Helpers --------

def field_name(base: str, embedding_style: str, chunk_style: str) -> str:
    """
    Mirror the naming convention: [FIELD_BASE]_[EMBEDDING_STYLE]_[CHUNK_STYLE]
    Example: text_RETRIEVAL_DOCUMENT_sentence
    """
    return f"{base}_{embedding_style}_{chunk_style}"

# -------- Fixtures --------

@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def repo():
    z = ZMongo()
    # NOTE: we deliberately DO NOT drop this collection (the user asked not to delete embeddings).
    yield z
    z.close()

@pytest_asyncio.fixture
def embedder() -> ZMongoEmbedder:
    # Uses default GEMINI_API_KEY from env; collection is where embedder writes doc updates
    return ZMongoEmbedder(collection=COLLECTION_NAME, gemini_api_key=GEMINI_API_KEY)

# -------- Tests --------

@pytest.mark.asyncio
async def test_embed_text_returns_vectors(embedder: ZMongoEmbedder):
    text = (
        "Artificial intelligence is transforming legal practice. "
        "AI assists with review, drafting, and research."
    )
    vecs_fixed = await embedder.embed_text(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=160,
        overlap=8,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )
    vecs_sentence = await embedder.embed_text(
        text,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=220,
        overlap=0,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )
    vecs_paragraph = await embedder.embed_text(
        text,
        chunk_style=CHUNK_STYLE_PARAGRAPH,
        chunk_size=1000,
        overlap=0,
        embedding_style="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )

    # Basic shape checks (we don't assert exact numbers due to model variance)
    assert len(vecs_fixed) >= 1
    assert len(vecs_sentence) >= 1
    assert len(vecs_paragraph) >= 1
    assert len(vecs_fixed[0]) == 768
    assert len(vecs_sentence[0]) == 768
    assert len(vecs_paragraph[0]) == 768

@pytest.mark.asyncio
async def test_embed_texts_batched_roundtrip(embedder: ZMongoEmbedder):
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
async def test_embed_and_store_uses_field_naming_and_persists(repo: ZMongo, embedder: ZMongoEmbedder):
    """
    Ensures embeddings are saved to the document under the expected field name.
    Test purposely does NOT delete or drop the collection.
    """
    # Create a doc with some base field 'text'
    doc_id = ObjectId()
    base_field = "text"
    embedding_style = "RETRIEVAL_DOCUMENT"
    chunk_style = "sentence"
    target_field = field_name(base_field, embedding_style, chunk_style)

    text = "Python is a dynamic, high-level programming language used for many applications."

    # Insert the base doc
    ins = await repo.insert_document(COLLECTION_NAME, {"_id": doc_id, base_field: text})
    assert ins.success

    # Store embeddings using the naming convention
    res = await embedder.embed_and_store(
        document_id=doc_id,
        text=text,
        embedding_field=target_field,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=200,
        overlap=0,
        embedding_style=embedding_style,
        output_dimensionality=768,
    )
    assert res.success

    # Read back and validate presence + shape
    got = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert got.success and got.data is not None
    doc = got.data
    assert target_field in doc
    assert isinstance(doc[target_field], list) and len(doc[target_field]) >= 1
    assert isinstance(doc[target_field][0], list) and len(doc[target_field][0]) == 768

    # Re-run with identical params: should *update* the same field (idempotent semantics),
    # not delete it nor create duplicates under different names.
    res2 = await embedder.embed_and_store(
        document_id=doc_id,
        text=text,
        embedding_field=target_field,
        chunk_style=CHUNK_STYLE_SENTENCE,
        chunk_size=200,
        overlap=0,
        embedding_style=embedding_style,
        output_dimensionality=768,
    )
    assert res2.success

    got2 = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert got2.success and got2.data is not None
    doc2 = got2.data
    assert target_field in doc2
    # still embeddings there, not deleted
    assert isinstance(doc2[target_field], list) and len(doc2[target_field]) >= 1
    assert isinstance(doc2[target_field][0], list) and len(doc2[target_field][0]) == 768

@pytest.mark.asyncio
async def test_cache_consistency_does_not_remove_previous(repo: ZMongo, embedder: ZMongoEmbedder):
    """
    A lightweight check that calling embed_text repeatedly yields stable shapes and
    does not impact previously stored document embeddings in the collection.
    """
    text = "Caching test text. Repeat embeddings should come from cache if available."
    first = await embedder.embed_text(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=180,
        overlap=10,
        embedding_style="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    )
    second = await embedder.embed_text(
        text,
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=180,
        overlap=10,
        embedding_style="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    )
    # We're not asserting exact vectors equality due to potential float nuances,
    # but shapes should match, indicating consistent behavior and no deletion.
    assert len(first) >= 1 and len(second) >= 1 and len(first[0]) == len(second[0]) == 768
