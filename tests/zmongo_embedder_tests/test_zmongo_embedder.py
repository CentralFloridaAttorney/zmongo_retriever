import pytest
import asyncio
import time
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder, CHUNK_SIZE, CHUNK_OVERLAP

# Optionally, you can define a dummy or in-memory embedding store for fast benchmarking of alternatives

def random_collection():
    import random, string
    return "test_embedder_" + ''.join(random.choices(string.ascii_lowercase, k=8))

@pytest.mark.asyncio
async def test_chunking_short_text():
    embedder = ZMongoEmbedder("irrelevant")  # Repo not used for this test
    text = "This is a test sentence. " * 5
    chunks = embedder._split_chunks(text, chunk_size=16, overlap=4)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
    # Check that reassembling chunks roughly recovers the text
    assert "".join(chunks).replace(" ", "")[:30] in text.replace(" ", "")

@pytest.mark.asyncio
async def test_chunking_long_text():
    embedder = ZMongoEmbedder("irrelevant")
    # Use enough tokens to force multiple chunks
    text = "word " * (CHUNK_SIZE + CHUNK_SIZE//2)
    chunks = embedder._split_chunks(text)
    assert len(chunks) >= 2
    # Overlap check: ensure some content is repeated at chunk boundaries
    assert any(chunks[0][-10:] in c for c in chunks[1:])

@pytest.mark.asyncio
async def test_embed_and_cache(tmp_path):
    # Real DB, real model, small text (will use cache on 2nd call)
    coll = random_collection()
    zm = ZMongo()
    embedder = ZMongoEmbedder(coll, repository=zm)
    text = "cat sat on the mat"
    # Clean out embedding cache before test
    await zm.delete_documents("_embedding_cache", {"source_text": text})
    emb1 = await embedder.embed_text(text)
    assert isinstance(emb1, list)
    assert all(isinstance(e, list) for e in emb1)
    # Second embed should hit cache (see log for "Reusing cached embedding")
    emb2 = await embedder.embed_text(text)
    assert emb1 == emb2

@pytest.mark.asyncio
async def test_embed_and_store_real():
    coll = random_collection()
    zm = ZMongo()
    embedder = ZMongoEmbedder(coll, repository=zm)
    # Use much shorter text than before (to avoid BSON DocumentTooLarge error)
    # This will still create multiple chunks, but total doc will be << 16MB
    text = "A " + "very " * (CHUNK_SIZE // 4) + "long test document."
    oid = ObjectId()
    await embedder.embed_and_store(oid, text)
    # Check stored document
    found = await zm.find_document(coll, {"_id": oid})
    assert found.success
    assert "embeddings" in found.data
    # Should be a list of lists (chunks)
    assert isinstance(found.data["embeddings"], list)
    assert all(isinstance(chunk_emb, list) for chunk_emb in found.data["embeddings"])
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_embedder_rejects_invalid_input():
    embedder = ZMongoEmbedder("irrelevant")
    oid = ObjectId()
    with pytest.raises(ValueError):
        await embedder.embed_and_store(oid, "")  # empty string
    with pytest.raises(ValueError):
        await embedder.embed_and_store("not_an_oid", "some text")

######################################
# Simple Benchmarking Section
######################################

def benchmark_embedding_system(system_name, embed_func, text, n_runs=3):
    times = []
    for i in range(n_runs):
        start = time.time()
        embed_func(text)
        times.append(time.time() - start)
    print(f"{system_name}: {sum(times)/len(times):.3f} seconds average over {n_runs} runs.")

@pytest.mark.asyncio
async def test_benchmark_embedding(tmp_path):
    coll = random_collection()
    zm = ZMongo()
    embedder = ZMongoEmbedder(coll, repository=zm)
    # Reduce the size to prevent BSON errors
    text = "benchmark text " * (CHUNK_SIZE // 100)

    # first embedding takes time to load the model ~5 seconds
    await embedder.embed_text(text)
    # time the embedder
    t0 = time.time()
    await embedder.embed_text(text)
    zmongo_time = time.time() - t0
    print(f"ZMongoEmbedder: {zmongo_time:.3f} seconds to embed text of length {len(text)}.")
    assert zmongo_time < 120  # Arbitrary upper bound

import pytest
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

@pytest.mark.asyncio
@pytest.mark.parametrize("bad_input", [
    None,         # Not a string
    123,          # Not a string
    ["hi"],       # Not a string
    "",           # Empty string
])
async def test_embed_text_invalid_input_raises(bad_input):
    embedder = ZMongoEmbedder("irrelevant_collection")
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        await embedder.embed_text(bad_input)

import pytest
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

class FakeLlama:
    def create_embedding(self, text):
        return "not a dict or list"

@pytest.mark.asyncio
async def test_embed_text_unexpected_embedding_result(monkeypatch):
    embedder = ZMongoEmbedder("irrelevant_collection")
    embedder._llama = FakeLlama()  # Patch in a fake llama
    # Patch chunking to one chunk for predictability
    embedder._split_chunks = lambda text: [text]
    # Patch repo to always miss cache
    class FakeRepo:
        async def find_document(self, *a, **kw): return None
        async def insert_document(self, *a, **kw): return None
    embedder.repository = FakeRepo()
    with pytest.raises(ValueError, match="Unexpected embedding result format"):
        await embedder.embed_text("some text")

import pytest
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.utils.safe_result import SafeResult

class FakeRepoFailUpdate:
    async def update_document(self, *a, **kw):
        return SafeResult.fail("mongo write failed!")
    async def find_document(self, *a, **kw):
        return None  # not used here
    async def insert_document(self, *a, **kw):
        return None  # not used here

@pytest.mark.asyncio
async def test_embed_and_store_raises_on_failed_update(monkeypatch):
    embedder = ZMongoEmbedder("fake_collection", repository=FakeRepoFailUpdate())

    # Patch embed_text to return dummy embeddings (to skip real model)
    async def fake_embed_text(text):
        return [[0.1, 0.2, 0.3]]
    embedder.embed_text = fake_embed_text

    with pytest.raises(RuntimeError, match="Failed to store embeddings: mongo write failed!"):
        await embedder.embed_and_store(ObjectId(), "dummy text")


import pytest
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo_retriever import ZRetriever
from zmongo_toolbag.zmongo import ZMongo

@pytest.mark.asyncio
async def test_get_zdocuments_skips_non_string_page_content(monkeypatch, caplog):
    coll = "test_zdocs_skip"
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm)
    oid = ObjectId()
    # Insert a doc with page_content that is not a string
    bad_doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": {"not": "a string"}}]}}
    }
    await zm.insert_document(coll, bad_doc)
    with caplog.at_level("WARNING"):
        docs = await retriever.get_zdocuments([oid])
    # Should skip the document
    assert docs == [] or all(hasattr(d, "page_content") and isinstance(d.page_content, str) for d in docs)
    # Confirm warning in logs
    assert any("Invalid page content" in r.message for r in caplog.records)
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_embed_documents_raises_on_invalid_embedding(monkeypatch):
    coll = "test_zdocs_embedfail"
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm)
    doc = retriever.get_chunk_sets([type("Doc", (), {"page_content": "ok"})()])[0][0]
    # Monkeypatch embedder to return an invalid result (not list)
    class DummyEmbedder:
        async def embed_text(self, text): return None
    retriever.embedder = DummyEmbedder()
    with pytest.raises(ValueError, match="embed_text returned invalid"):
        await retriever.embed_documents([doc])
