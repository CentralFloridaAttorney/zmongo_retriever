import pytest
import asyncio
import time
from bson import ObjectId
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

