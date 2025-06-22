import pytest
import asyncio
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zretriever import ZRetriever
from langchain.schema import Document

def random_collection():
    import random, string
    return "test_retriever_" + ''.join(random.choices(string.ascii_lowercase, k=8))

@pytest.mark.asyncio
async def test_get_zdocuments_and_chunking():
    # Setup
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    oid = ObjectId()
    doc = {"_id": oid, "database_name": "testdb", "collection_name": coll, "casebody": {"data": {"opinions": [{"text": text}]}}}
    await zm.insert_document(coll, doc)

    # Test retrieval and chunking
    docs = await retriever.get_zdocuments(oid)
    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)
    # Check that text is chunked (depends on chunk_size)
    assert any("Sentence" in d.page_content for d in docs)
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_invoke_documents_only():
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    oid = ObjectId()
    doc = {"_id": oid, "database_name": "testdb", "collection_name": coll, "casebody": {"data": {"opinions": [{"text": text}]}}}
    await zm.insert_document(coll, doc)
    result = await retriever.invoke([oid])
    assert isinstance(result, list)
    assert all(isinstance(chunk_set, list) or isinstance(chunk_set, Document) for chunk_set in result)
    await zm.delete_documents(coll)

import pytest
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zretriever import ZRetriever

@pytest.mark.asyncio
async def test_invoke_with_embedding():
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=True)
    text = "The quick brown fox jumps over the lazy dog." * 2
    oid = ObjectId()
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": text}]}}
    }
    await zm.insert_document(coll, doc)
    result = await retriever.invoke([oid])

    # Now, just check the shape:
    assert isinstance(result, list)
    # Each element should be a list of floats (an embedding vector)
    for document_set in result:
        assert isinstance(document_set, list)
        for document_chunks in document_set:
            for vector in document_chunks:
                assert isinstance(vector, list)
                assert all(isinstance(x, float) for x in vector)

    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_invoke_with_embedding_multiple_oids():
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=True)
    text = "The quick brown fox jumps over the lazy dog." * 2

    oids = [ObjectId() for _ in range(2)]
    for oid in oids:
        doc = {
            "_id": oid,
            "database_name": "testdb",
            "collection_name": coll,
            "casebody": {"data": {"opinions": [{"text": text}]}}
        }
        await zm.insert_document(coll, doc)

    result = await retriever.invoke(oids)

    assert isinstance(result, list)
    for document_set in result:
        assert isinstance(document_set, list)
        for document_chunks in document_set:
            for vector in document_chunks:
                assert isinstance(vector, list)
                assert all(isinstance(x, float) for x in vector)
    await zm.delete_documents(coll)



@pytest.mark.asyncio
async def test_invoke_invalid_id_returns_empty():
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm)
    result = await retriever.invoke([ObjectId()])
    assert result == [] or all(r == [] for r in result)

@pytest.mark.asyncio
async def test_invoke_max_tokens_zero_returns_documents():
    coll = random_collection()
    zm = ZMongo()
    # Set max_tokens_per_set to 0!
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False, max_tokens_per_set=0)
    text = "The quick brown fox jumps over the lazy dog." * 2
    oid = ObjectId()
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": text}]}}
    }
    await zm.insert_document(coll, doc)
    result = await retriever.invoke([oid])

    # Should return a list of Document objects (not chunk sets, not embeddings)
    assert isinstance(result, list)
    assert all(isinstance(d, Document) for d in result)
    for d in result:
        assert hasattr(d, "page_content")
        assert isinstance(d.page_content, str)
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_invoke_max_tokens_zero_returns_documents():
    coll = random_collection()
    zm = ZMongo()
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False, max_tokens_per_set=0)
    text = "The quick brown fox jumps over the lazy dog." * 2
    oid = ObjectId()
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": text}]}}
    }
    await zm.insert_document(coll, doc)
    result = await retriever.invoke([oid])

    # Should return a list of Document objects
    assert isinstance(result, list)
    assert all(isinstance(d, Document) for d in result)
    for d in result:
        assert hasattr(d, "page_content")
        assert isinstance(d.page_content, str)
        assert hasattr(d, "metadata")
        assert isinstance(d.metadata, dict)
    await zm.delete_documents(coll)


