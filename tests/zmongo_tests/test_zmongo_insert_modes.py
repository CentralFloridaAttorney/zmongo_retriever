import pytest
import asyncio
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.utils.safe_result import SafeResult
from bson.objectid import ObjectId

@pytest.mark.asyncio
async def test_insert_documents_fast_mode(tmp_path):
    zm = ZMongo()
    collection = "test_fast_mode"
    docs = [{"_id": ObjectId(), "val": i} for i in range(10)]
    await zm.delete_documents(collection)
    result = await zm.insert_documents(collection, docs, fast_mode=True)
    assert isinstance(result, SafeResult)
    db_docs = await zm.db[collection].find({}).to_list(length=20)
    ids_in_db = set([d["_id"] for d in db_docs])
    for d in docs:
        assert d["_id"] in ids_in_db

@pytest.mark.asyncio
async def test_insert_documents_buffered_mode(tmp_path):
    zm = ZMongo()
    collection = "test_buffered_mode"
    docs = [{"_id": ObjectId(), "val": i} for i in range(10)]
    await zm.delete_documents(collection)
    result = await zm.insert_documents(collection, docs, buffer_only=True)
    # Docs should be in DB because flush is called in insert_documents
    db_docs_after = await zm.db[collection].find({}).to_list(length=20)
    assert len(db_docs_after) == 10
    ids_in_db = set([d["_id"] for d in db_docs_after])
    for d in docs:
        assert d["_id"] in ids_in_db
    assert result.data.get("buffered") is True
    assert result.data.get("count") == len(docs)

@pytest.mark.asyncio
async def test_manual_buffered_mode(tmp_path):
    zm = ZMongo()
    collection = "test_manual_buffered"
    docs = [{"_id": ObjectId(), "val": i} for i in range(5)]
    await zm.delete_documents(collection)
    # Buffer without flush
    buffered_cache = zm._get_buffered_cache(collection)
    for doc in docs:
        await buffered_cache.set(str(doc["_id"]), doc, buffer_only=True)
    db_docs_before = await zm.db[collection].find({}).to_list(length=20)
    assert len(db_docs_before) == 0
    # Now flush
    await zm.flush_buffered_inserts(collection)
    db_docs_after = await zm.db[collection].find({}).to_list(length=20)
    assert len(db_docs_after) == 5
    for d in docs:
        assert d["_id"] in [x["_id"] for x in db_docs_after]

@pytest.mark.asyncio
async def test_insert_documents_default_mode_with_cache(tmp_path):
    zm = ZMongo()
    collection = "test_default_mode"
    docs = [{"_id": ObjectId(), "val": i} for i in range(5)]
    await zm.delete_documents(collection)
    result = await zm.insert_documents(collection, docs)
    assert isinstance(result, SafeResult)
    # DB contains all
    db_docs = await zm.db[collection].find({}).to_list(length=20)
    ids_in_db = set([d["_id"] for d in db_docs])
    for d in docs:
        assert d["_id"] in ids_in_db
    # Cached read (simulate cache check)
    cached_doc = await zm.find_document(collection, {"_id": docs[0]["_id"]})
    assert cached_doc.ok
    assert ObjectId(cached_doc.data["_id"]) == docs[0]["_id"]

@pytest.mark.asyncio
async def test_flush_buffered_inserts_actually_flushes(tmp_path):
    zm = ZMongo()
    collection = "test_flush_buffered"
    docs = [{"_id": ObjectId(), "val": i} for i in range(3)]
    await zm.delete_documents(collection)
    # Manually buffer
    for doc in docs:
        buffered_cache = zm._get_buffered_cache(collection)
        await buffered_cache.set(str(doc["_id"]), doc, buffer_only=True)
    # Docs not in DB yet
    db_docs = await zm.db[collection].find({}).to_list(length=20)
    assert not db_docs
    await zm.flush_buffered_inserts(collection)
    db_docs = await zm.db[collection].find({}).to_list(length=20)
    assert len(db_docs) == 3
    for d in docs:
        assert d["_id"] in [x["_id"] for x in db_docs]
