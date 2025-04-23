import pytest
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from zmongo_toolbag_dev import ZMongo

TEST_COLLECTION = "test_zmongo"

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def zmongo():
    zm = ZMongo()
    await zm.delete_all_documents(TEST_COLLECTION)
    yield zm
    await zm.delete_all_documents(TEST_COLLECTION)

@pytest.mark.asyncio
async def test_insert_and_find_document(zmongo):
    doc = {"name": "Alice", "age": 30}
    inserted = await zmongo.insert_document(TEST_COLLECTION, doc)
    assert inserted is not None
    assert "_id" in inserted

    found = await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    assert found is not None
    assert found["name"] == "Alice"

@pytest.mark.asyncio
async def test_update_document(zmongo):
    doc = {"name": "Bob", "age": 25}
    inserted = await zmongo.insert_document(TEST_COLLECTION, doc)
    await zmongo.update_document(TEST_COLLECTION, {"_id": inserted["_id"]}, {"$set": {"age": 26}})
    updated = await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    assert updated["age"] == 26

@pytest.mark.asyncio
async def test_delete_document(zmongo):
    doc = {"name": "Charlie"}
    inserted = await zmongo.insert_document(TEST_COLLECTION, doc)
    deleted = await zmongo.delete_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    assert deleted.deleted_count == 1
    assert await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]}) is None

@pytest.mark.asyncio
async def test_bulk_write(zmongo):
    ops = [
        {"operation": "insert", "document": {"name": "D1"}},
        {"operation": "insert", "document": {"name": "D2"}},
        {"operation": "update", "filter": {"name": "D1"}, "update": {"$set": {"updated": True}}},
        {"operation": "delete", "filter": {"name": "D2"}},
    ]
    result = await zmongo.bulk_write(TEST_COLLECTION, ops)
    assert result["inserted_count"] == 2
    assert result["matched_count"] == 1
    assert result["modified_count"] == 1
    assert result["deleted_count"] == 1

@pytest.mark.asyncio
async def test_get_field_names_and_sample(zmongo):
    await zmongo.insert_document(TEST_COLLECTION, {"x": 1, "y": 2})
    await zmongo.insert_document(TEST_COLLECTION, {"x": 3, "z": 4})
    fields = await zmongo.get_field_names(TEST_COLLECTION)
    assert set(fields).issuperset({"x", "y", "z"})
    samples = await zmongo.sample_documents(TEST_COLLECTION, 2)
    assert len(samples) == 2

@pytest.mark.asyncio
async def test_text_search(zmongo):
    await zmongo.insert_document(TEST_COLLECTION, {"content": "The quick brown fox"})
    await zmongo.insert_document(TEST_COLLECTION, {"content": "Jumps over the lazy dog"})
    try:
        await zmongo.db[TEST_COLLECTION].create_index([("content", "text")])
    except:
        pass
    results = await zmongo.text_search(TEST_COLLECTION, "quick")
    assert any("quick" in doc["content"] for doc in results)

@pytest.mark.asyncio
async def test_cache_hit(zmongo):
    doc = {"key": "value"}
    inserted = await zmongo.insert_document(TEST_COLLECTION, doc)
    _ = await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    hit = await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    assert hit == inserted

@pytest.mark.asyncio
async def test_clear_cache(zmongo):
    doc = {"key": "clearme"}
    inserted = await zmongo.insert_document(TEST_COLLECTION, doc)
    _ = await zmongo.find_document(TEST_COLLECTION, {"_id": inserted["_id"]})
    await zmongo.clear_cache()
    assert zmongo.cache == {}

@pytest.mark.asyncio
async def test_count_documents(zmongo):
    await zmongo.insert_document(TEST_COLLECTION, {"foo": "bar"})
    count = await zmongo.count_documents(TEST_COLLECTION)
    assert count >= 1

@pytest.mark.asyncio
async def test_list_collections(zmongo):
    await zmongo.insert_document(TEST_COLLECTION, {"sample": True})
    collections = await zmongo.list_collections()
    assert TEST_COLLECTION in collections
