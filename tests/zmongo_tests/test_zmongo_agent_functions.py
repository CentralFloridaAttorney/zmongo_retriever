import pytest
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from zmongo_toolbag.zmongo import ZMongo
from bson.objectid import ObjectId


@pytest.mark.asyncio
async def test_get_field_names():
    zmongo = ZMongo()
    test_collection = "test_get_fields"
    await zmongo.db[test_collection].insert_many([
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "phone": "123456789"}
    ])
    fields = await zmongo.get_field_names(test_collection)
    assert "name" in fields
    assert "email" in fields or "phone" in fields
    await zmongo.db[test_collection].drop()


@pytest.mark.asyncio
async def test_sample_documents():
    zmongo = ZMongo()
    test_collection = "test_sample"
    await zmongo.db[test_collection].insert_many([
        {"x": i} for i in range(10)
    ])
    samples = await zmongo.sample_documents(test_collection, sample_size=3)
    assert isinstance(samples, list)
    assert len(samples) <= 3
    await zmongo.db[test_collection].drop()


@pytest.mark.asyncio
async def test_count_documents():
    zmongo = ZMongo()
    test_collection = "test_count"
    await zmongo.db[test_collection].insert_many([
        {"value": i} for i in range(5)
    ])
    count = await zmongo.count_documents(test_collection)
    assert count >= 5
    await zmongo.db[test_collection].drop()


@pytest.mark.asyncio
async def test_get_document_by_id():
    zmongo = ZMongo()
    test_collection = "test_get_by_id"
    doc = {"name": "TestDoc"}
    insert_result = await zmongo.db[test_collection].insert_one(doc)
    retrieved = await zmongo.get_document_by_id(test_collection, insert_result.inserted_id)
    assert retrieved is not None
    assert retrieved["name"] == "TestDoc"
    await zmongo.db[test_collection].drop()


@pytest.mark.asyncio
async def test_text_search():
    zmongo = ZMongo()
    test_collection = "test_text_search"
    await zmongo.db[test_collection].insert_many([
        {"title": "Foreclosure document regarding lost note"},
        {"title": "General client notes"},
        {"title": "Foreclosure notes and process"}
    ])
    # Create text index
    await zmongo.db[test_collection].create_index([("title", "text")])

    results = await zmongo.text_search(test_collection, "foreclosure")
    assert any("Foreclosure" in doc["title"] for doc in results)
    await zmongo.db[test_collection].drop()
