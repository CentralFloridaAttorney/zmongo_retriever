import pytest
from unittest.mock import AsyncMock
from zmongo_toolbag.zmongo import ZMongo


@pytest.mark.asyncio
async def test_find_document_real():
    zmongo = ZMongo()
    test_collection = "test_find_document"
    test_doc = {"_id": "find-doc-id", "name": "Alice", "role": "admin"}

    await zmongo.db[test_collection].delete_many({})
    await zmongo.db[test_collection].insert_one(test_doc)

    # Should hit database
    result = await zmongo.find_document(test_collection, {"_id": "find-doc-id"})

    assert result is not None
    assert result["_id"] == "find-doc-id"
    assert result["name"] == "Alice"
    assert result["role"] == "admin"

    # Should hit cache now
    cached = await zmongo.find_document(test_collection, {"_id": "find-doc-id"})
    assert cached == result

    await zmongo.db[test_collection].delete_many({})


@pytest.mark.asyncio
async def test_find_document_inserts_into_cache_and_returns_serialized():
    zmongo = ZMongo()
    collection = "test_find_document"
    doc_id = "unique-doc-123"
    test_query = {"_id": doc_id}
    test_doc = {"_id": doc_id, "name": "Test User", "role": "engineer"}

    await zmongo.db[collection].delete_many({})
    await zmongo.db[collection].insert_one(test_doc)

    normalized = zmongo._normalize_collection_name(collection)
    cache_key = zmongo._generate_cache_key(test_query)
    assert cache_key not in zmongo.cache[normalized]

    result = await zmongo.find_document(collection, test_query)

    assert result is not None
    assert isinstance(result, dict)
    assert "$oid" in result["_id"] or result["_id"] == doc_id

    assert cache_key in zmongo.cache[normalized]
    assert zmongo.cache[normalized][cache_key] == result

    cached = await zmongo.find_document(collection, test_query)
    assert cached == result

    await zmongo.db[collection].delete_many({})


@pytest.mark.asyncio
async def test_find_document_cache_miss(monkeypatch):
    zmongo = ZMongo()
    collection = "test_collection"
    query = {"_id": "abc123"}

    normalized = zmongo._normalize_collection_name(collection)
    cache_key = zmongo._generate_cache_key(query)
    assert cache_key not in zmongo.cache[normalized]

    mock_doc = {"_id": "abc123", "name": "Test"}

    class MockCollection:
        async def find_one(self, filter):
            assert filter == query
            return mock_doc

    class MockDB:
        def __getitem__(self, name):
            assert name == collection
            return MockCollection()

    monkeypatch.setattr(zmongo, "db", MockDB())

    result = await zmongo.find_document(collection, query)

    expected = zmongo.serialize_document(mock_doc)
    assert result == expected
    assert zmongo.cache[normalized][cache_key] == expected


@pytest.mark.asyncio
async def test_find_document_cache_miss_hits_find_one():
    zmongo = ZMongo()
    collection = "test_collection"
    query = {"_id": "abc123"}
    mock_doc = {"_id": "abc123", "name": "Test"}

    # Insert real document
    await zmongo.db[collection].delete_many({})
    await zmongo.db[collection].insert_one(mock_doc)

    normalized = zmongo._normalize_collection_name(collection)
    cache_key = zmongo._generate_cache_key(query)
    assert cache_key not in zmongo.cache[normalized]

    result = await zmongo.find_document(collection, query)

    expected_serialized = zmongo.serialize_document(mock_doc)
    assert result == expected_serialized
    assert zmongo.cache[normalized][cache_key] == expected_serialized

    await zmongo.db[collection].delete_many({})
