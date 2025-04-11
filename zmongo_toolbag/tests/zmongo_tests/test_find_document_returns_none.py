# test_find_document_cache_hit_branch.py

import pytest
from unittest.mock import AsyncMock
from zmongo_toolbag.zmongo import ZMongo


@pytest.mark.asyncio
async def test_find_document_returns_from_cache_branch(monkeypatch):
    zmongo = ZMongo()
    collection = "test_cache_branch"
    query = {"_id": "cached-id"}

    # Generate normalized key and cache key
    normalized = zmongo._normalize_collection_name(collection)
    cache_key = zmongo._generate_cache_key(query)

    # Expected document to cache
    expected = {"_id": "cached-id", "name": "Cached"}

    # Populate cache
    zmongo.cache[normalized][cache_key] = expected

    # Monkeypatch db[collection] to fail if accessed
    class FailOnAccess:
        async def find_one(self, *args, **kwargs):
            raise RuntimeError("Database should not be accessed if cache hit works")

    class MockDB:
        def __getitem__(self, name):
            return FailOnAccess()

    monkeypatch.setattr(zmongo, "db", MockDB())

    result = await zmongo.find_document(collection, query)

    assert result == expected
