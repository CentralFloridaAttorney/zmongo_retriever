# test_find_document_cache_hit_only.py

import pytest
from zmongo_toolbag.zmongo import ZMongo


@pytest.mark.asyncio
async def test_find_document_cache_hit_returns_early():
    zmongo = ZMongo()
    collection = "test_cache_hit"
    query = {"_id": "cache-hit-id"}

    # Generate keys
    normalized = zmongo._normalize_collection_name(collection)
    cache_key = zmongo._generate_cache_key(query)

    # Inject into cache manually (bypassing any DB call)
    expected = {"_id": "cache-hit-id", "name": "Cached Result"}
    zmongo.cache[normalized][cache_key] = expected

    # This should return immediately from cache
    result = await zmongo.find_document(collection, query)

    # Confirm cache short-circuit works
    assert result == expected
