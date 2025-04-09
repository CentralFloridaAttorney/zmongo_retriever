# Re-import necessary components after kernel reset
import pytest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_find_document_caches_result():
    zmag = ZMagnum(disable_cache=False)
    collection = "test_collection"
    query = {"_id": 1}
    expected_result = {"_id": 1, "field": "value"}

    mock_coll = AsyncMock()
    mock_coll.find_one.return_value = expected_result
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    # First call should hit the DB
    result = await zmag.find_document(collection, query)
    assert result == expected_result

    # Ensure cache was populated
    normalized = zmag._normalize_collection_name(collection)
    cache_key = zmag._generate_cache_key(query)
    assert cache_key in zmag.cache[normalized]

    # Second call should hit the cache
    mock_coll.find_one.reset_mock()
    cached_result = await zmag.find_document(collection, query)
    assert cached_result == expected_result
    mock_coll.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_insert_documents_success():
    zmag = ZMagnum(disable_cache=False)
    collection = "insert_collection"
    docs = [{"name": "A"}, {"name": "B"}]
    inserted_ids = [1, 2]

    mock_result = MagicMock()
    mock_result.inserted_ids = inserted_ids

    mock_coll = AsyncMock()
    mock_coll.insert_many.return_value = mock_result
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.insert_documents(collection, docs)
    assert result["inserted_count"] == 2

    normalized = zmag._normalize_collection_name(collection)
    for _id in inserted_ids:
        cache_key = zmag._generate_cache_key({"_id": str(_id)})
        assert cache_key in zmag.cache[normalized]
