import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo import UpdateOne
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_bulk_write_updateone_caches_result():
    # Given
    zmag = ZMagnum(disable_cache=False)
    collection = "Test_Collection"
    normalized = "test_collection"

    # Operation: simulate an update
    query_filter = {"_id": 123}
    update = {"$set": {"name": "updated"}}
    update_op = UpdateOne(query_filter, update)

    # Mock document to return from find_one after update
    updated_doc = {"_id": 123, "name": "updated"}

    # Create mock collection
    mock_collection = AsyncMock()
    mock_collection.bulk_write.return_value = MagicMock(
        matched_count=1,
        modified_count=1,
        deleted_count=0,
        inserted_count=0,
        upserted_count=0,
        acknowledged=True,
    )
    mock_collection.find_one = AsyncMock(return_value=updated_doc)

    # Patch db[normalized] to return our mock
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    # When
    result = await zmag.bulk_write(collection, [update_op])

    # Then
    cache_key = ZMagnum._generate_cache_key(query_filter)
    assert result["matched_count"] == 1
    assert result["modified_count"] == 1
    assert normalized in zmag.cache
    assert cache_key in zmag.cache[normalized]
    assert zmag.cache[normalized][cache_key]["name"] == "updated"
