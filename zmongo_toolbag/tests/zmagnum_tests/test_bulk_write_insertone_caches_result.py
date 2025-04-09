import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo import InsertOne
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_bulk_write_insertone_caches_result():
    # Given
    zmag = ZMagnum(disable_cache=False)
    collection = "My_Collection"
    normalized = "my_collection"

    # InsertOne with explicit _id
    inserted_doc = {"_id": 101, "name": "Cached Insert"}
    insert_op = InsertOne(inserted_doc)

    # Mock result with no errors, simulating a successful insert
    mock_result = MagicMock(
        matched_count=0,
        modified_count=0,
        deleted_count=0,
        inserted_count=1,
        upserted_count=0,
        acknowledged=True,
    )

    # Patch db[normalized]
    mock_collection = AsyncMock()
    mock_collection.bulk_write.return_value = mock_result
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    # When
    result = await zmag.bulk_write(collection, [insert_op])

    # Then
    cache_key = ZMagnum._generate_cache_key({"_id": str(inserted_doc["_id"])})
    assert result["inserted_count"] == 1
    assert normalized in zmag.cache
    assert cache_key in zmag.cache[normalized]
    assert zmag.cache[normalized][cache_key]["name"] == "Cached Insert"
