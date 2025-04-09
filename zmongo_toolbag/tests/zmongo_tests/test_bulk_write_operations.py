import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_bulk_write_successful_insert_and_update():
    zmag = ZMagnum(disable_cache=False)

    collection_name = "test_collection"
    operations = [
        InsertOne({"_id": 1, "field": "value"}),
        UpdateOne({"_id": 1}, {"$set": {"field": "updated"}}),
    ]

    mock_result = MagicMock(
        inserted_count=1,
        matched_count=1,
        modified_count=1,
        deleted_count=0,
        upserted_count=0,
    )

    # Patch db interactions
    zmag.db = MagicMock()
    mock_collection = AsyncMock()
    zmag.db.__getitem__.return_value = mock_collection
    mock_collection.bulk_write.return_value = mock_result
    mock_collection.find_one = AsyncMock(return_value={"_id": 1, "field": "updated"})

    result = await zmag.bulk_write(collection_name, operations)

    assert result["inserted_count"] == 1
    assert result["matched_count"] == 1
    assert result["modified_count"] == 1
    assert result["deleted_count"] == 0
    assert result["upserted_count"] == 0
@pytest.mark.asyncio
async def test_bulk_write_returns_message_if_no_ops():
    zmag = ZMagnum(disable_cache=True)
    result = await zmag.bulk_write("test_collection", [])
    assert result == {
        "inserted_count": 0,
        "matched_count": 0,
        "modified_count": 0,
        "deleted_count": 0,
        "upserted_count": 0,
        "acknowledged": True,
    }

@pytest.mark.asyncio
async def test_bulk_write_bulk_write_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = BulkWriteError({"writeErrors": ["simulated error"]})
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.bulk_write(collection_name, operations)
    assert "error" in result
    assert "writeErrors" in result["error"]

@pytest.mark.asyncio
async def test_bulk_write_pymongo_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = PyMongoError("Simulated PyMongoError")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.bulk_write(collection_name, operations)
    assert "error" in result
    assert "Simulated PyMongoError" in result["error"]

@pytest.mark.asyncio
async def test_bulk_write_generic_exception_logged():
    zmag = ZMagnum(disable_cache=True)
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = Exception("Unexpected boom")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.bulk_write(collection_name, operations)
    assert "error" in result
    assert "Unexpected boom" in result["error"]
