import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from zmongo_toolbag.zmongo import ZMongo

@pytest.mark.asyncio
async def test_bulk_write_successful_insert_and_update():
    zmongo = ZMongo()
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
        acknowledged=True
    )

    mock_collection = AsyncMock()
    mock_collection.bulk_write.return_value = mock_result

    zmongo.db = MagicMock()
    zmongo.db.__getitem__.return_value = mock_collection

    result = await zmongo.bulk_write(collection_name, operations)

    assert result["inserted_count"] == 1
    assert result["matched_count"] == 1
    assert result["modified_count"] == 1
    assert result["deleted_count"] == 0
    assert result["upserted_count"] == 0
    assert result["acknowledged"] is True

@pytest.mark.asyncio
async def test_bulk_write_returns_message_if_no_ops():
    zmongo = ZMongo()
    result = await zmongo.bulk_write("test_collection", [])
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
    zmongo = ZMongo()
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = BulkWriteError({"writeErrors": ["simulated error"]})

    zmongo.db = MagicMock()
    zmongo.db.__getitem__.return_value = mock_collection

    result = await zmongo.bulk_write(collection_name, operations)
    assert "error" in result
    assert "writeErrors" in result["error"]

@pytest.mark.asyncio
async def test_bulk_write_pymongo_error_logged():
    zmongo = ZMongo()
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = PyMongoError("Simulated PyMongoError")

    zmongo.db = MagicMock()
    zmongo.db.__getitem__.return_value = mock_collection

    result = await zmongo.bulk_write(collection_name, operations)
    assert "error" in result
    assert "Simulated PyMongoError" in result["error"]

@pytest.mark.asyncio
async def test_bulk_write_generic_exception_logged():
    zmongo = ZMongo()
    collection_name = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_collection = AsyncMock()
    mock_collection.bulk_write.side_effect = Exception("Unexpected boom")

    zmongo.db = MagicMock()
    zmongo.db.__getitem__.return_value = mock_collection

    result = await zmongo.bulk_write(collection_name, operations)
    assert "error" in result
    assert "Unexpected boom" in result["error"]
