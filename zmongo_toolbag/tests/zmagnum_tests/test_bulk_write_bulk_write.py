import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo import UpdateOne, InsertOne
from pymongo.errors import BulkWriteError, PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_bulk_write_success():
    zmag = ZMagnum(disable_cache=True)
    collection = "test_coll"
    operations = [
        InsertOne({"_id": 1, "field": "val"}),
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

    mock_coll = AsyncMock()
    mock_coll.bulk_write.return_value = mock_result
    mock_coll.find_one = AsyncMock(return_value={"_id": 1, "field": "updated"})

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)

    assert result == {
        "inserted_count": 1,
        "matched_count": 1,
        "modified_count": 1,
        "deleted_count": 0,
        "upserted_count": 0,
        "acknowledged": True,
    }


@pytest.mark.asyncio
async def test_bulk_write_empty_operations():
    zmag = ZMagnum()
    result = await zmag.bulk_write("test", [])
    assert result == {
        "inserted_count": 0,
        "matched_count": 0,
        "modified_count": 0,
        "deleted_count": 0,
        "upserted_count": 0,
        "acknowledged": True,
    }


@pytest.mark.asyncio
async def test_bulk_write_bulk_write_error():
    zmag = ZMagnum()
    collection = "test"
    operations = [InsertOne({"_id": 1})]

    mock_coll = AsyncMock()
    mock_coll.bulk_write.side_effect = BulkWriteError({"writeErrors": [{"errmsg": "Duplicate key"}]})

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)
    assert "error" in result
    assert "writeErrors" in result["error"]


@pytest.mark.asyncio
async def test_bulk_write_pymongo_error():
    zmag = ZMagnum()
    collection = "test"
    operations = [InsertOne({"_id": 1})]

    mock_coll = AsyncMock()
    mock_coll.bulk_write.side_effect = PyMongoError("Some Mongo error")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)
    assert "error" in result
    assert "Some Mongo error" in result["error"]


@pytest.mark.asyncio
async def test_bulk_write_generic_error():
    zmag = ZMagnum()
    collection = "test"
    operations = [InsertOne({"_id": 1})]

    mock_coll = AsyncMock()
    mock_coll.bulk_write.side_effect = ValueError("Something broke")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)
    assert "error" in result
    assert "Something broke" in result["error"]


@pytest.mark.asyncio
async def test_bulk_write_cache_skips_broken_insert():
    zmag = ZMagnum(disable_cache=False)
    collection = "test"
    operations = [InsertOne({"field": "no_id"})]

    mock_result = MagicMock(
        inserted_count=1,
        matched_count=0,
        modified_count=0,
        deleted_count=0,
        upserted_count=0,
        acknowledged=True,
    )

    mock_coll = AsyncMock()
    mock_coll.bulk_write.return_value = mock_result
    mock_coll.find_one = AsyncMock(return_value={})

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)
    assert result["inserted_count"] == 1


@pytest.mark.asyncio
async def test_bulk_write_update_caching_exception_handled():
    zmag = ZMagnum(disable_cache=False)
    collection = "test"
    operations = [UpdateOne({"_id": 123}, {"$set": {"field": "val"}})]

    mock_result = MagicMock(
        inserted_count=0,
        matched_count=1,
        modified_count=1,
        deleted_count=0,
        upserted_count=0,
        acknowledged=True,
    )

    mock_coll = AsyncMock()
    mock_coll.bulk_write.return_value = mock_result
    mock_coll.find_one = AsyncMock(side_effect=RuntimeError("fail read"))

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.bulk_write(collection, operations)
    assert result["matched_count"] == 1


@pytest.mark.asyncio
async def test_bulk_write_collection_name_normalized():
    zmag = ZMagnum(disable_cache=True)
    collection = "TEST_Collection "
    normalized = "test_collection"
    operations = [InsertOne({"_id": 1})]

    mock_result = MagicMock(
        inserted_count=1,
        matched_count=0,
        modified_count=0,
        deleted_count=0,
        upserted_count=0,
        acknowledged=True
    )

    mock_coll = AsyncMock()
    mock_coll.bulk_write.return_value = mock_result

    zmag.db = MagicMock()
    zmag.db.__getitem__.side_effect = lambda name: mock_coll if name == normalized else None

    result = await zmag.bulk_write(collection, operations)
    assert result["inserted_count"] == 1
