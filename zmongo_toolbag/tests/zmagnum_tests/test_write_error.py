import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import BulkWriteError, PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_insert_documents_bulk_write_error():
    zmagnum = ZMagnum(disable_cache=True)

    mock_collection = MagicMock()
    mock_collection.insert_many = AsyncMock(
        side_effect=BulkWriteError({"writeErrors": [{"errmsg": "Duplicate key"}]})
    )

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    docs = [{"_id": 1, "name": "Alice"}]
    result = await zmagnum.insert_documents("some_collection", docs)
    assert isinstance(result, dict)
    assert "error" in result
    assert "writeErrors" in result["error"]


@pytest.mark.asyncio
async def test_insert_documents_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)

    mock_collection = MagicMock()
    mock_collection.insert_many = AsyncMock(
        side_effect=PyMongoError("Connection lost")
    )

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    docs = [{"_id": 2, "name": "Bob"}]
    result = await zmagnum.insert_documents("some_collection", docs)
    assert isinstance(result, dict)
    assert "error" in result
    assert "Connection lost" in result["error"]
