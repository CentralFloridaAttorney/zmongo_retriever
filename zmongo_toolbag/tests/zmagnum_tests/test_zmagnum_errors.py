import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo.errors import BulkWriteError, PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_insert_documents_bulk_write_error():
    zmag = ZMagnum(disable_cache=True)

    mock_coll = AsyncMock()
    mock_coll.insert_many.side_effect = BulkWriteError({"writeErrors": [{"errmsg": "Duplicate key"}]})

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.insert_documents("test_collection", [{"_id": 1}, {"_id": 1}])
    assert "error" in result
    assert "writeErrors" in result["error"]

@pytest.mark.asyncio
async def test_insert_documents_pymongo_error():
    zmag = ZMagnum(disable_cache=True)

    mock_coll = AsyncMock()
    mock_coll.insert_many.side_effect = PyMongoError("Some Mongo issue")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.insert_documents("test_collection", [{"_id": 2}])
    assert "error" in result
    assert "Some Mongo issue" in result["error"]

@pytest.mark.asyncio
async def test_insert_documents_generic_error():
    zmag = ZMagnum(disable_cache=True)

    mock_coll = AsyncMock()
    mock_coll.insert_many.side_effect = ValueError("Some unexpected error")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_coll

    result = await zmag.insert_documents("test_collection", [{"_id": 3}])
    assert "error" in result
    assert "Some unexpected error" in result["error"]
