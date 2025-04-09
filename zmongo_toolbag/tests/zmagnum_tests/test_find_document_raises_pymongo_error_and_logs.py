import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_find_document_raises_pymongo_error_and_logs(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    caplog.set_level("ERROR", logger="zmagnum")

    # Mock the find_one to raise PyMongoError
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(side_effect=PyMongoError("Simulated find_one error"))

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.find_document("test_collection", {"foo": "bar"})

    # Assert return value includes error
    assert isinstance(result, dict)
    assert "error" in result
    assert "Simulated find_one error" in result["error"]

    # Confirm error log entry exists
    assert any("MongoDB error in find_document: Simulated find_one error" in message for message in caplog.text.splitlines())
