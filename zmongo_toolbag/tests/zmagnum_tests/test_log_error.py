import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_find_document_pymongo_error_logs_and_returns_error(caplog):
    zmag = ZMagnum(disable_cache=True)
    mock_collection = AsyncMock()
    mock_collection.find_one.side_effect = PyMongoError("Database unreachable")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    with caplog.at_level("ERROR"):
        result = await zmag.find_document("test_coll", {"field": "value"})

    assert "MongoDB error in find_document: Database unreachable" in caplog.text
    assert isinstance(result, dict)
    assert "error" in result
    assert "Database unreachable" in result["error"]


@pytest.mark.asyncio
async def test_find_document_generic_error_logs_and_returns_none(caplog):
    zmag = ZMagnum(disable_cache=True)
    mock_collection = AsyncMock()
    mock_collection.find_one.side_effect = RuntimeError("Unexpected crash")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    with caplog.at_level("ERROR"):
        result = await zmag.find_document("test_coll", {"field": "value"})

    assert "Error in find_document: Unexpected crash" in caplog.text
    assert result is None
