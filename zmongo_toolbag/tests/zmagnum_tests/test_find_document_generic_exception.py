import pytest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_find_document_generic_exception(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    caplog.set_level("ERROR", logger="zmagnum")

    # Simulate a non-PyMongo exception inside find_one
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(side_effect=TypeError("Unexpected type in find_one"))

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.find_document("test_collection", {"invalid": object()})  # Cause potential serialization error

    assert result is None
    assert any("Error in find_document: Unexpected type in find_one" in message for message in caplog.text.splitlines())
