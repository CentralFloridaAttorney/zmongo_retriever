import pytest
import logging
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_delete_all_documents_handles_pymongo_error(caplog):
    zmag = ZMagnum(disable_cache=True)
    logging.getLogger("zmagnum").setLevel(logging.INFO)

    collection = "test_collection"

    mock_collection = AsyncMock()
    mock_collection.delete_many.side_effect = PyMongoError("Simulated pymongo failure")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        deleted_count = await zmag.delete_all_documents(collection)

    assert deleted_count == 0
    assert "Mongo delete_all_documents failed: Simulated pymongo failure" in caplog.text
