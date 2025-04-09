
import logging

logger = logging.getLogger("zmagnum")

import pytest
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmagnum import ZMagnum
import logging

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_delete_all_documents_success(caplog):
    zmag = ZMagnum(disable_cache=True)
    collection = "test_collection"

    # Reset logger level back to INFO for testing
    logging.getLogger("zmagnum").setLevel(logging.INFO)

    mock_result = MagicMock()
    mock_result.deleted_count = 7

    mock_collection = AsyncMock()
    mock_collection.delete_many.return_value = mock_result

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    caplog.clear()
    with caplog.at_level(logging.INFO):
        deleted_count = await zmag.delete_all_documents(collection)

    assert deleted_count == 7
    assert f"Deleted {deleted_count} documents from '{collection}'" in caplog.text


@pytest.mark.asyncio
async def test_delete_all_documents_handles_pymongo_error(caplog):
    zmag = ZMagnum(disable_cache=True)
    collection = "test_collection"

    mock_collection = AsyncMock()
    mock_collection.delete_many.side_effect = Exception("simulated failure")

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    with caplog.at_level(logging.ERROR):
        deleted_count = await zmag.delete_all_documents(collection)

    assert deleted_count == 0
    assert "Unexpected error in delete_all_documents" in caplog.text
