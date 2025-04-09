import pytest
from unittest.mock import MagicMock
from zmongo_toolbag.zmagnum import ZMagnum
import logging

@pytest.mark.asyncio
async def test_close_success_logs_info(caplog):
    zmag = ZMagnum(disable_cache=True)
    zmag.mongo_client = MagicMock()

    # Temporarily raise log level to capture INFO messages
    logger = logging.getLogger("zmagnum")
    prev_level = logger.level
    logger.setLevel(logging.INFO)

    with caplog.at_level("INFO"):
        await zmag.close()

    logger.setLevel(prev_level)  # Reset logger level after test

    zmag.mongo_client.close.assert_called_once()
    assert "MongoDB connection closed." in caplog.text
