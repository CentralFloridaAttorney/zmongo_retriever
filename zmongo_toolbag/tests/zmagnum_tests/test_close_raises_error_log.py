# test_zmagnum_close_error.py

import pytest
from unittest.mock import MagicMock, patch
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_close_raises_and_logs_error(caplog):
    zmag = ZMagnum(disable_cache=True)

    # Simulate exception when calling mongo_client.close()
    broken_client = MagicMock()
    broken_client.close.side_effect = Exception("Boom")
    zmag.mongo_client = broken_client

    with caplog.at_level("ERROR"):
        await zmag.close()

    assert "Error closing MongoDB connection: Boom" in caplog.text
