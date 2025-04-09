import pytest
from unittest.mock import MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

def failing_mongo_op():
    raise PyMongoError("Simulated Mongo failure")

def test_profile_raises_pymongo_error_and_logs(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    caplog.set_level("ERROR", logger="zmagnum")

    with pytest.raises(PyMongoError, match="Simulated Mongo failure"):
        zmagnum._profile("test_operation", failing_mongo_op)

    # Verify error message was logged
    assert any("[PROFILE][test_operation] Mongo error: Simulated Mongo failure" in message for message in caplog.text.splitlines())
