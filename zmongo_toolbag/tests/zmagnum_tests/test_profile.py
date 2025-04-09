import logging

import pytest
import time
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


def successful_function(x):
    time.sleep(0.01)  # simulate brief workload
    return x * 2


def raises_pymongo():
    raise PyMongoError("Mongo is down")


def raises_generic():
    raise ValueError("Something else broke")


def test_profile_success_logs_and_returns(caplog):
    zmag = ZMagnum(disable_cache=True)

    # 🔧 Restore logger level so INFO logs from _profile are captured
    logger = logging.getLogger("zmagnum")
    logger.setLevel(logging.INFO)

    with caplog.at_level("INFO"):
        result = zmag._profile("test_success", successful_function, 5)

    assert result == 10
    assert "[PROFILE] test_success took" in caplog.text


def test_profile_pymongo_error_logs_and_raises(caplog):
    zmag = ZMagnum(disable_cache=True)
    with caplog.at_level("ERROR"), pytest.raises(PyMongoError) as excinfo:
        zmag._profile("mongo_fail", raises_pymongo)

    assert "Mongo error: Mongo is down" in caplog.text
    assert "[PROFILE][mongo_fail]" in caplog.text
    assert "Mongo is down" in str(excinfo.value)


def test_profile_generic_error_logs_and_raises(caplog):
    zmag = ZMagnum(disable_cache=True)
    with caplog.at_level("ERROR"), pytest.raises(ValueError) as excinfo:
        zmag._profile("generic_fail", raises_generic)

    assert "[PROFILE][generic_fail] error: Something else broke" in caplog.text
    assert "Something else broke" in str(excinfo.value)
