import asyncio
import os
import pytest
import pytest_asyncio
import logging
from unittest.mock import Mock, patch

# Project root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from zmongo_toolbag.zmagnum import ZMagnum, TTLCache, UpdateResponse

@pytest_asyncio.fixture(scope="function")
async def zmagnum():
    os.environ["MONGO_URI"] = "mongodb://localhost:27017"
    os.environ["MONGO_DATABASE_NAME"] = "test_zmagnum_db"
    instance = ZMagnum(disable_cache=True)
    yield instance
    await instance.close()

def test_cache_expiry_behavior():
    cache = TTLCache(ttl=0)  # 0 second TTL for instant expiry
    cache['foo'] = 'bar'
    import time
    time.sleep(0.01)  # give enough time for TTL to expire
    with pytest.raises(KeyError):
        _ = cache['foo']

def test_cache_clear():
    cache = TTLCache(ttl=300)
    cache['key1'] = 'value1'
    cache.clear()
    assert len(cache) == 0
    assert len(cache.timestamps) == 0

def test_disable_cache_sets_warning_level():
    with patch("logging.Logger.setLevel") as mock_set_level:
        ZMagnum(disable_cache=True)
        mock_set_level.assert_called_with(logging.WARNING)

def test_profile_logs_time():
    zmag = ZMagnum(disable_cache=True)
    mock_func = Mock(return_value=42)
    result = zmag._profile("mock_test", mock_func)
    assert result == 42
    zmag.mongo_client.close()

@pytest.mark.asyncio
async def test_insert_documents_empty(zmagnum):
    result = await zmagnum.insert_documents("test_collection", [])
    assert result == {"inserted_count": 0}

@pytest.mark.asyncio
async def test_update_document_error(zmagnum):
    with patch.object(zmagnum.db["test_collection"], "update_one", side_effect=Exception("Boom")):
        response = await zmagnum.update_document("test_collection", {"fail": True}, {"$set": {"x": 1}})
        assert isinstance(response, UpdateResponse)
        assert response.matched_count == 0
        assert response.modified_count == 0
