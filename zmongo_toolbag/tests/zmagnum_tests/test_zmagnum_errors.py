import os
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from zmongo_toolbag.zmagnum import ZMagnum, UpdateResponse

@pytest_asyncio.fixture(scope="function")
async def zmagnum():
    os.environ["MONGO_URI"] = "mongodb://localhost:27017"
    os.environ["MONGO_DATABASE_NAME"] = "test_zmagnum_db"
    instance = ZMagnum(disable_cache=False)
    yield instance
    await instance.close()

@pytest.mark.asyncio
async def test_find_document_error_logged(zmagnum):
    with patch.object(zmagnum.db["fake_collection"], "find_one", side_effect=Exception("Simulated find error")):
        result = await zmagnum.find_document("fake_collection", {"bad": True})
        assert result is None

@pytest.mark.asyncio
async def test_recommend_indexes_error_logged(zmagnum):
    with patch.object(zmagnum.db["whatever"], "find", side_effect=Exception("Find error")):
        result = await zmagnum.recommend_indexes("whatever")
        assert isinstance(result, dict)
        assert result == {}

@pytest.mark.asyncio
async def test_update_document_catches_exception(zmagnum):
    with patch.object(zmagnum.db["stuff"], "update_one", side_effect=Exception("Bang update")):
        response = await zmagnum.update_document("stuff", {"x": 1}, {"$set": {"y": 2}})
        assert isinstance(response, UpdateResponse)
        assert response.matched_count == 0
        assert response.modified_count == 0

@pytest.mark.asyncio
async def test_close_handles_exception():
    zmag = ZMagnum(disable_cache=True)
    # MotorClient.close is a method descriptor and can't be mocked directly
    with patch.object(zmag, "mongo_client", create=True) as mock_client:
        mock_client.close.side_effect = Exception("Shutdown fail")
        await zmag.close()
    # This test passes if it does not raise.

def test_profile_error_logged():
    zmag = ZMagnum(disable_cache=True)
    def raise_error():
        raise ValueError("Oops profile")
    with pytest.raises(ValueError):
        zmag._profile("explosive", raise_error)
    zmag.mongo_client.close()
