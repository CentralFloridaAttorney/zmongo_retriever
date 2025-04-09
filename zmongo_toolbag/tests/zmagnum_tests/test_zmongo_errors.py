import asyncio
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
        assert result == {}

@pytest.mark.asyncio
async def test_analyze_embedding_schema_error(zmagnum):
    with patch.object(zmagnum.db["weird"], "find", new_callable=MagicMock) as mock_find:
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_find.return_value = mock_cursor
        result = await zmagnum.analyze_embedding_schema("weird")
        assert result["error"] == "No embeddings found."

@pytest.mark.asyncio
async def test_insert_documents_exception_is_logged(zmagnum):
    with patch.object(zmagnum.db["test"], "insert_many", side_effect=Exception("Boom insert")):
        result = await zmagnum.insert_documents("test", [])
        assert result["inserted_count"] == 0

@pytest.mark.asyncio
async def test_update_document_catches_exception(zmagnum):
    with patch.object(zmagnum.db["stuff"], "update_one", side_effect=Exception("Bang update")):
        response = await zmagnum.update_document("stuff", {"x": 1}, {"$set": {"y": 2}})
        assert isinstance(response, UpdateResponse)
        assert response.matched_count == 0
        assert response.modified_count == 0

def test_is_sharded_cluster_failure():
    zmag = ZMagnum(disable_cache=True)
    with patch.object(zmag.sync_client.admin, "command", side_effect=Exception("Not a cluster")):
        result = zmag.is_sharded_cluster()
        assert result is False
    zmag.mongo_client.close()

def test_profile_error_logged():
    zmag = ZMagnum(disable_cache=True)
    def raise_error():
        raise ValueError("Oops profile")
    with pytest.raises(ValueError):
        zmag._profile("explosive", raise_error)
    zmag.mongo_client.close()
