import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum, UpdateResponse


@pytest.mark.asyncio
async def test_update_document_success():
    zmagnum = ZMagnum(disable_cache=True)

    mock_update_result = MagicMock(matched_count=1, modified_count=1, upserted_id=None)
    mock_collection = MagicMock()
    mock_collection.update_one = AsyncMock(return_value=mock_update_result)
    mock_collection.find_one = AsyncMock(return_value={"_id": "abc123", "name": "Updated"})

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    response = await zmagnum.update_document("some_collection", {"_id": "abc123"}, {"$set": {"name": "Updated"}})
    assert isinstance(response, UpdateResponse)
    assert response.matched_count == 1
    assert response.modified_count == 1
    assert response.upserted_id is None


@pytest.mark.asyncio
async def test_update_document_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)

    mock_collection = MagicMock()
    mock_collection.update_one = AsyncMock(side_effect=PyMongoError("Update failed"))

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    response = await zmagnum.update_document("some_collection", {"_id": "fail"}, {"$set": {"x": 1}})
    assert isinstance(response, UpdateResponse)
    assert response.matched_count == 0
    assert response.modified_count == 0
    assert response.upserted_id is None


@pytest.mark.asyncio
async def test_update_document_generic_exception():
    zmagnum = ZMagnum(disable_cache=True)

    mock_collection = MagicMock()
    mock_collection.update_one = AsyncMock(side_effect=RuntimeError("boom"))

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    response = await zmagnum.update_document("some_collection", {"_id": "fail"}, {"$set": {"x": 1}})
    assert isinstance(response, UpdateResponse)
    assert response.matched_count == 0
    assert response.modified_count == 0
    assert response.upserted_id is None
