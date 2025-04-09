# test_update_document.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_update_document_success_and_cache():
    zmag = ZMagnum(disable_cache=False)
    collection = "test_col"
    query = {"_id": 1}
    update_data = {"$set": {"value": 42}}

    # Mock result from update_one
    mock_update_result = MagicMock(matched_count=1, modified_count=1, upserted_id=None)

    # Mock updated document from find_one
    updated_doc = {"_id": 1, "value": 42}
    mock_collection = AsyncMock()
    mock_collection.update_one.return_value = mock_update_result
    mock_collection.find_one.return_value = updated_doc

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.update_document(collection, query, update_data)

    assert result == {
        "matched_count": 1,
        "modified_count": 1,
        "upserted_id": None,
    }

    normalized = zmag._normalize_collection_name(collection)
    cache_key = zmag._generate_cache_key(query)
    assert cache_key in zmag.cache[normalized]
    assert zmag.cache[normalized][cache_key]["value"] == 42

@pytest.mark.asyncio
async def test_update_document_pymongo_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection = "test_col"
    query = {"_id": 1}
    update_data = {"$set": {"value": 42}}

    mock_collection = AsyncMock()
    mock_collection.update_one.side_effect = PyMongoError("Simulated Mongo failure")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.update_document(collection, query, update_data)
    assert "error" in result
    assert "Simulated Mongo failure" in result["error"]

@pytest.mark.asyncio
async def test_update_document_generic_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection = "test_col"
    query = {"_id": 1}
    update_data = {"$set": {"value": 42}}

    mock_collection = AsyncMock()
    mock_collection.update_one.side_effect = Exception("Unexpected failure")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.update_document(collection, query, update_data)
    assert "error" in result
    assert "Unexpected failure" in result["error"]
