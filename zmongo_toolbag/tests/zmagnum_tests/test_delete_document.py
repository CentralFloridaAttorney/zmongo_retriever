# test_delete_document.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_delete_document_success_and_cache():
    zmag = ZMagnum(disable_cache=False)
    collection = "test_collection"
    query = {"_id": 1}

    # Add item to cache
    normalized = zmag._normalize_collection_name(collection)
    cache_key = zmag._generate_cache_key(query)
    zmag.cache[normalized][cache_key] = {"_id": 1, "value": 42}

    mock_result = MagicMock(deleted_count=1)
    mock_collection = AsyncMock()
    mock_collection.delete_one.return_value = mock_result

    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.delete_document(collection, query)

    assert result == {"deleted_count": 1}
    assert cache_key not in zmag.cache[normalized]  # cache should be invalidated


@pytest.mark.asyncio
async def test_delete_document_pymongo_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection = "test_collection"
    query = {"_id": 2}

    mock_collection = AsyncMock()
    mock_collection.delete_one.side_effect = PyMongoError("simulated pymongo error")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.delete_document(collection, query)
    assert "error" in result
    assert "simulated pymongo error" in result["error"]


@pytest.mark.asyncio
async def test_delete_document_generic_error_logged():
    zmag = ZMagnum(disable_cache=True)
    collection = "test_collection"
    query = {"_id": 3}

    mock_collection = AsyncMock()
    mock_collection.delete_one.side_effect = Exception("unexpected failure")
    zmag.db = MagicMock()
    zmag.db.__getitem__.return_value = mock_collection

    result = await zmag.delete_document(collection, query)
    assert "error" in result
    assert "unexpected failure" in result["error"]
