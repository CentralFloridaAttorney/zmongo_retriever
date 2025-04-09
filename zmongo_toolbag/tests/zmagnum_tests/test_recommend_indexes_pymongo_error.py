import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_recommend_indexes_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)

    # Patch the internal .find().limit().to_list() chain to raise PyMongoError
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(side_effect=PyMongoError("PyMongo boom"))

    mock_find = MagicMock()
    mock_find.limit.return_value = mock_cursor

    mock_collection = MagicMock()
    mock_collection.find.return_value = mock_find

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.recommend_indexes("some_collection")

    print("📢 Final result:", result)
    assert isinstance(result, dict)
    assert "error" in result
    assert "PyMongo boom" in result["error"]

import pytest
from unittest.mock import MagicMock
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_recommend_indexes_generic_exception():
    zmagnum = ZMagnum(disable_cache=True)

    # Patch the collection's .find to raise a generic (non-PyMongo) exception
    mock_collection = MagicMock()
    mock_collection.find.side_effect = RuntimeError("Some generic failure")

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.recommend_indexes("some_collection")

    print("📢 Final result:", result)
    assert isinstance(result, dict)
    assert "error" in result
    assert "Some generic failure" in result["error"]
