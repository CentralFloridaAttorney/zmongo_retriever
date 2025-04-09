import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

@pytest.mark.asyncio
async def test_recommend_indexes_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)

    # Patch find().limit().to_list() chain
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(side_effect=PyMongoError("PyMongo boom"))

    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = mock_cursor

    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.recommend_indexes("some_collection")

    print("📢 Final result:", result)
    assert isinstance(result, dict)
    assert "error" in result
    assert "PyMongo boom" in result["error"]
