import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from zmongo_toolbag.zmagnum import ZMagnum

@pytest_asyncio.fixture
async def zmagnum():
    instance = ZMagnum(disable_cache=True)
    yield instance
    await instance.close()

@pytest.mark.asyncio
async def test_insert_documents_exception_is_logged():
    # Patch ZMagnum so that .db["test"].insert_many throws
    with patch("zmongo_toolbag.zmagnum.AsyncIOMotorClient") as mock_motor:
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_many = AsyncMock(side_effect=Exception("Boom insert"))
        mock_db.__getitem__.return_value = mock_collection
        mock_motor.return_value.__getitem__.return_value = mock_db

        zmagnum = ZMagnum(disable_cache=True)
        result = await zmagnum.insert_documents("test", [{"a": 1}, {"b": 2}])

        assert isinstance(result, dict)
        assert result["inserted_count"] == 0
