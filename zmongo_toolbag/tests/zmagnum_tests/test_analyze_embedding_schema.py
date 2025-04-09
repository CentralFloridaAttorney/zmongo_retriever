import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_analyze_embedding_schema_success():
    zmagnum = ZMagnum(disable_cache=True)

    fake_docs = [
        {"embedding": [1, 2, 3]},
        {"embedding": [4, 5, 6, 7]},
        {"embedding": [8, 9]},
    ]
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=fake_docs)

    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = mock_cursor
    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.analyze_embedding_schema("test")
    assert result["sampled"] == 3
    assert result["avg_embedding_length"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_analyze_embedding_schema_no_embeddings_found():
    zmagnum = ZMagnum(disable_cache=True)

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[])

    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = mock_cursor
    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.analyze_embedding_schema("test")
    assert result["error"] == "No embeddings found."


@pytest.mark.asyncio
async def test_analyze_embedding_schema_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(side_effect=PyMongoError("DB down"))

    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = mock_cursor
    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.analyze_embedding_schema("test")
    assert "error" in result
    assert "DB down" in result["error"]


@pytest.mark.asyncio
async def test_analyze_embedding_schema_generic_error():
    zmagnum = ZMagnum(disable_cache=True)

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(side_effect=RuntimeError("Unexpected failure"))

    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = mock_cursor
    zmagnum.db = MagicMock()
    zmagnum.db.__getitem__.return_value = mock_collection

    result = await zmagnum.analyze_embedding_schema("test")
    assert "error" in result
    assert "Unexpected failure" in result["error"]
