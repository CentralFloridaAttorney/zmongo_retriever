import pytest
from unittest.mock import MagicMock, patch
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum

def test_create_indexes_success(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    mock_collection = MagicMock()
    zmagnum.sync_db = { "my_collection": mock_collection }

    fields = ["name", "email"]
    zmagnum.create_indexes("my_collection", fields)

    # Check that create_index was called for each field
    assert mock_collection.create_index.call_count == len(fields)
    mock_collection.create_index.assert_any_call("name")
    mock_collection.create_index.assert_any_call("email")

def test_create_indexes_pymongo_error(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    mock_collection = MagicMock()
    mock_collection.create_index.side_effect = PyMongoError("Index boom")

    zmagnum.sync_db = { "my_collection": mock_collection }

    with caplog.at_level("ERROR"):
        zmagnum.create_indexes("my_collection", ["name"])
        assert "Mongo error creating index on 'name'" in caplog.text

def test_create_indexes_generic_exception(caplog):
    zmagnum = ZMagnum(disable_cache=True)
    mock_collection = MagicMock()
    mock_collection.create_index.side_effect = RuntimeError("Unexpected fail")

    zmagnum.sync_db = { "my_collection": mock_collection }

    with caplog.at_level("ERROR"):
        zmagnum.create_indexes("my_collection", ["name"])
        assert "Failed to create index on 'name'" in caplog.text
