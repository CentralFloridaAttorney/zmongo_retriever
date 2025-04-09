import pytest
from unittest.mock import MagicMock, patch
from pymongo.errors import PyMongoError
from zmongo_toolbag.zmagnum import ZMagnum


def test_is_sharded_cluster_returns_true():
    zmagnum = ZMagnum(disable_cache=True)
    mock_admin = MagicMock()
    mock_admin.command.return_value = {"msg": "isdbgrid"}

    zmagnum.sync_client.admin = mock_admin

    result = zmagnum.is_sharded_cluster()
    assert result is True


def test_is_sharded_cluster_returns_false_on_pymongo_error():
    zmagnum = ZMagnum(disable_cache=True)
    mock_admin = MagicMock()
    mock_admin.command.side_effect = PyMongoError("Sharded check failure")

    zmagnum.sync_client.admin = mock_admin

    result = zmagnum.is_sharded_cluster()
    assert result is False


def test_is_sharded_cluster_returns_false_on_generic_exception():
    zmagnum = ZMagnum(disable_cache=True)
    mock_admin = MagicMock()
    mock_admin.command.side_effect = RuntimeError("Non-mongo failure")

    zmagnum.sync_client.admin = mock_admin

    result = zmagnum.is_sharded_cluster()
    assert result is False
