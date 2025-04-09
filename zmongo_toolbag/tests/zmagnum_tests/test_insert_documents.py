# test_zmagnum_insert_empty.py

import pytest
from zmongo_toolbag.zmagnum import ZMagnum


@pytest.mark.asyncio
async def test_insert_documents_empty_returns_zero():
    zmag = ZMagnum(disable_cache=True)
    result = await zmag.insert_documents("test_collection", [])
    assert isinstance(result, dict)
    assert result == {"inserted_count": 0}
