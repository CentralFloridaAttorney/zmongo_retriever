import asyncio
import pytest

from utils.write_buffer import WriteBuffer

results = {}

async def mock_bulk_write(collection: str, ops: list):
    # Simulate MongoDB bulk write and capture results
    results[collection] = ops

@pytest.mark.asyncio
async def test_write_buffer_manual_flush():
    buffer = WriteBuffer(ttl=5)

    # Add test operations
    buffer.add("test_coll", "key1", {"operation": "insert", "document": {"name": "Doc1"}})
    buffer.add("test_coll", "key2", {"operation": "insert", "document": {"name": "Doc2"}})

    assert len(list(buffer.buffers["test_coll"].items())) == 2

    await buffer.flush_to("test_coll", mock_bulk_write)
    assert "test_coll" in results
    assert len(results["test_coll"]) == 2

    # After flush, buffer should be cleared
    assert len(list(buffer.buffers["test_coll"].items())) == 0

@pytest.mark.asyncio
async def test_write_buffer_auto_flush():
    buffer = WriteBuffer(ttl=5, flush_interval=1)
    buffer.add("auto_coll", "a1", {"operation": "insert", "document": {"name": "Auto1"}})
    buffer.add("auto_coll", "a2", {"operation": "insert", "document": {"name": "Auto2"}})

    # Start auto flush loop
    flush_task = asyncio.create_task(buffer.auto_flush(mock_bulk_write))
    await asyncio.sleep(2.5)  # Let it flush at least once
    buffer.stop()
    await flush_task

    assert "auto_coll" in results
    assert len(results["auto_coll"]) == 2
