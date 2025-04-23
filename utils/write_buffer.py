import asyncio
import threading
from typing import Any, Dict, List, Callable
from collections import defaultdict
from zmongo_toolbag.utils.ttl_cache import TTLCache  # assumes your TTLCache is here


class WriteBuffer:
    def __init__(self, ttl: int = 60, flush_interval: int = 10):
        """
        A class-based utility for staging MongoDB write operations in memory using TTL.

        Args:
            ttl (int): Time-to-live for each staged operation.
            flush_interval (int): Seconds between background flush attempts.
        """
        self.buffers: Dict[str, TTLCache] = defaultdict(lambda: TTLCache(ttl=ttl))
        self.ttl = ttl
        self.flush_interval = flush_interval
        self.running = False
        self.lock = threading.Lock()

    def add(self, collection: str, op_key: Any, op_data: Dict[str, Any]) -> None:
        """Add a write operation to the buffer."""
        self.buffers[collection].set(op_key, op_data)

    def get_operations(self, collection: str) -> List[Dict[str, Any]]:
        """Fetch and clear valid operations for a collection."""
        with self.lock:
            ops = list(self.buffers[collection].items())
            self.buffers[collection].clear()
            return [v for _, v in ops]

    async def flush_to(self, collection: str, flush_fn: Callable[[str, List[Dict[str, Any]]], Any]) -> None:
        """
        Flush current buffer to Mongo using the provided bulk write function.

        Args:
            collection (str): The collection name.
            flush_fn (Callable): A coroutine that takes (collection, operations).
        """
        operations = self.get_operations(collection)
        if operations:
            await flush_fn(collection, operations)

    async def auto_flush(self, flush_fn: Callable[[str, List[Dict[str, Any]]], Any]) -> None:
        """Run auto-flush loop in the background."""
        self.running = True
        while self.running:
            for collection in list(self.buffers.keys()):
                await self.flush_to(collection, flush_fn)
            await asyncio.sleep(self.flush_interval)

    def stop(self) -> None:
        self.running = False
