import time
import asyncio
from typing import Any, Optional, Dict, Tuple, List, Callable, Awaitable

class BufferedAsyncTTLCache:
    """
    An async TTL (Time-to-Live) cache with write-back buffer for high-throughput batch inserts.
    """
    def __init__(
        self,
        ttl: int = 60,
        buffer_limit: int = 1000,
        flush_fn: Optional[Callable[[List[Tuple[Any, Any]]], Awaitable[None]]] = None
    ) -> None:
        self.default_ttl = ttl
        self._cache: Dict[Any, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        self._buffer: List[Tuple[Any, Any]] = []
        self.buffer_limit = buffer_limit
        self.flush_fn = flush_fn

    def _is_expired(self, expire_time: float) -> bool:
        return time.time() > expire_time

    async def set(self, key: Any, value: Any, ttl: Optional[int] = None, buffer_only: bool = False) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expire_time = time.time() + ttl
        async with self._lock:
            if buffer_only:
                self._buffer.append((key, value))
                # --- DO NOT CALL flush() here ---
            else:
                self._cache[key] = (value, expire_time)

    async def get(self, key: Any, default: Optional[Any] = None) -> Any:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return default
            value, expire_time = entry
            if self._is_expired(expire_time):
                del self._cache[key]
                return default
            return value

    async def delete(self, key: Any) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._buffer.clear()

    async def cleanup(self) -> None:
        current_time = time.time()
        async with self._lock:
            keys_to_delete = [key for key, (_, expire_time) in self._cache.items() if current_time > expire_time]
            for key in keys_to_delete:
                del self._cache[key]

    async def flush(self) -> None:
        async with self._lock:
            if self.flush_fn and self._buffer:
                buffer_snapshot = self._buffer.copy()
                self._buffer.clear()
            else:
                return
        # Call flush_fn outside lock for safety
        try:
            await self.flush_fn(buffer_snapshot)
        except Exception as e:
            print(f"BufferedAsyncTTLCache flush_fn error: {e}")

    def __contains__(self, key: Any) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        value, expire_time = entry
        if self._is_expired(expire_time):
            return False
        return True

    def __len__(self) -> int:
        current_time = time.time()
        return sum(
            1 for (_, expire_time) in self._cache.values()
            if current_time <= expire_time
        )

    async def async_contains(self, key: Any) -> bool:
        return (await self.get(key)) is not None

    async def async_len(self) -> int:
        await self.cleanup()
        async with self._lock:
            return len(self._cache)

    async def items(self):
        await self.cleanup()
        async with self._lock:
            for key, (value, _) in self._cache.items():
                yield key, value

    async def buffered_count(self) -> int:
        async with self._lock:
            return len(self._buffer)
