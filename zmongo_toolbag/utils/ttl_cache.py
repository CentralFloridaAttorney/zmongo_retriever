import time
import threading
from typing import Any, Optional, Dict, Tuple


class TTLCache:
    """
    A simple TTL (Time-to-Live) cache.

    Each value inserted into the cache is stored together with its expiry time.
    Items that have expired will be purged upon access or during an explicit cleanup.
    """

    def __init__(self, ttl: int = 60) -> None:
        """
        Initialize the cache.

        Args:
            ttl (int): The default time-to-live (in seconds) for cache entries.
        """
        self.default_ttl = ttl
        # Internal storage: key => (value, expire_time)
        self._cache: Dict[Any, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _is_expired(self, expire_time: float) -> bool:
        """Return True if the expire_time is in the past."""
        return time.time() > expire_time

    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a cache entry with an optional TTL.

        Args:
            key (Any): The key under which the value is stored.
            value (Any): The value to cache.
            ttl (Optional[int]): Optional TTL for the entry in seconds; if None, default_ttl is used.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expire_time = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expire_time)

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the cache if it exists and is still valid.

        Args:
            key (Any): The key to retrieve.
            default (Optional[Any]): The value to return if key is not found or has expired.

        Returns:
            The cached value, or `default` if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return default

            value, expire_time = entry
            if self._is_expired(expire_time):
                # Purge expired key
                del self._cache[key]
                return default
            return value

    def delete(self, key: Any) -> None:
        """
        Delete a cache entry if it exists.

        Args:
            key (Any): The key to remove.
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """
        Clear the cache completely.
        """
        with self._lock:
            self._cache.clear()

    def cleanup(self) -> None:
        """
        Purge all expired entries from the cache.
        """
        current_time = time.time()
        with self._lock:
            keys_to_delete = [key for key, (_, expire_time) in self._cache.items() if current_time > expire_time]
            for key in keys_to_delete:
                del self._cache[key]

    def __contains__(self, key: Any) -> bool:
        """
        Check whether a key exists in the cache and is not expired.

        Args:
            key (Any): The key to check.

        Returns:
            bool: True if key is in the cache and valid, False otherwise.
        """
        return self.get(key) is not None

    def __len__(self) -> int:
        """
        Return the number of entries currently in the cache (excluding expired entries).
        """
        self.cleanup()
        return len(self._cache)

    def items(self):
        """
        Return a generator of (key, value) pairs for valid items in the cache.
        """
        self.cleanup()
        with self._lock:
            for key, (value, _) in self._cache.items():
                yield key, value


# --- Example usage ---
if __name__ == '__main__':
    cache = TTLCache(default_ttl=5)

    cache.set("foo", "bar")
    print("Set key 'foo' to 'bar'")
    print("Getting key 'foo':", cache.get("foo"))

    time.sleep(6)  # Wait until the entry expires
    print("After sleep, getting key 'foo':", cache.get("foo", default="expired"))
