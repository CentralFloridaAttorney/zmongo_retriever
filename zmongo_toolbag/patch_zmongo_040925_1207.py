import asyncio
import logging

import logger
import time
from collections import defaultdict
from typing import Callable

from cachetools import TTLCache
from pymongo.errors import PyMongoError

# TTL-based cache replacement
DEFAULT_CACHE_TTL = 300
DEFAULT_CACHE_MAXSIZE = 1024

class ZMongo:
    def __init__(self, disable_cache: bool = False) -> None:
        self.disable_cache = disable_cache

        # ... (other connection code)

        self.cache = (
            defaultdict(lambda: TTLCache(maxsize=DEFAULT_CACHE_MAXSIZE, ttl=DEFAULT_CACHE_TTL))
            if not self.disable_cache else {}
        )

        if self.disable_cache:
            logger.warning("Fast mode enabled: disabling cache and reducing logging noise.")
            logger.setLevel(logging.WARNING)

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    @staticmethod
    def _profile(operation_name: str, fn: Callable, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except PyMongoError as e:
            logger.error(f"[PROFILE][{operation_name}] Mongo error: {e}")
            raise
        except Exception as e:
            logger.error(f"[PROFILE][{operation_name}] error: {e}")
            raise
        elapsed = time.perf_counter() - start
        logger.info(f"[PROFILE] {operation_name} took {elapsed:.4f} seconds")
        return result

    def clear_cache(self) -> None:
        if not self.disable_cache:
            self.cache.clear()
            logger.info("Cache cleared.")

    # Aliases (replace previous definitions)
    reset_cache = flush_cache = clear_cache

# --- PATCH END ---
