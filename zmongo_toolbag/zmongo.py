import os
import asyncio
import threading
import weakref
import logging
from typing import Any, Dict, List, Optional

from motor import motor_asyncio
from bson import ObjectId
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult, BulkWriteResult

from zmongo_toolbag.safe_result import SafeResult
from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZMongo:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.db_name = db_name or os.getenv("MONGO_DATABASE_NAME", "test")

        # Background event loop for sync API
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        self._client_bg = motor_asyncio.AsyncIOMotorClient(self.uri)
        self.db = self._client_bg[self.db_name]

        # Async clients bound per event loop
        self._async_clients = weakref.WeakKeyDictionary()
        self.caches: Dict[str, BufferedAsyncTTLCache] = {}

    # ------------------------------------------------------------
    # Loop Management
    # ------------------------------------------------------------

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _client_for_async(self):
        loop = asyncio.get_running_loop()
        cli = self._async_clients.get(loop)
        if cli is None:
            cli = motor_asyncio.AsyncIOMotorClient(self.uri)
            self._async_clients[loop] = cli
        return cli

    def run_sync(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=20)
        except Exception as e:
            return SafeResult.fail(str(e))

    # ------------------------------------------------------------
    # Async Motor Operations (safe per-loop)
    # ------------------------------------------------------------
    async def insert_many_async(self, coll: str, docs: List[Dict[str, Any]]) -> SafeResult:
        """Insert multiple documents asynchronously."""
        try:
            collection = self._client_for_async()[self.db_name][coll]
            r = await collection.insert_many(docs)
            return SafeResult.ok({"inserted_ids": r.inserted_ids})
        except Exception as e:
            return SafeResult.fail(str(e))

    async def update_many_async(
        self,
        coll: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> SafeResult:
        """Update multiple documents asynchronously."""
        try:
            collection = self._client_for_async()[self.db_name][coll]
            r = await collection.update_many(query, update, upsert=upsert)
            return SafeResult.ok(
                {
                    "matched_count": r.matched_count,
                    "modified_count": r.modified_count,
                    "upserted_id": r.upserted_id,
                }
            )
        except Exception as e:
            return SafeResult.fail(str(e))


    async def insert_one_async(self, coll: str, doc: Dict[str, Any]) -> SafeResult:
        try:
            collection = self._client_for_async()[self.db_name][coll]
            r = await collection.insert_one(doc)
            return SafeResult.ok({"inserted_id": r.inserted_id})
        except Exception as e:
            return SafeResult.fail(str(e))

    async def find_one_async(self, coll: str, query: Dict[str, Any]) -> SafeResult:
        try:
            collection = self._client_for_async()[self.db_name][coll]
            doc = await collection.find_one(query)
            return SafeResult.ok(doc)
        except Exception as e:
            return SafeResult.fail(str(e))

    async def update_one_async(self, coll: str, query: Dict[str, Any], update: Dict[str, Any], upsert=False) -> SafeResult:
        try:
            collection = self._client_for_async()[self.db_name][coll]
            r = await collection.update_one(query, update, upsert=upsert)
            return SafeResult.ok({"matched_count": r.matched_count, "modified_count": r.modified_count})
        except Exception as e:
            return SafeResult.fail(str(e))

    async def delete_many_async(self, coll: str, query: Dict[str, Any]) -> SafeResult:
        try:
            collection = self._client_for_async()[self.db_name][coll]
            r = await collection.delete_many(query)
            return SafeResult.ok({"deleted_count": r.deleted_count})
        except Exception as e:
            return SafeResult.fail(str(e))

    async def aggregate_async(self, coll: str, pipeline: List[Dict[str, Any]]) -> SafeResult:
        try:
            collection = self._client_for_async()[self.db_name][coll]
            cur = collection.aggregate(pipeline)
            docs = await cur.to_list(None)
            return SafeResult.ok(docs)
        except Exception as e:
            return SafeResult.fail(str(e))

    # ------------------------------------------------------------
    # Sync Wrappers
    # ------------------------------------------------------------

    def insert_one(self, *a, **kw):
        return self.run_sync(self.insert_one_async(*a, **kw))

    def find_one(self, *a, **kw):
        return self.run_sync(self.find_one_async(*a, **kw))

    def update_one(self, *a, **kw):
        return self.run_sync(self.update_one_async(*a, **kw))

    def delete_many(self, *a, **kw):
        return self.run_sync(self.delete_many_async(*a, **kw))

    def aggregate(self, *a, **kw):
        return self.run_sync(self.aggregate_async(*a, **kw))

    def insert_many(self, *a, **kw):
        return self.run_sync(self.insert_many_async(*a, **kw))

    def update_many(self, *a, **kw):
        return self.run_sync(self.update_many_async(*a, **kw))


    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def close(self):
        if self._client_bg:
            self._client_bg.close()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def clear_cache(self, coll: str):
        if coll in self.caches:
            self.caches.pop(coll)
