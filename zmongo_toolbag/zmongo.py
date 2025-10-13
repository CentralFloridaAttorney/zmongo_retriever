import os
import asyncio
import threading
import weakref
import logging
from typing import Any, Dict, List, Optional, Tuple

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

    async def list_collections_async(self) -> SafeResult:
        """
        Asynchronously list all collections in the current database.
        Returns a SafeResult containing a list of collection names.
        """
        try:
            client = self._client_for_async()
            names = await client[self.db_name].list_collection_names()
            return SafeResult.ok({"collections": names})
        except Exception as e:
            return SafeResult.fail(str(e))

    async def sync_timestamp_async(self) -> SafeResult:
        """
        Asynchronously retrieve the MongoDB server's current time and latency check.
        Returns SafeResult with fields:
          - 'server_time': MongoDB's reported server time
          - 'local_time': local system time
          - 'offset_seconds': difference between local and server timestamps
        """
        import datetime, time

        try:
            client = self._client_for_async()
            admin_db = client["admin"]

            # Run a lightweight serverStatus command
            start = time.time()
            status = await admin_db.command("serverStatus")
            end = time.time()

            server_time = status.get("localTime", None)
            if server_time is None:
                raise RuntimeError("serverStatus did not return localTime")

            offset = (
                (server_time.timestamp() - datetime.datetime.utcnow().timestamp())
                if hasattr(server_time, "timestamp")
                else None
            )
            latency = end - start

            return SafeResult.ok({
                "server_time": server_time,
                "local_time": datetime.datetime.utcnow(),
                "offset_seconds": offset,
                "latency_seconds": latency,
            })
        except Exception as e:
            return SafeResult.fail(str(e))

    async def find_many_async(
        self,
        coll: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None,
        limit: int = 1000,
        sort: Optional[List[Tuple[str, int]]] = None,
    ) -> SafeResult:
        """
        Asynchronously find multiple documents from a collection.

        Args:
            coll: Collection name.
            query: MongoDB filter dict.
            projection: Optional projection dict (fields to include/exclude).
            limit: Maximum number of documents to return.
            sort: Optional list of (field, direction) tuples.

        Returns:
            SafeResult containing a list of documents.
        """
        try:
            collection = self._client_for_async()[self.db_name][coll]
            cursor = collection.find(query or {}, projection)
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            docs = await cursor.to_list(length=limit)
            return SafeResult.ok(docs)
        except Exception as e:
            return SafeResult.fail(str(e))

    async def insert_or_update_async(
        self,
        coll: str,
        query_or_doc: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        upsert: bool = True
    ) -> SafeResult:
        """
        Asynchronously insert or update a document.
        - If `data` is None, performs an insert.
        - Otherwise, performs an update (upsert=True by default).
        Automatically wraps update dict in {"$set": data} if needed.
        """
        try:
            collection = self._client_for_async()[self.db_name][coll]

            # Pure insert mode
            if data is None:
                result = await collection.insert_one(query_or_doc)
                return SafeResult.ok({
                    "upserted_id": result.inserted_id,
                    "modified_count": 0
                })

            # Update mode
            if not any(k.startswith("$") for k in data.keys()):
                data = {"$set": data}

            result = await collection.update_one(query_or_doc, data, upsert=upsert)
            return SafeResult.ok({
                "upserted_id": result.upserted_id,
                "modified_count": result.modified_count
            })

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

    def list_collections(self) -> SafeResult:
        """Synchronous wrapper for list_collections_async."""
        return self.run_sync(self.list_collections_async())

    def sync_timestamp(self) -> SafeResult:
        """
        Synchronous wrapper for sync_timestamp_async.
        Verifies MongoDB connectivity and clock offset.
        """
        return self.run_sync(self.sync_timestamp_async())

    def find_many(
        self,
        coll: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None,
        limit: int = 1000,
        sort: Optional[List[Tuple[str, int]]] = None,
    ) -> SafeResult:
        """Synchronous wrapper for find_many_async."""
        return self.run_sync(self.find_many_async(coll, query, projection, limit, sort))

    def insert_or_update(
        self,
        coll: str,
        query_or_doc: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        upsert: bool = True
    ) -> SafeResult:
        """
        Sync wrapper for insert_or_update_async.
        Runs safely inside ZMongo’s dedicated async loop.
        """
        return self.run_sync(self.insert_or_update_async(coll, query_or_doc, data, upsert))


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
