"""
ZMongo — Dual API MongoDB Helper with SafeResult Enforcement
=============================================================

This version provides a comprehensive, dual async/sync API.
- Primary methods are `async` for modern, non-blocking code.
- Synchronous versions are available with a `_sync` suffix for legacy
  or UI code (e.g., Tkinter).

It includes full pymongo/motor aliases, caching, and robust error handling.
"""

import os
import threading
import logging
import asyncio
from typing import Any, Dict, List, Optional

import motor.motor_asyncio
from bson import ObjectId

from pymongo.results import (
    InsertOneResult, UpdateResult, DeleteResult, BulkWriteResult
)
from pymongo.operations import InsertOne, UpdateOne, DeleteOne, ReplaceOne

# Make sure these imports point to your actual files
from zmongo_toolbag.data_processing import SafeResult
from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache

# ------------------------------------------------------------
# ENV + Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = int(os.getenv("ZMONGO_CACHE_TTL", "300"))
STRINGIFY_IDS = os.getenv("ZMONGO_STRINGIFY_IDS", "true").lower() not in {"0", "false", "no"}

def _parse_result(res: Any) -> Dict[str, Any]:
    """Convert PyMongo/Motor results to plain dicts."""
    if isinstance(res, InsertOneResult):
        return {"inserted_id": str(res.inserted_id), "acknowledged": res.acknowledged}
    if isinstance(res, (UpdateResult, ReplaceOne)):
        return {
            "matched_count": res.matched_count,
            "modified_count": res.modified_count,
            "upserted_id": str(res.upserted_id) if res.upserted_id else None,
            "acknowledged": res.acknowledged,
        }
    if isinstance(res, DeleteResult):
        return {"deleted_count": res.deleted_count, "acknowledged": res.acknowledged}
    if isinstance(res, BulkWriteResult):
        return {
            "inserted_count": res.inserted_count,
            "modified_count": res.modified_count,
            "deleted_count": res.deleted_count,
            "upserted_count": res.upserted_count,
            "acknowledged": res.acknowledged,
        }
    if isinstance(res, dict):
        return res
    if isinstance(res, int):
        return {"count": res}
    return {"acknowledged": True}


def _convert_query_ids(query: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert '_id' string fields into ObjectId if valid."""
    if not isinstance(query, dict): return query
    def _maybe_oid(x):
        return ObjectId(x) if isinstance(x, str) and ObjectId.is_valid(x) else x
    new_query = {}
    for k, v in query.items():
        if k == "_id":
            if isinstance(v, str): new_query[k] = _maybe_oid(v)
            elif isinstance(v, dict): new_query[k] = {op: ([_maybe_oid(i) for i in val] if op in {"$in", "$nin"} else val) for op, val in v.items()}
            elif isinstance(v, list): new_query[k] = [_maybe_oid(i) for i in v]
            else: new_query[k] = v
        elif k in {"$or", "$and", "$nor"} and isinstance(v, list): new_query[k] = [_convert_query_ids(i) for i in v]
        elif isinstance(v, dict): new_query[k] = _convert_query_ids(v)
        else: new_query[k] = v
    return new_query


class ZMongo:
    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        stringify_ids: bool = STRINGIFY_IDS,
    ):
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.db_name = db_name or os.getenv("MONGO_DATABASE_NAME", "test")
        self._stringify_ids = stringify_ids
        self.default_cache_ttl = cache_ttl
        self.caches: Dict[str, BufferedAsyncTTLCache] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop_forever, name="ZMongoLoop", daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._create_motor_client(), self._loop)
        self.client, self.db = fut.result(timeout=10)

    def _run_loop_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_motor_client(self):
        client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
        return client, client[self.db_name]

    def close(self):
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)
        if self.client: self.client.close()
        logger.info("🛑 ZMongo background loop stopped cleanly.")

    def run_sync(self, coro, timeout=60) -> SafeResult:
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            result = fut.result(timeout=timeout)
            if isinstance(result, SafeResult):
                return result
            return SafeResult.ok(result)
        except Exception as e:
            logger.error(f"run_sync error: {e}", exc_info=True)
            return SafeResult.fail(str(e))

    # --- Caching Internals ---
    def _get_cache(self, coll: str) -> BufferedAsyncTTLCache:
        self.caches.setdefault(coll, BufferedAsyncTTLCache(ttl=self.default_cache_ttl))
        return self.caches[coll]
    async def _cset(self, c, k, v): await self._get_cache(c).set(k, v)
    async def _cget(self, c, k): return await self._get_cache(c).get(k)
    async def _cinvalidate(self, c, k): await self._get_cache(c).delete(k)
    async def _cclear(self, c): self.caches.pop(c, None)

    # --- Primary Async API ---

    async def insert_one(self, coll: str, doc: Dict[str, Any]) -> SafeResult:
        try:
            if "_id" not in doc: doc["_id"] = ObjectId()
            res = await self.db[coll].insert_one(doc)
            return SafeResult.ok(_parse_result(res))
        except Exception as e: return SafeResult.fail(str(e))

    async def insert_many(self, coll: str, docs: List[Dict[str, Any]], **kwargs) -> SafeResult:
        try:
            res = await self.db[coll].insert_many(docs)
            return SafeResult.ok({"inserted_ids": [str(i) for i in res.inserted_ids]})
        except Exception as e: return SafeResult.fail(str(e))

    async def find_one(self, coll: str, query: Dict[str, Any], cache: bool = False) -> SafeResult:
        try:
            q = _convert_query_ids(query)
            cache_key = str(query.get("_id")) if query.get("_id") else None
            if cache and cache_key and (cached := await self._cget(coll, cache_key)): return SafeResult.ok(cached)
            doc = await self.db[coll].find_one(q)
            if doc and self._stringify_ids and '_id' in doc: doc['_id'] = str(doc['_id'])
            if cache and cache_key and doc: await self._cset(coll, cache_key, doc)
            return SafeResult.ok(doc)
        except Exception as e: return SafeResult.fail(str(e))

    async def find(self, coll: str, query: Dict[str, Any], limit: int = 0) -> SafeResult:
        try:
            cursor = self.db[coll].find(_convert_query_ids(query))
            docs = await cursor.to_list(length=limit if limit > 0 else None)
            if self._stringify_ids:
                for doc in docs:
                    if '_id' in doc: doc['_id'] = str(doc['_id'])
            return SafeResult.ok(docs)
        except Exception as e: return SafeResult.fail(str(e))

    async def update_one(self, coll: str, query: Dict[str, Any], data: Dict[str, Any], upsert=False) -> SafeResult:
        try:
            q = _convert_query_ids(query)
            cache_key = str(query.get("_id")) if query.get("_id") else None
            if cache_key: await self._cinvalidate(coll, cache_key)
            payload = data if any(k.startswith("$") for k in data) else {"$set": data}
            res = await self.db[coll].update_one(q, payload, upsert=upsert)
            return SafeResult.ok(_parse_result(res))
        except Exception as e: return SafeResult.fail(str(e))

    async def update_many(self, coll: str, query: Dict[str, Any], data: Dict[str, Any], upsert=False) -> SafeResult:
        try:
            await self._cclear(coll)
            payload = data if any(k.startswith("$") for k in data) else {"$set": data}
            res = await self.db[coll].update_many(_convert_query_ids(query), payload, upsert=upsert)
            return SafeResult.ok(_parse_result(res))
        except Exception as e: return SafeResult.fail(str(e))

    async def delete_one(self, coll: str, query: Dict[str, Any]) -> SafeResult:
        try:
            q = _convert_query_ids(query)
            cache_key = str(query.get("_id")) if query.get("_id") else None
            if cache_key: await self._cinvalidate(coll, cache_key)
            res = await self.db[coll].delete_one(q)
            return SafeResult.ok(_parse_result(res))
        except Exception as e: return SafeResult.fail(str(e))

    async def delete_many(self, coll: str, query: Dict[str, Any]) -> SafeResult:
        try:
            await self._cclear(coll)
            res = await self.db[coll].delete_many(_convert_query_ids(query))
            return SafeResult.ok(_parse_result(res))
        except Exception as e: return SafeResult.fail(str(e))

    async def count_documents(self, coll: str, query: Dict[str, Any]) -> SafeResult:
        try:
            count = await self.db[coll].count_documents(_convert_query_ids(query))
            return SafeResult.ok({"count": count})
        except Exception as e: return SafeResult.fail(str(e))

    async def aggregate(self, coll: str, pipeline: list, limit: int = 0) -> SafeResult:
        try:
            safe_pipeline = [{'$match': _convert_query_ids(s['$match'])} if '$match' in s else s for s in pipeline]
            cursor = self.db[coll].aggregate(safe_pipeline)
            docs = await cursor.to_list(length=limit if limit > 0 else None)
            if self._stringify_ids:
                for doc in docs:
                    if '_id' in doc: doc['_id'] = str(doc['_id'])
            return SafeResult.ok(docs)
        except Exception as e: return SafeResult.fail(str(e))

    async def bulk_write(self, coll: str, operations: List) -> SafeResult:
        try:
            await self._cclear(coll)
            res = await self.db[coll].bulk_write(operations)
            return SafeResult.ok(_parse_result(res))
        except Exception as e:
            return SafeResult.fail(str(e))

    async def list_collection_names(self) -> SafeResult:
        try:
            return SafeResult.ok(await self.db.list_collection_names())
        except Exception as e: return SafeResult.fail(str(e))

    async def insert_or_update(self, coll: str, doc: Dict[str, Any], query_key: str = "_id") -> SafeResult:
        query_val = doc.get(query_key)
        if not query_val:
            return await self.insert_one(coll, doc)

        if query_key == "_id":
             await self._cinvalidate(coll, str(query_val))

        query = {query_key: query_val}
        return await self.update_one(coll, query, doc, upsert=True)

    async def drop_database(self, db_name: str) -> SafeResult:
        try:
            await self.client.drop_database(db_name)
            return SafeResult.ok(f"Database '{db_name}' dropped.")
        except Exception as e:
            return SafeResult.fail(str(e))

    # --- Synchronous API (for UI/legacy code) ---
    def insert_one_sync(self, *a, **kw): return self.run_sync(self.insert_one(*a, **kw))
    def insert_many_sync(self, *a, **kw): return self.run_sync(self.insert_many(*a, **kw))
    def find_one_sync(self, *a, **kw): return self.run_sync(self.find_one(*a, **kw))
    def find_sync(self, *a, **kw): return self.run_sync(self.find(*a, **kw))
    def update_one_sync(self, *a, **kw): return self.run_sync(self.update_one(*a, **kw))
    def update_many_sync(self, *a, **kw): return self.run_sync(self.update_many(*a, **kw))
    def delete_one_sync(self, *a, **kw): return self.run_sync(self.delete_one(*a, **kw))
    def delete_many_sync(self, *a, **kw): return self.run_sync(self.delete_many(*a, **kw))
    def count_documents_sync(self, *a, **kw): return self.run_sync(self.count_documents(*a, **kw))
    def aggregate_sync(self, *a, **kw): return self.run_sync(self.aggregate(*a, **kw))
    def bulk_write_sync(self, *a, **kw): return self.run_sync(self.bulk_write(*a, **kw))
    def list_collection_names_sync(self, *a, **kw): return self.run_sync(self.list_collection_names(*a, **kw))
    def insert_or_update_sync(self, *a, **kw): return self.run_sync(self.insert_or_update(*a, **kw))
    def drop_database_sync(self, *a, **kw): return self.run_sync(self.drop_database(*a, **kw))

    # --- Legacy Aliases (pointing to ASYNC versions for test compatibility) ---
    insert_document = insert_one
    insert_documents = insert_many
    find_document = find_one
    find_documents = find
    update_document = update_one
    update_documents = update_many
    delete_document = delete_one
    delete_documents = delete_many

    # NEW: Aliases for cleanup methods
    delete_all_documents_sync = delete_many_sync
    delete_all_documents = delete_many

