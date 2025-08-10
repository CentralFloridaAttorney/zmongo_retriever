import os

import motor.motor_asyncio
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult, BulkWriteResult

import time
import asyncio
from typing import Dict, Tuple, List, Callable, Awaitable, Union

import logging
from typing import Any, Optional
from bson import ObjectId
import json
JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]
DocsLike = Union[List[DocLike], Any]

_KEYMAP_FIELD = "__keymap"
DEFAULT_QUERY_LIMIT = 100
DEFAULT_CACHE_TTL = 300  # seconds


def _needs_fix(k: str) -> bool:
    return k.startswith("_") and k != "_id"


def _sanitize_dict(d: JsonDict) -> JsonDict:
    """Recursively sanitize a dict's keys for MongoDB compatibility, with alias mapping."""
    if not any(_needs_fix(k) for k in d):
        return d
    fixed, keymap = {}, {}
    for k, v in d.items():
        if _needs_fix(k):
            safe_k = f"u{k.lstrip('_')}"
            while safe_k in d or safe_k in fixed:
                safe_k = f"u{safe_k}"
            keymap[safe_k] = k
            fixed[safe_k] = v
        else:
            fixed[k] = v
    fixed[_KEYMAP_FIELD] = keymap
    return fixed


def _restore_dict(d: JsonDict) -> JsonDict:
    """Restore original key names from sanitized dict with alias mapping."""
    if _KEYMAP_FIELD not in d:
        return d
    keymap = d.pop(_KEYMAP_FIELD, {})
    for safe_k, orig_k in keymap.items():
        if safe_k in d:
            d[orig_k] = d.pop(safe_k)
    return d


def _serialise_doc(obj: DocLike) -> JsonDict:
    """Convert an object to a dict (by alias) and sanitize keys."""
    if hasattr(obj, 'dict'):
        d = obj.dict(by_alias=True)
    elif hasattr(obj, 'model_dump'):
        d = obj.model_dump(by_alias=True)
    else:
        d = dict(obj)
    return _sanitize_dict(d)


def _sanitize_query(query: Any) -> Any:
    """Sanitize a query dict for MongoDB; skip for _id-only queries."""
    if isinstance(query, dict) and set(query) == {"_id"}:
        return query
    if isinstance(query, dict):
        def fix_key(k):
            if k.startswith("$"):
                return k
            if _needs_fix(k):
                safe_k = f"u{k.lstrip('_')}"
                return safe_k
            return k

        return {fix_key(k): _sanitize_query(v) for k, v in query.items()}
    elif isinstance(query, list):
        return [_sanitize_query(x) for x in query]
    return query

logging.basicConfig(level=logging.INFO)

class SafeResult:
    """
    Wraps all MongoDB results in a predictable, serializable object.
    Provides:
      - .success: True/False
      - .data: main result (dict/list/primitive)
      - .error: error string or None
      - .model_dump(): JSON-serializable dict output
      - .original(): original dict(s) with restored keys (for documents)
    """

    def __init__(
            self,
            data: Any = None,
            *,
            success: Optional[bool] = None,
            error: Optional[str] = None,
    ):
        self.success = success if success is not None else (error is None)
        self.error = error
        self.data = self._convert(data)

    @staticmethod
    def _convert(obj):
        # Convert ObjectIds and other BSON types to strings for safe serialization
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, dict):
            return {k: SafeResult._convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SafeResult._convert(x) for x in obj]
        return obj

    @classmethod
    def ok(cls, data: Any):
        return cls(data=data, success=True, error=None)

    @classmethod
    def fail(cls, error: str, data: Any = None):
        return cls(data=data, success=False, error=error)

    def model_dump(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }

    def __repr__(self):
        return f"SafeResult(success={self.success}, error={self.error!r}, data={str(self.data)[:300]})"

    def original(self):
        """
        Reconstitute original dict(s) with any key aliases restored (e.g. 'usecret' -> '_secret').
        Handles both dict and list of dict.
        """

        def restore_one(d):
            # Detect "__keymap" presence and restore original keys
            if isinstance(d, dict) and "__keymap" in d:
                keymap = d.pop("__keymap")
                for safe_k, orig_k in keymap.items():
                    if safe_k in d:
                        d[orig_k] = d.pop(safe_k)
            # Convert string _id back to ObjectId, if possible
            if isinstance(d, dict) and "_id" in d:
                try:
                    d["_id"] = ObjectId(d["_id"])
                except Exception:
                    pass
            return d

        data = self.data
        if isinstance(data, dict):
            return restore_one(data.copy())
        if isinstance(data, list):
            return [restore_one(x.copy()) if isinstance(x, dict) else x for x in data]
        return data

    def to_json(self):
        return json.dumps(self.model_dump(), default=str)

    @staticmethod
    def unwrap(result: 'SafeResult', *, quiet: bool = False):
        """
        Return result.original() on success or raise/return None on failure.

        Args:
            result (SafeResult): The SafeResult instance to unwrap
            quiet (bool): If True, log errors and return None instead of raising

        Returns:
            Any: The original data (with ObjectIds restored) from successful result or None if quiet=True and failed

        Raises:
            RuntimeError: If the result failed and quiet=False
        """
        if result.success:
            return result.original()
        if quiet:
            logging.info("Mongo error: %s", result.error)
            return None
        raise RuntimeError(result.error)

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

class ZMongo:
    """
    High-performance MongoDB client with:
    - SafeResult-wrapped results
    - Automatic Pydantic alias handling
    - Fast TTLCache for repeated _id lookups
    - Only affected cache keys are evicted on update/delete (not full clear)
    """

    def __init__(self, db=None, cache_ttl=DEFAULT_CACHE_TTL):
        """
        Args:
            db: Optionally pass your own Motor database instance (else uses .env)
            cache_ttl: Time-to-live for cache entries, in seconds
        """
        if db:
            self.db = db
        else:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
            mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
            client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = client[mongo_db]
        self._cache: Dict[str, BufferedAsyncTTLCache] = {}
        self._cache_ttl = cache_ttl

    def _norm(self, coll: str) -> str:
        return coll.lower()

    def _get_cache(self, coll: str) -> BufferedAsyncTTLCache:
        """Get or create a TTLCache for the collection. (Synchronous)"""
        norm = self._norm(coll)
        if norm not in self._cache:
            self._cache[norm] = BufferedAsyncTTLCache(ttl=self._cache_ttl)
        return self._cache[norm]

    async def _cput(self, coll: str, query: dict, doc: dict):
        """Store a document in the cache, keyed by _id. (Asynchronous)"""
        cache = self._get_cache(coll)
        key = str(query.get("_id"))
        await cache.set(key, doc)

    async def _cget(self, col: str, q: dict) -> Optional[dict]:
        """Retrieve a document from the cache by _id. (Asynchronous)"""
        cache = self._get_cache(col)
        key = str(q.get("_id"))
        return await cache.get(key)

    async def _cdelete(self, coll: str, query: dict):
        """Evict from cache by _id, or clear whole cache if not _id-based. (Asynchronous)"""
        cache = self._get_cache(coll)
        if "_id" in query:
            await cache.delete(str(query["_id"]))
        else:
            await cache.clear()

    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        """Insert a single document, cache it for fast repeated access."""
        raw = _serialise_doc(document)
        res: InsertOneResult = await self.db[collection].insert_one(raw)
        if cache and res.inserted_id:
            await self._cput(collection, {"_id": res.inserted_id}, _restore_dict({**raw, "_id": res.inserted_id}))
        return SafeResult.ok(res)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        """Insert a list of documents, caching each for fast repeated access."""
        if not documents:
            return SafeResult.ok({"inserted_ids": []})
        raws = [_serialise_doc(d) for d in documents]
        res: InsertManyResult = await self.db[collection].insert_many(raws)
        if cache and res.inserted_ids:
            for raw, _id in zip(raws, res.inserted_ids):
                await self._cput(collection, {"_id": _id}, _restore_dict({**raw, "_id": _id}))
        return SafeResult.ok(res)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        """Find one document matching query. If only _id, use cache. Otherwise query DB."""
        if cache and set(query) == {"_id"}:
            cached = await self._cget(collection, query)
            if cached:
                return SafeResult.ok(cached)
        sanitized_query = _sanitize_query(query)
        doc = await self.db[collection].find_one(sanitized_query)
        if not doc:
            return SafeResult.ok(None)
        doc = _restore_dict(doc)
        if cache and "_id" in doc:
            await self._cput(collection, {"_id": doc["_id"]}, doc)
        return SafeResult.ok(doc)

    async def find_documents(self, collection: str, query: JsonDict, *, limit: int = None,
                             sort: list = None) -> SafeResult:
        """Find all documents matching query, with optional limit/sort. Does not use cache."""
        limit = limit or DEFAULT_QUERY_LIMIT
        sanitized_query = _sanitize_query(query)
        cur = self.db[collection].find(sanitized_query)
        if sort:
            cur = cur.sort(sort)
        docs = []
        async for d in cur.limit(limit):
            docs.append(_restore_dict(d))
        return SafeResult.ok(docs)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *,
                              upsert: bool = False) -> SafeResult:
        """Update one document. Evict only affected cache key."""
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        res: UpdateResult = await self.db[collection].update_one(sanitized_query, upd, upsert=upsert)
        await self._cdelete(collection, query)
        return SafeResult.ok(res)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike, *,
                               upsert: bool = False) -> SafeResult:
        """Batch update, evict only affected keys if not too many; else clear cache."""
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        ids_to_evict = []
        cache = self._get_cache(collection)

        if query and "_id" in query:
            ids_to_evict = [query["_id"]]
        else:
            cursor = self.db[collection].find(sanitized_query, {"_id": 1})
            ids_to_evict = [doc["_id"] async for doc in cursor]
            if len(ids_to_evict) > 1000:
                await cache.clear()
                ids_to_evict = []

        res: UpdateResult = await self.db[collection].update_many(sanitized_query, upd, upsert=upsert)
        for _id in ids_to_evict:
            await cache.delete(str(_id))
        return SafeResult.ok(res)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        """Delete one document by query. Evict only that cache key."""
        sanitized_query = _sanitize_query(query)
        # To ensure we can evict, find the doc first if query is not by _id
        if "_id" not in query:
            doc_to_delete = await self.db[collection].find_one(sanitized_query, {"_id": 1})
            if doc_to_delete:
                query["_id"] = doc_to_delete["_id"]

        res: DeleteResult = await self.db[collection].delete_one(sanitized_query)
        await self._cdelete(collection, query)
        return SafeResult.ok(res)

    async def delete_documents(self, collection: str, query: JsonDict = None) -> SafeResult:
        """Delete all docs matching query. Evict only affected keys, unless huge batch."""
        cache = self._get_cache(collection)
        if query is None:
            await self.db[collection].drop()
            await cache.clear()
            return SafeResult.ok({"dropped": True})

        sanitized_query = _sanitize_query(query)
        ids_to_evict = []
        if query and "_id" in query:
            ids_to_evict = [query["_id"]]
        else:
            cursor = self.db[collection].find(sanitized_query, {"_id": 1})
            ids_to_evict = [doc["_id"] async for doc in cursor]
            if len(ids_to_evict) > 1000:
                await cache.clear()
                ids_to_evict = []

        res: DeleteResult = await self.db[collection].delete_many(sanitized_query)
        for _id in ids_to_evict:
            await cache.delete(str(_id))
        return SafeResult.ok(res)

    async def bulk_write(self, collection: str, ops: List[Any]) -> SafeResult:
        """Bulk write (insert, update, delete ops). Only evict affected docs from cache, else clear."""
        res: BulkWriteResult = await self.db[collection].bulk_write(ops)
        cache = self._get_cache(collection)
        affected_ids = []
        has_non_id_query = False
        for op in ops:
            filt = getattr(op, '_filter', None)
            if filt and '_id' in filt:
                affected_ids.append(filt['_id'])
            else:
                has_non_id_query = True

        if has_non_id_query or len(affected_ids) > 1000:
            await cache.clear()
        else:
            for _id in affected_ids:
                await cache.delete(str(_id))

        return SafeResult.ok(res)

    # --- ADD THIS METHOD ---
    async def clear_cache(self):
        """
        Clears all collection caches.
        """
        for cache in self._cache.values():
            await cache.clear()

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        """Return count of documents matching query."""
        sanitized_query = _sanitize_query(query)
        count = await self.db[collection].count_documents(sanitized_query)
        return SafeResult.ok({"count": count})

    async def list_collections(self) -> SafeResult:
        """Return list of all collection names in DB."""
        names = await self.db.list_collection_names()
        return SafeResult.ok(names)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        """Run an aggregation pipeline on collection, returning up to 'limit' docs."""
        cur = self.db[collection].aggregate(pipeline)
        out = []
        async for d in cur:
            out.append(_restore_dict(d))
            if len(out) >= limit:
                break
        return SafeResult.ok(out)