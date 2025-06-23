import os
from typing import Any, Optional, List, Dict, Union
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult, BulkWriteResult
from zmongo_toolbag.utils.safe_result import SafeResult
from zmongo_toolbag.utils.ttl_cache import TTLCache

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
        import motor.motor_asyncio
        mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
        client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = client[mongo_db]
        self._cache = {}
        self._cache_ttl = cache_ttl

    def _norm(self, coll: str) -> str:
        return coll.lower()

    def _get_cache(self, coll: str) -> TTLCache:
        """Get or create a TTLCache for the collection."""
        norm = self._norm(coll)
        if norm not in self._cache:
            self._cache[norm] = TTLCache(ttl=self._cache_ttl)
        return self._cache[norm]

    def _cput(self, coll: str, query: dict, doc: dict):
        """Store a document in the cache, keyed by _id."""
        cache = self._get_cache(coll)
        key = str(query.get("_id"))
        cache.set(key, doc)

    def _cget(self, col: str, q: dict) -> Optional[dict]:
        """Retrieve a document from the cache by _id."""
        cache = self._get_cache(col)
        key = str(q.get("_id"))
        return cache.get(key)

    def _cdelete(self, coll: str, query: dict):
        """Evict from cache by _id, or clear whole cache if not _id-based."""
        cache = self._get_cache(coll)
        if "_id" in query:
            cache.delete(str(query["_id"]))
        else:
            cache.clear()

    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        """
        Insert a single document, cache it for fast repeated access.
        Args:
            collection: Name of collection
            document: Dict, pydantic, or object with dict/model_dump
            cache: Whether to cache inserted doc
        Returns: SafeResult wrapping InsertOneResult
        """
        raw = _serialise_doc(document)
        res: InsertOneResult = await self.db[collection].insert_one(raw)
        if cache:
            self._cput(collection, {"_id": res.inserted_id}, _restore_dict({**raw, "_id": res.inserted_id}))
        return SafeResult.ok(res)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        """
        Insert a list of documents, caching each for fast repeated access.
        Args:
            collection: Name of collection
            documents: List of dicts, pydantic, etc.
            cache: Whether to cache inserted docs
        Returns: SafeResult wrapping InsertManyResult
        """
        if not documents:
            return SafeResult.ok({"inserted_ids": []})
        raws = [_serialise_doc(d) for d in documents]
        res: InsertManyResult = await self.db[collection].insert_many(raws)
        if cache:
            for raw, _id in zip(raws, res.inserted_ids):
                self._cput(collection, {"_id": _id}, _restore_dict({**raw, "_id": _id}))
        return SafeResult.ok(res)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        """
        Find one document matching query. If only _id, use cache. Otherwise query DB.
        Args:
            collection: Name of collection
            query: Query dict (sanitized)
            cache: Whether to cache result
        Returns: SafeResult[doc]
        """
        if set(query) == {"_id"}:
            cached = self._cget(collection, query)
            if cached:
                return SafeResult.ok(cached)
        sanitized_query = _sanitize_query(query)
        doc = await self.db[collection].find_one(sanitized_query)
        if not doc:
            return SafeResult.ok(None)
        doc = _restore_dict(doc)
        if cache and "_id" in doc:
            self._cput(collection, {"_id": doc["_id"]}, doc)
        return SafeResult.ok(doc)

    async def find_documents(self, collection: str, query: JsonDict, *, limit: int = None, sort: list = None) -> SafeResult:
        """
        Find all documents matching query, with optional limit/sort. Does not use cache.
        Args:
            collection: Name of collection
            query: Query dict
            limit: Max docs to return (default 100)
            sort: Optional sort (MongoDB syntax)
        Returns: SafeResult[list[doc]]
        """
        limit = limit or DEFAULT_QUERY_LIMIT
        sanitized_query = _sanitize_query(query)
        cur = self.db[collection].find(sanitized_query)
        if sort:
            cur = cur.sort(sort)
        docs = []
        async for d in cur.limit(limit):
            docs.append(_restore_dict(d))
        return SafeResult.ok(docs)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False) -> SafeResult:
        """
        Update one document. Evict only affected cache key.
        Args:
            collection: Name of collection
            query: Query dict (sanitized)
            update_data: Dict/obj of new values
            upsert: Insert if not found
        Returns: SafeResult[UpdateResult]
        """
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        res: UpdateResult = await self.db[collection].update_one(sanitized_query, upd, upsert=upsert)
        self._cdelete(collection, query)
        return SafeResult.ok(res)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False) -> SafeResult:
        """
        Batch update, evict only affected keys if not too many; else clear cache.
        Args:
            collection: Name of collection
            query: Query dict
            update_data: Dict/obj of new values
            upsert: Insert if not found
        Returns: SafeResult[UpdateResult]
        """
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        ids = []
        if query and "_id" in query:
            ids = [query["_id"]]
        else:
            cursor = self.db[collection].find(sanitized_query, {"_id": 1})
            ids = [doc["_id"] async for doc in cursor]
            if len(ids) > 1000:
                self._get_cache(collection).clear()
                ids = []
        res: UpdateResult = await self.db[collection].update_many(sanitized_query, upd, upsert=upsert)
        cache = self._get_cache(collection)
        for _id in ids:
            cache.delete(str(_id))
        return SafeResult.ok(res)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        """
        Delete one document by query. Evict only that cache key.
        Args:
            collection: Name of collection
            query: Query dict
        Returns: SafeResult[DeleteResult]
        """
        sanitized_query = _sanitize_query(query)
        res: DeleteResult = await self.db[collection].delete_one(sanitized_query)
        self._cdelete(collection, query)
        return SafeResult.ok(res)

    async def delete_documents(self, collection: str, query: JsonDict = None) -> SafeResult:
        """
        Delete all docs matching query. Evict only affected keys, unless huge batch.
        Args:
            collection: Name of collection
            query: Query dict (or None for drop)
        Returns: SafeResult[DeleteResult]
        """
        if query is None:
            await self.db[collection].drop()
            self._get_cache(collection).clear()
            return SafeResult.ok({"dropped": True})
        sanitized_query = _sanitize_query(query)
        ids = []
        if query and "_id" in query:
            ids = [query["_id"]]
        else:
            cursor = self.db[collection].find(sanitized_query, {"_id": 1})
            ids = [doc["_id"] async for doc in cursor]
            if len(ids) > 1000:
                self._get_cache(collection).clear()
                ids = []
        res: DeleteResult = await self.db[collection].delete_many(sanitized_query)
        cache = self._get_cache(collection)
        for _id in ids:
            cache.delete(str(_id))
        return SafeResult.ok(res)

    async def bulk_write(self, collection: str, ops: List[Any]) -> SafeResult:
        """
        Bulk write (insert, update, delete ops). Only evict affected docs from cache, else clear.
        Args:
            collection: Name of collection
            ops: List of pymongo InsertOne, UpdateOne, DeleteOne ops
        Returns: SafeResult[BulkWriteResult]
        """
        res: BulkWriteResult = await self.db[collection].bulk_write(ops)
        affected_ids = []
        for op in ops:
            if hasattr(op, '_filter') and '_id' in op._filter:
                affected_ids.append(op._filter['_id'])
        cache = self._get_cache(collection)
        for _id in affected_ids:
            cache.delete(str(_id))
        if not affected_ids:
            cache.clear()
        return SafeResult.ok(res)

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        """
        Return count of documents matching query.
        Args:
            collection: Name of collection
            query: Query dict
        Returns: SafeResult[int]
        """
        sanitized_query = _sanitize_query(query)
        count = await self.db[collection].count_documents(sanitized_query)
        return SafeResult.ok({"count": count})

    async def list_collections(self) -> SafeResult:
        """
        Return list of all collection names in DB.
        Returns: SafeResult[list[str]]
        """
        names = await self.db.list_collection_names()
        return SafeResult.ok(names)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        """
        Run an aggregation pipeline on collection, returning up to 'limit' docs.
        Args:
            collection: Name of collection
            pipeline: List of aggregation pipeline stages
            limit: Max number of docs to return
        Returns: SafeResult[list[dict]]
        """
        cur = self.db[collection].aggregate(pipeline)
        out = []
        async for d in cur:
            out.append(_restore_dict(d))
            if len(out) >= limit:
                break
        return SafeResult.ok(out)
