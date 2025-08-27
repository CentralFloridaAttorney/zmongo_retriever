"""
ZMongo — Minimal Async MongoDB Helper with Per‑Collection TTL Cache
=================================================================

`ZMongo` is a thin, asyncio‑friendly wrapper around Motor/PyMongo that standardizes
CRUD operations and wraps results in a simple `SafeResult` container. It also
provides a small, per‑collection **TTL cache** to reduce round‑trips for common
lookups.

Highlights
---------
- Async CRUD: `insert_*`, `find_*`, `update_*`, `delete_*`, `bulk_write`
- Result shaping: always returns `SafeResult.ok(data)` / `SafeResult.fail(error)`
- ObjectId ergonomics: accepts 24‑char hex strings and converts them in queries
- Per‑collection buffered TTL cache (opt‑in on reads, auto‑invalidated on writes)
- Deterministic parsing of PyMongo result objects (InsertOneResult, UpdateResult, …)

Quick Start
-----------
```python
import asyncio
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo

async def main():
    repo = ZMongo()  # reads MONGO_URI and MONGO_DATABASE_NAME from ~/.resources/.env_fleet

    # Insert a document
    _id = ObjectId()
    ins = await repo.insert_document("kb", {"_id": _id, "text": "Hello, Mongo."})
    assert ins.success, ins.error

    # Find it (uses cache by default when _id is present)
    got = await repo.find_document("kb", {"_id": _id})
    assert got.success and got.data["_id"] == str(_id)

    # Update it
    upd = await repo.update_document("kb", {"_id": _id}, {"$set": {"text": "Updated."}})
    assert upd.success

    # Delete it
    dele = await repo.delete_document("kb", {"_id": _id})
    assert dele.success

    repo.close()

asyncio.run(main())
```

Environment
-----------
- `MONGO_URI` (default: `mongodb://127.0.0.1:27017`)
- `MONGO_DATABASE_NAME` (default: `test`)

Both are read from `~/resources/.env_fleet` if present (via `python-dotenv`).

Notes on Caching
----------------
- Each collection gets its **own** `BufferedAsyncTTLCache` instance.
- `find_document(..., cache=True)` caches by `_id` only.
- Any multi‑document write (`update_many`, `delete_many`, `bulk_write`) clears the
  entire collection cache to avoid stale reads.

"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import motor.motor_asyncio
from dotenv import load_dotenv
from bson import ObjectId
from pymongo.errors import OperationFailure
from pymongo.operations import (
    DeleteMany,
    DeleteOne,
    InsertOne,
    UpdateMany,
    UpdateOne,
)
from pymongo.results import (
    BulkWriteResult,
    DeleteResult,
    InsertManyResult,
    InsertOneResult,
    UpdateResult,
)

from zmongo_toolbag.data_processing import SafeResult
# Local import for type hinting
from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache


load_dotenv(Path.home() / "resources" / ".env_fleet")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]
DocsLike = Union[List[DocLike], Any]
MongoOp = Union[InsertOne, DeleteOne, UpdateOne, DeleteMany, UpdateMany]

DEFAULT_QUERY_LIMIT = 100
DEFAULT_CACHE_TTL = 300


class ZMongo:
    """Async convenience wrapper around Motor with per‑collection TTL caching.

    Parameters
    ----------
    db : motor.motor_asyncio.AsyncIOMotorDatabase, optional
        An existing Motor database instance. When omitted, a new client/database
        is created from `MONGO_URI` and `MONGO_DATABASE_NAME` environment vars.
    cache_ttl : int, optional
        Default TTL (seconds) for the per‑collection caches. Default is 300.

    Examples
    --------
    Basic lifecycle and CRUD:

    >>> import asyncio
    >>> from bson import ObjectId
    >>> from zmongo_toolbag.zmongo import ZMongo
    >>> async def demo():
    ...     repo = ZMongo()
    ...     _id = ObjectId()
    ...     ins = await repo.insert_document("kb", {"_id": _id, "text": "Hello"})
    ...     assert ins.success
    ...     got = await repo.find_document("kb", {"_id": _id})
    ...     assert got.success and got.data["_id"] == str(_id)
    ...     upd = await repo.update_document("kb", {"_id": _id}, {"$set": {"text": "Hi"}})
    ...     assert upd.success
    ...     dele = await repo.delete_document("kb", {"_id": _id})
    ...     assert dele.success
    ...     repo.close()
    >>> asyncio.run(demo())
    """

    def __init__(
        self,
        db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ):
        if db is not None:
            self.db = db
        else:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
            mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
            client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = client[mongo_db]

        self._cache_ttl = cache_ttl
        # Per‑collection cache registry
        self.caches: Dict[str, "BufferedAsyncTTLCache"] = {}

    async def __aenter__(self) -> "ZMongo":
        """Enable `async with ZMongo() as repo:` usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the underlying MongoDB client at context exit."""
        self.close()

    # ---------- Result helpers ----------
    @staticmethod
    def ok(data: Any = None) -> "SafeResult":
        """Return a successful `SafeResult` with `data` payload."""
        return SafeResult.ok(data)

    @staticmethod
    def fail(error: str, data: Any = None, exc: Optional[Exception] = None) -> "SafeResult":
        """Return a failed `SafeResult` with message and optional `data`/exception."""
        return SafeResult.fail(error, data=data, exc=exc)

    # ---------- ObjectId helpers ----------
    @staticmethod
    def _to_objectid(value: Any) -> Any:
        """Convert 24‑hex string to `ObjectId` when appropriate; else return original.

        This helper is used by `_normalize_ids_in_query` to allow passing strings
        in queries while still hitting `_id` indexes correctly.
        """
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str) and len(value) == 24:
            try:
                return ObjectId(value)
            except Exception:
                return value
        return value

    @classmethod
    def _normalize_ids_in_query(cls, obj: Any) -> Any:
        """Recursively coerce `_id` filters inside a Mongo query to `ObjectId`.

        Handles simple equality, `$in`/`$nin`, and logical ops like `$or`/`$and`.
        Non‑dict inputs are passed through unchanged.
        """
        if isinstance(obj, list):
            return [cls._normalize_ids_in_query(x) for x in obj]
        if not isinstance(obj, dict):
            return obj

        normalized: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "_id":
                if isinstance(v, list):
                    normalized[k] = [cls._to_objectid(x) for x in v]
                elif isinstance(v, dict):
                    sub = {}
                    for op, subv in v.items():
                        if op in ("$in", "$nin") and isinstance(subv, list):
                            sub[op] = [cls._to_objectid(x) for x in subv]
                        else:
                            sub[op] = cls._normalize_ids_in_query(subv)
                    normalized[k] = sub
                else:
                    normalized[k] = cls._to_objectid(v)
            elif k in ("$or", "$and", "$nor") and isinstance(v, list):
                normalized[k] = [cls._normalize_ids_in_query(x) for x in v]
            else:
                normalized[k] = cls._normalize_ids_in_query(v)
        return normalized

    @staticmethod
    def _stringify_id_in_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy with `_id` converted to `str`, if present.

        Useful for JSON‑safe payloads in APIs and tests that compare by string.
        """
        if not isinstance(doc, dict):
            return doc
        if "_id" in doc and isinstance(doc["_id"], ObjectId):
            newd = dict(doc)
            newd["_id"] = str(newd["_id"])
            return newd
        return doc

    # ---------- Cache helpers (Per‑Collection) ----------
    def _get_or_create_cache(self, coll: str) -> "BufferedAsyncTTLCache":
        """Return the cache for `coll`, creating it on first use.

        Each collection maintains an isolated TTL cache to prevent cross‑pollution
        of cached entries across different collections.
        """
        if coll not in self.caches:
            from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache
            self.caches[coll] = BufferedAsyncTTLCache(ttl=self._cache_ttl)
        return self.caches[coll]

    async def _cget(self, coll: str, key: Union[str, Any]) -> Optional[Any]:
        """Get an item from the collection cache by key (usually `_id`)."""
        cache = self._get_or_create_cache(coll)
        return await cache.get(str(key))

    async def _cput(self, coll: str, key: Union[str, Any], value: Any) -> None:
        """Put an item into the collection cache under `key`."""
        cache = self._get_or_create_cache(coll)
        await cache.set(str(key), value, ttl=self._cache_ttl)

    async def _cdelete(self, coll: str, query: Dict) -> None:
        """Invalidate cache entries affected by `query`.

        If the query targets a specific `_id`, only that key is removed. For
        multi‑document operations, the entire collection cache is dropped to
        avoid stale reads.
        """
        _id = query.get("_id")
        if _id is not None:
            if coll in self.caches:
                cache = self.caches[coll]
                oid = self._to_objectid(_id)
                await cache.delete(str(oid))
        else:
            if coll in self.caches:
                del self.caches[coll]

    # ---------- Utility ----------
    @staticmethod
    def _doc_to_dict(doc: DocLike) -> Dict:
        """Convert a Pydantic/model instance to `dict`, preserving aliases.

        Falls back to the input when no model API is present.
        """
        if hasattr(doc, "model_dump"):
            return doc.model_dump(by_alias=True)
        if hasattr(doc, "dict"):
            return doc.dict(by_alias=True)
        return doc

    @staticmethod
    def _parse_mongo_result(res: Any) -> Dict[str, Any]:
        """Normalize PyMongo result objects into serializable dicts."""
        if isinstance(res, InsertOneResult):
            return {"inserted_id": res.inserted_id, "acknowledged": res.acknowledged}
        if isinstance(res, InsertManyResult):
            return {"inserted_ids": res.inserted_ids, "acknowledged": res.acknowledged}
        if isinstance(res, UpdateResult):
            return {
                "matched_count": res.matched_count,
                "modified_count": res.modified_count,
                "upserted_id": res.upserted_id,
                "acknowledged": res.acknowledged,
            }
        if isinstance(res, DeleteResult):
            return {"deleted_count": res.deleted_count, "acknowledged": res.acknowledged}
        if isinstance(res, BulkWriteResult):
            return {
                "inserted_count": res.inserted_count,
                "matched_count": res.matched_count,
                "modified_count": res.modified_count,
                "deleted_count": res.deleted_count,
                "upserted_count": res.upserted_count,
                "acknowledged": res.acknowledged,
            }
        return {"raw_result": str(res)}

    def close(self):
        """Close the underlying Motor client and log a message."""
        if self.db is not None and self.db.client is not None:
            self.db.client.close()
            logger.info("MongoDB connection closed.")

    def close_connection(self):
        """Backward‑compat alias for :meth:`close`."""
        self.close()

    # ---------- CRUD ----------
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        """Insert a single document.

        Parameters
        ----------
        collection : str
            Collection name.
        document : dict or model
            A JSON‑serializable dict or a Pydantic/model instance.
        cache : bool, optional
            When True (default), the inserted document is cached by its `_id`.

        Returns
        -------
        SafeResult
            `.data` contains `{inserted_id, acknowledged}` on success.
        """
        try:
            doc_dict = self._doc_to_dict(document)
            res = await self.db[collection].insert_one(doc_dict)
            if cache and res.inserted_id:
                await self._cput(collection, res.inserted_id, {**doc_dict, "_id": res.inserted_id})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        """Insert many documents in a single call.

        Empty input returns a successful result with an empty `inserted_ids` list.
        When `cache=True`, each inserted doc is added to the collection cache.
        """
        if not documents:
            return self.ok({"inserted_ids": [], "acknowledged": True})
        try:
            doc_list = [self._doc_to_dict(doc) for doc in documents]
            res = await self.db[collection].insert_many(doc_list)
            if cache and res.inserted_ids:
                for doc, doc_id in zip(doc_list, res.inserted_ids):
                    await self._cput(collection, doc_id, {**doc, "_id": doc_id})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        """Find a single document by query.

        If `cache=True` and the query is an equality on `_id`, a cached copy is
        returned when present. Found documents are cached by `_id` automatically.
        Returns a JSON‑safe doc with `_id` stringified.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            if cache and "_id" in norm_query:
                cached = await self._cget(collection, norm_query["_id"])
                if cached is not None:
                    return self.ok(self._stringify_id_in_doc(cached))
            doc = await self.db[collection].find_one(norm_query)
            if doc and cache and "_id" in doc:
                await self._cput(collection, doc["_id"], doc)
            return self.ok(self._stringify_id_in_doc(doc) if doc else None)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_documents(
        self,
        collection: str,
        query: JsonDict,
        *,
        limit: int = DEFAULT_QUERY_LIMIT,
        sort: Optional[List[Tuple[str, int]]] = None,
    ) -> SafeResult:
        """Find multiple documents by query, with optional sort and limit.

        Parameters
        ----------
        limit : int, optional
            Maximum number of documents to return. Default 100.
        sort : list[tuple[str, int]], optional
            PyMongo sort spec, e.g. `[('created_at', -1)]`.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            cursor = self.db[collection].find(norm_query)
            if sort:
                cursor = cursor.sort(sort)
            docs = await cursor.to_list(length=limit)
            docs = [self._stringify_id_in_doc(d) for d in docs]
            return self.ok(docs)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_document(
        self,
        collection: str,
        query: JsonDict,
        update_data: DocLike,
        *,
        upsert: bool = False,
    ) -> SafeResult:
        """Update a single document matching `query`.

        Accepts either a full Mongo update spec (with `$set`, `$inc`, etc.) or a
        plain object which will be wrapped in `{"$set": ...}`.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_one(norm_query, update_dict, upsert=upsert)
            await self._cdelete(collection, norm_query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike) -> SafeResult:
        """Update many documents at once.

        On success, the entire collection cache is invalidated to avoid stale
        entries. Accepts either a Mongo update spec or a plain object.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_many(norm_query, update_dict)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        """Delete a single document matching `query`.

        Invalidates the cached entry for the document if `_id` is present in the
        query.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            res = await self.db[collection].delete_one(norm_query)
            await self._cdelete(collection, norm_query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_documents(self, collection: str, query: JsonDict) -> SafeResult:
        """Delete multiple documents matching `query`.

        On success, clears the entire collection cache.
        """
        try:
            norm_query = self._normalize_ids_in_query(query)
            res = await self.db[collection].delete_many(norm_query)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def bulk_write(self, collection: str, ops: List[MongoOp]) -> SafeResult:
        """Perform a bulk write with PyMongo operations (InsertOne/UpdateOne/etc.).

        On success, clears the collection cache.
        """
        try:
            res = await self.db[collection].bulk_write(ops)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    # ---------- Misc ----------
    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        """Return the count of documents matching `query`."""
        try:
            norm_query = self._normalize_ids_in_query(query)
            count = await self.db[collection].count_documents(norm_query)
            return self.ok({"count": count})
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def list_collections(self) -> SafeResult:
        """List collection names in the current database."""
        try:
            names = await self.db.list_collection_names()
            return self.ok(names)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        """Run an aggregation pipeline and return a list of documents.

        `$match` stages are normalized via `_normalize_ids_in_query` so 24‑hex
        `_id` strings are treated as real `ObjectId`s. Documents are returned
        with `_id` stringified for JSON‑safety.
        """
        try:
            norm_pipeline: List[JsonDict] = []
            for stage in pipeline:
                if "$match" in stage and isinstance(stage["$match"], dict):
                    norm_pipeline.append({"$match": self._normalize_ids_in_query(stage["$match"])})
                else:
                    norm_pipeline.append(stage)
            cursor = self.db[collection].aggregate(norm_pipeline)
            docs = await cursor.to_list(length=limit)
            docs = [self._stringify_id_in_doc(d) for d in docs]
            return self.ok(docs)
        except OperationFailure as e:
            # Allow callers/tests to assert on Mongo errors explicitly
            raise e
        except Exception as e:
            return self.fail(str(e), exc=e)
