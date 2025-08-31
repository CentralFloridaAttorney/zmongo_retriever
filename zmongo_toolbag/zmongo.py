"""
ZMongo — Optimized Async MongoDB Helper with Per-Collection TTL Cache
====================================================================

This module provides a lightweight, high-throughput async helper around
**Motor** (MongoDB’s async driver) tailored for retrieval and RAG-style
workloads. It exposes ergonomic CRUD helpers that return a unified
:class:`SafeResult` object, adds a simple per-collection TTL cache for hot
reads, and includes practical production options (projection, hint, batch
size, timeouts, comments, negative‐cache for `_id` misses, etc.).

Highlights
----------
- **API-compatible superset** of a basic repository wrapper (drop-in friendly).
- **Per-collection TTL cache** with negative caching for `_id` misses to
  avoid repeated round-trips for "not found".
- **Smart `_id` normalization**: coerces valid hex strings to `ObjectId`
  in filters (also inside `$in`/`$nin`) to prevent full collection scans.
- **Stringify `_id`** opt-in (useful for JSON/Front-end): controlled via
  env `ZMONGO_STRINGIFY_IDS` (default: on).
- **Motor client tuning**: compressors, pool sizes, and timeouts sourced
  from environment variables or sensible defaults.
- **Consistent results**: all public methods return :class:`SafeResult`.

Environment Variables
---------------------
- ``MONGO_URI`` (default: ``mongodb://127.0.0.1:27017``)
- ``MONGO_DATABASE_NAME`` (default: ``test``)
- ``ZMONGO_CACHE_TTL`` (seconds; default: 300)
- ``ZMONGO_NEGATIVE_CACHE_TTL`` (seconds; default: 30; capped by CACHE_TTL)
- ``ZMONGO_STRINGIFY_IDS`` (“true|false”; default: true)
- Client tuning (all optional): ``ZMONGO_APPNAME``, ``ZMONGO_COMPRESSORS``,
  ``ZMONGO_MAX_POOL``, ``ZMONGO_MIN_POOL``, ``ZMONGO_SRV_SEL_MS``,
  ``ZMONGO_CONNECT_MS``, ``ZMONGO_SOCKET_MS``, ``ZMONGO_RETRY_READS``,
  ``ZMONGO_RETRY_WRITES``.

Return Type
-----------
Every public method returns a :class:`SafeResult`.

- On success: ``SafeResult.ok(data)`` with driver/native fields extracted
  into plain dicts where helpful (see `_parse_mongo_result`).
- On failure: ``SafeResult.fail(message, exc=Exception)``.

Caveats & Notes
---------------
- This module focuses on correctness and ergonomics. It **does not**
  implement complex read-through caching or write coalescing; the TTL cache
  is intentionally simple and collection-scoped.
- For bulk operations we normalize only the **filter** part (ObjectId
  coercion) to preserve your operator payloads intact.
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import motor.motor_asyncio
from dotenv import load_dotenv
from bson import ObjectId
from pymongo.errors import OperationFailure
from pymongo.operations import DeleteMany, DeleteOne, InsertOne, UpdateMany, UpdateOne
from pymongo.results import BulkWriteResult, DeleteResult, InsertManyResult, InsertOneResult, UpdateResult

try:
    from zmongo_toolbag.data_processing import SafeResult
    from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache
except Exception:
    # Fallback to local files if needed (useful during isolated testing)
    from zmongo_toolbag import SafeResult  # type: ignore
    from zmongo_toolbag import BufferedAsyncTTLCache  # type: ignore

# ---------- env & logging ----------
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]
DocsLike = Union[List[DocLike], Any]
MongoOp = Union[InsertOne, DeleteOne, UpdateOne, DeleteMany, UpdateMany]

DEFAULT_QUERY_LIMIT = 100
DEFAULT_CACHE_TTL = int(os.getenv("ZMONGO_CACHE_TTL", "300"))
NEGATIVE_CACHE_TTL = int(os.getenv("ZMONGO_NEGATIVE_CACHE_TTL", "30"))
MISS = object()  # sentinel for negative cache

STRINGIFY_IDS_DEFAULT = os.getenv("ZMONGO_STRINGIFY_IDS", "true").lower() not in {"0", "false", "no"}

# Client tuneables (can also be set in MONGO_URI)
CLIENT_OPTS = {
    "appname": os.getenv("ZMONGO_APPNAME", "ZMongo"),
    "compressors": os.getenv("ZMONGO_COMPRESSORS", "zstd,snappy,zlib"),
    "maxPoolSize": int(os.getenv("ZMONGO_MAX_POOL", "100")),
    "minPoolSize": int(os.getenv("ZMONGO_MIN_POOL", "0")),
    "serverSelectionTimeoutMS": int(os.getenv("ZMONGO_SRV_SEL_MS", "5000")),
    "connectTimeoutMS": int(os.getenv("ZMONGO_CONNECT_MS", "5000")),
    "socketTimeoutMS": int(os.getenv("ZMONGO_SOCKET_MS", "60000")),
    "retryReads": os.getenv("ZMONGO_RETRY_READS", "true").lower() not in {"0", "false", "no"},
    "retryWrites": os.getenv("ZMONGO_RETRY_WRITES", "true").lower() not in {"0", "false", "no"},
}


def _to_objectid_if_valid(value):
    """
    Coerce a value to :class:`bson.ObjectId` when it is a valid hex string.

    Parameters
    ----------
    value : Any
        The input value.

    Returns
    -------
    Any
        ``ObjectId(value)`` if `value` is a valid hex string; otherwise the
        original value (including when it is already an ``ObjectId``).
    """
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        try:
            return ObjectId(value)
        except Exception:
            return value
    return value


def _normalize_ids_in_query(obj):
    """
    Recursively coerce `_id` filters to :class:`ObjectId` when the value is a
    valid hex string. This function safely recurses only into dicts and lists.

    Examples
    --------
    >>> _normalize_ids_in_query({"_id": "650f2c..."})            # -> {"_id": ObjectId(...)}
    >>> _normalize_ids_in_query({"_id": {"$in": ["...", "..."]}})  # -> {"_id": {"$in": [ObjectId(...), ...]}}
    """
    if isinstance(obj, list):
        return [_normalize_ids_in_query(x) for x in obj]
    if not isinstance(obj, dict):
        return obj

    out = {}
    for k, v in obj.items():
        if k == "_id":
            if isinstance(v, dict):
                sub = {}
                for op, subv in v.items():
                    if op in ("$in", "$nin") and isinstance(subv, list):
                        sub[op] = [_to_objectid_if_valid(x) for x in subv]
                    else:
                        sub[op] = _normalize_ids_in_query(subv)
                out[k] = sub
            elif isinstance(v, list):
                out[k] = [_to_objectid_if_valid(x) for x in v]
            else:
                out[k] = _to_objectid_if_valid(v)
        elif isinstance(v, dict):
            out[k] = _normalize_ids_in_query(v)
        elif isinstance(v, list):
            out[k] = [_normalize_ids_in_query(i) for i in v]
        else:
            out[k] = v
    return out


def _now_ms() -> float:
    """
    Current high-resolution time in milliseconds.

    Returns
    -------
    float
        ``time.perf_counter() * 1000.0``
    """
    return time.perf_counter() * 1000.0


def _log_timing(label: str, t0: float, extra: str = "") -> None:
    """
    Emit a DEBUG-level timing log for a code section.

    Parameters
    ----------
    label : str
        Name of the timed operation (e.g., ``"find_one"``).
    t0 : float
        Start time in milliseconds (from :func:`_now_ms`).
    extra : str, optional
        Free-form suffix appended to the message (e.g., counts).
    """
    if logger.isEnabledFor(logging.DEBUG):
        dt = _now_ms() - t0
        logger.debug("%s: %.2f ms %s", label, dt, extra)


def _normalize_projection(projection: Optional[Union[Dict[str, int], List[str]]]) -> Optional[Dict[str, int]]:
    """
    Normalize user projection into Mongo's ``{field: 0|1}`` dict form.

    Accepts either a dict of ``{field: 0|1}`` or a list of field names.

    Returns
    -------
    dict or None
    """
    if projection is None:
        return None
    if isinstance(projection, dict):
        return {str(k): int(v) for k, v in projection.items()}
    if isinstance(projection, list):
        return {str(k): 1 for k in projection}
    raise ValueError("projection must be dict[field->0/1] or list[field]")


def _normalize_hint(hint: Optional[Union[str, List[Tuple[str, int]]]]) -> Optional[Union[str, List[Tuple[str, int]]]]:
    """
    Normalize an index hint.

    Parameters
    ----------
    hint : str | list[tuple[str,int]] | None
        Either an index name or a list of (key, direction) pairs.

    Returns
    -------
    str | list[tuple[str,int]] | None
    """
    if hint is None:
        return None
    if isinstance(hint, str):
        return hint
    if isinstance(hint, list):
        return [(str(k), int(v)) for k, v in hint]
    raise ValueError("hint must be index name (str) or list of (key, direction)")


def _normalize_sort(sort: Optional[Union[Dict[str, int], List[Tuple[str, int]], Tuple[str, int]]]):
    """
    Normalize sort spec into a list of pairs suitable for Motor/PyMongo.

    Accepts:
      - dict: ``{\"field\": 1, \"other\": -1}``
      - list of pairs: ``[(\"field\", 1), (\"other\", -1)]``
      - single pair tuple: ``(\"field\", 1)``

    Returns
    -------
    list[tuple[str,int]] | None
    """
    if sort is None:
        return None
    if isinstance(sort, dict):
        return [(str(k), int(v)) for k, v in sort.items()]
    if isinstance(sort, (list, tuple)):
        if sort and isinstance(sort[0], (list, tuple)):
            return [(str(a), int(b)) for a, b in sort]  # list of pairs
        if len(sort) == 2 and isinstance(sort[0], str):
            return [(str(sort[0]), int(sort[1]))]
    raise ValueError("invalid sort format")


def _stringify_id_in_doc(doc: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    """
    Optionally stringify ``_id`` in a single document (shallow copy).

    Parameters
    ----------
    doc : dict
        Document to convert.
    enabled : bool
        If False, returns the input unchanged.

    Returns
    -------
    dict
    """
    if not enabled or not isinstance(doc, dict):
        return doc
    if "_id" in doc and isinstance(doc["_id"], ObjectId):
        newd = dict(doc)
        newd["_id"] = str(newd["_id"])
        return newd
    return doc


def _stringify_many(docs: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:
    """
    Apply :func:`_stringify_id_in_doc` over a list of documents.

    Parameters
    ----------
    docs : list[dict]
    enabled : bool

    Returns
    -------
    list[dict]
    """
    if not enabled:
        return docs
    out = []
    for d in docs:
        out.append(_stringify_id_in_doc(d, True))
    return out


def _parse_mongo_result(res: Any) -> Dict[str, Any]:
    """
    Convert PyMongo/Motor result objects into plain dicts.

    Returns
    -------
    dict
        A subset of fields relevant to application code (acknowledged, counts,
        inserted ids, upserted id, etc.).
    """
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


# ---------- ZMongo ----------
class ZMongo:
    """
    Thin async repository over Motor with simple per-collection TTL caching.

    Construction
    ------------
    ZMongo connects to the database using ``MONGO_URI`` and selects the database
    from ``MONGO_DATABASE_NAME``. You may also pass an existing Motor
    :class:`AsyncIOMotorDatabase` to take ownership of a preconfigured client.

    Parameters
    ----------
    db : motor.motor_asyncio.AsyncIOMotorDatabase, optional
        Existing Motor database handle. If omitted, a new client is created.
    cache_ttl : int, default 300
        TTL for the per-collection in-memory cache, in seconds.
    stringify_ids : bool, default True
        If True, returned documents have ``_id`` converted to strings.

    Notes
    -----
    - The cache is **collection-scoped**. Writes invalidate either the whole
      collection cache or the specific `_id` entry when applicable.
    - The cache stores a negative sentinel for missing `_id` lookups to
      avoid repeated DB hits on hot "not found" keys.
    """

    def __init__(
        self,
        db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        stringify_ids: bool = STRINGIFY_IDS_DEFAULT,
    ):
        if db is not None:
            self.db = db
        else:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
            mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
            client = motor.motor_asyncio.AsyncIOMotorClient(
                mongo_uri,
                **CLIENT_OPTS,  # compressors, pools, timeouts, retry flags, appname
            )
            self.db = client[mongo_db]

        self._cache_ttl = cache_ttl
        self._neg_ttl = min(NEGATIVE_CACHE_TTL, self._cache_ttl)
        self._stringify_ids = stringify_ids
        self.caches: Dict[str, BufferedAsyncTTLCache] = {}

    async def __aenter__(self) -> "ZMongo":
        """
        Async context manager entry.

        Returns
        -------
        ZMongo
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit; closes underlying Mongo client.
        """
        self.close()

    # Result helpers
    @staticmethod
    def ok(data: Any = None) -> SafeResult:
        """Shorthand for ``SafeResult.ok(data)``."""
        return SafeResult.ok(data)

    @staticmethod
    def fail(error: str, data: Any = None, exc: Optional[Exception] = None) -> SafeResult:
        """Shorthand for ``SafeResult.fail(error, data=data, exc=exc)``."""
        return SafeResult.fail(error, data=data, exc=exc)

    # Cache helpers
    def _get_or_create_cache(self, coll: str) -> BufferedAsyncTTLCache:
        """
        Return the TTL cache for a collection, creating it if absent.
        """
        if coll not in self.caches:
            self.caches[coll] = BufferedAsyncTTLCache(ttl=self._cache_ttl)
        return self.caches[coll]

    async def _cget(self, coll: str, key: Union[str, Any]) -> Optional[Any]:
        """
        Read a value from the collection cache by key.

        Returns
        -------
        Any | None
        """
        cache = self._get_or_create_cache(coll)
        return await cache.get(str(key))

    async def _cput(self, coll: str, key: Union[str, Any], value: Any, *, ttl: Optional[int] = None) -> None:
        """
        Write a value to the collection cache with optional TTL override.
        """
        cache = self._get_or_create_cache(coll)
        await cache.set(str(key), value, ttl=ttl)

    async def _cdelete(self, coll: str, query: Dict) -> None:
        """
        Invalidate cache entries affected by a write operation.

        Strategy
        --------
        - If the write filter contains a concrete `_id`, delete just that key
          (keeps other cached items intact).
        - Otherwise, drop the entire collection cache to avoid stale reads.
        """
        _id = query.get("_id")
        if _id is not None:
            # Invalidate a single document by its ID
            if coll in self.caches:
                oid = _to_objectid_if_valid(_id)
                await self.caches[coll].delete(str(oid))
        else:
            # Invalidate the entire cache for the collection by deleting it
            if coll in self.caches:
                del self.caches[coll]

    # Lifecycle
    def close(self):
        """
        Close the underlying Motor client (idempotent).
        """
        if self.db is not None and self.db.client is not None:
            self.db.client.close()
            logger.info("MongoDB connection closed.")

    def close_connection(self):
        """
        Backwards-compatible alias for :meth:`close`.
        """
        self.close()

    # CRUD
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        """
        Insert a single document.

        Parameters
        ----------
        collection : str
        document : dict | pydantic model | mapping
        cache : bool, default True
            If True, store the inserted document in the per-collection cache.

        Returns
        -------
        SafeResult
            ``{"inserted_id": <ObjectId>, "acknowledged": bool}``
        """
        try:
            t0 = _now_ms()
            doc_dict = self._doc_to_dict(document)
            res = await self.db[collection].insert_one(doc_dict)
            if cache and res.inserted_id:
                await self._cput(collection, res.inserted_id, {**doc_dict, "_id": res.inserted_id})
            _log_timing("insert_one", t0)
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        """
        Insert many documents.

        Parameters
        ----------
        collection : str
        documents : list[dict] | iterable
        cache : bool, default True
            If True, cache each inserted document by its returned `_id`.

        Returns
        -------
        SafeResult
            ``{"inserted_ids": [ObjectId, ...], "acknowledged": bool}``
        """
        if not documents:
            return self.ok({"inserted_ids": [], "acknowledged": True})
        try:
            t0 = _now_ms()
            doc_list = [self._doc_to_dict(doc) for doc in documents]
            res = await self.db[collection].insert_many(doc_list)
            if cache and res.inserted_ids:
                for doc, doc_id in zip(doc_list, res.inserted_ids):
                    await self._cput(collection, doc_id, {**doc, "_id": doc_id})
            _log_timing("insert_many", t0, f"n={len(doc_list)}")
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_document(
        self,
        collection: str,
        query: Dict[str, Any],
        *,
        cache: bool = True,
        projection: Optional[Union[Dict[str, int], List[str]]] = None,
        hint: Optional[Union[str, List[Tuple[str, int]]]] = None,
        max_time_ms: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> SafeResult:
        """
        Find a single document (like ``find_one``) with optional caching and options.

        Caching
        -------
        When the query is exactly ``{"_id": <value>}`` (no operators), a small
        negative cache is used to memoize "not found" for a short TTL.

        Parameters
        ----------
        collection : str
        query : dict
        cache : bool, default True
        projection : dict[str,int] | list[str] | None
        hint : str | list[tuple[str,int]] | None
        max_time_ms : int | None
        comment : str | None

        Returns
        -------
        SafeResult
            ``None`` on miss, or a document (with optional ``_id`` stringified).
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            proj = _normalize_projection(projection)
            use_id_cache = cache and set(norm_query.keys()) == {"_id"} and not isinstance(norm_query["_id"], dict)
            if use_id_cache:
                cached = await self._cget(collection, norm_query["_id"])
                if cached is MISS:
                    _log_timing("find_one (neg-cache)", t0)
                    return self.ok(None)
                if cached is not None:
                    _log_timing("find_one (cache)", t0)
                    return self.ok(_stringify_id_in_doc(cached, self._stringify_ids))

            kw: Dict[str, Any] = {}
            if proj is not None:
                kw["projection"] = proj
            h = _normalize_hint(hint)
            if h is not None:
                kw["hint"] = h
            if max_time_ms is not None:
                kw["max_time_ms"] = int(max_time_ms)
            if comment is not None:
                kw["comment"] = comment

            doc = await self.db[collection].find_one(norm_query, **kw)

            if use_id_cache:
                if doc:
                    await self._cput(collection, doc["_id"], doc)
                else:
                    await self._cput(collection, norm_query["_id"], MISS, ttl=self._neg_ttl)

            _log_timing("find_one", t0)
            return self.ok(_stringify_id_in_doc(doc, self._stringify_ids) if doc else None)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        *,
        limit: int = DEFAULT_QUERY_LIMIT,
        sort: Optional[Union[Dict[str, int], List[Tuple[str, int]], Tuple[str, int]]] = None,
        projection: Optional[Union[Dict[str, int], List[str]]] = None,
        hint: Optional[Union[str, List[Tuple[str, int]]]] = None,
        batch_size: Optional[int] = None,
        max_time_ms: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> SafeResult:
        """
        Find many documents (simple wrapper over Motor cursor with extras).

        Parameters
        ----------
        collection : str
        query : dict
        limit : int, default 100
        sort : dict | list[tuple[str,int]] | tuple[str,int] | None
        projection : dict[str,int] | list[str] | None
        hint : str | list[tuple[str,int]] | None
        batch_size : int | None
        max_time_ms : int | None
        comment : str | None

        Returns
        -------
        SafeResult
            A list of documents (with optional ``_id`` stringified).
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            sort_spec = _normalize_sort(sort)
            proj = _normalize_projection(projection)

            kw: Dict[str, Any] = {}
            if proj is not None:
                kw["projection"] = proj
            h = _normalize_hint(hint)
            if h is not None:
                kw["hint"] = h
            if batch_size is not None:
                kw["batch_size"] = int(batch_size)
            if max_time_ms is not None:
                kw["max_time_ms"] = int(max_time_ms)
            if comment is not None:
                kw["comment"] = comment

            cursor = self.db[collection].find(norm_query, **kw)
            if sort_spec:
                cursor = cursor.sort(sort_spec)

            docs = await cursor.to_list(length=limit)
            docs = _stringify_many(docs, self._stringify_ids)
            _log_timing("find_many", t0, f"n={len(docs)}")
            return self.ok(docs)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False) -> SafeResult:
        """
        Update a single document.

        If `update_data` does not contain an update operator (e.g., ``$set``),
        it is wrapped as ``{"$set": update_data}`` for convenience.

        Parameters
        ----------
        collection : str
        query : dict
        update_data : dict | pydantic model
        upsert : bool, default False

        Returns
        -------
        SafeResult
            Includes ``matched_count``, ``modified_count``, and ``upserted_id``.
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_one(norm_query, update_dict, upsert=upsert)
            await self._cdelete(collection, norm_query)
            _log_timing("update_one", t0)
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike) -> SafeResult:
        """
        Update many documents (see :meth:`update_document` for operator wrapping).
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_many(norm_query, update_dict)
            await self._cdelete(collection, {})
            _log_timing("update_many", t0)
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        """
        Delete a single document matching the filter.
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            res = await self.db[collection].delete_one(norm_query)
            await self._cdelete(collection, norm_query)
            _log_timing("delete_one", t0)
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_documents(self, collection: str, query: JsonDict) -> SafeResult:
        """
        Delete many documents matching the filter.
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            res = await self.db[collection].delete_many(norm_query)
            await self._cdelete(collection, {})
            _log_timing("delete_many", t0)
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def bulk_write(self, collection: str, ops: List[MongoOp]) -> SafeResult:
        """
        Execute a bulk write sequence (ops may be InsertOne/UpdateOne/etc.).

        Normalization
        -------------
        - For operations that have a private ``_filter`` attribute (e.g., UpdateOne),
          that filter is normalized via :func:`_normalize_ids_in_query` to ensure
          valid `_id` hex strings are coerced to :class:`ObjectId`.

        Cache Invalidation
        ------------------
        - Entire collection cache is invalidated after a bulk write, since the
          affected set is typically unknown.

        Returns
        -------
        SafeResult
            A dict with the usual bulk write counters.
        """
        if not ops:
            return self.ok({"acknowledged": True})
        try:
            t0 = _now_ms()
            # Modify the filter in-place for applicable operations.
            for op in ops:
                if hasattr(op, "_filter"):
                    op._filter = _normalize_ids_in_query(op._filter)

            res = await self.db[collection].bulk_write(ops)
            await self._cdelete(collection, {})  # Invalidate the entire collection's cache
            _log_timing("bulk_write", t0, f"ops={len(ops)}")
            return self.ok(_parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    # Misc
    async def count_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        *,
        hint: Optional[Union[str, List[Tuple[str, int]]]] = None,
        max_time_ms: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> SafeResult:
        """
        Count documents matching a query.

        Parameters
        ----------
        collection : str
        query : dict
        hint : str | list[tuple[str,int]] | None
        max_time_ms : int | None
        comment : str | None

        Returns
        -------
        SafeResult
            ``{"count": <int>}``
        """
        try:
            t0 = _now_ms()
            norm_query = _normalize_ids_in_query(query)
            kw: Dict[str, Any] = {}
            if hint is not None:
                kw["hint"] = _normalize_hint(hint)
            if max_time_ms is not None:
                kw["maxTimeMS"] = int(max_time_ms)
            if comment:
                kw["comment"] = comment

            try:
                count = await self.db[collection].count_documents(norm_query, **kw)
            except TypeError:
                # Some server/driver combos did not accept 'comment' historically
                if "comment" in kw:
                    kw.pop("comment", None)
                    count = await self.db[collection].count_documents(norm_query, **kw)
                else:
                    raise

            _log_timing("count_documents", t0)
            return self.ok({"count": int(count)})

        except Exception as e:
            # Fallback: estimated count for whole collection when query is empty
            try:
                if not query:
                    est = await self.db[collection].estimated_document_count()
                    return self.ok({"count": int(est)})
            except Exception:
                pass
            return self.fail(str(e), exc=e)

    async def list_collections(self) -> SafeResult:
        """
        List collection names in the current database.

        Returns
        -------
        SafeResult
            ``[str, ...]``
        """
        try:
            t0 = _now_ms()
            names = await self.db.list_collection_names()
            _log_timing("list_collections", t0)
            return self.ok(names)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]],
        *,
        limit: int = 1000,
        allow_disk_use: Optional[bool] = None,
        max_time_ms: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> SafeResult:
        """
        Run an aggregation pipeline with helpful options and id normalization in $match.

        Behavior
        --------
        - Any ``$match`` stage has its filter normalized by
          :func:`_normalize_ids_in_query` to coerce `_id` hex strings.

        Parameters
        ----------
        collection : str
        pipeline : list[dict]
        limit : int, default 1000
        allow_disk_use : bool | None
        max_time_ms : int | None
        comment : str | None

        Returns
        -------
        SafeResult
            A list of documents (with optional ``_id`` stringified).
        """
        try:
            t0 = _now_ms()
            norm_pipeline: List[Dict[str, Any]] = []
            for stage in pipeline:
                if "$match" in stage and isinstance(stage["$match"], dict):
                    norm_pipeline.append({"$match": _normalize_ids_in_query(stage["$match"])})
                else:
                    norm_pipeline.append(stage)

            agg_kwargs: Dict[str, Any] = {}
            if allow_disk_use is not None:
                agg_kwargs["allowDiskUse"] = bool(allow_disk_use)
            if max_time_ms is not None:
                agg_kwargs["maxTimeMS"] = int(max_time_ms)
            if comment is not None:
                agg_kwargs["comment"] = comment

            cursor = self.db[collection].aggregate(norm_pipeline, **agg_kwargs)
            docs = await cursor.to_list(length=limit)
            docs = _stringify_many(docs, self._stringify_ids)

            _log_timing("aggregate", t0, f"n={len(docs)}")
            return self.ok(docs)

        except OperationFailure:
            # bubble up operation failures (e.g., invalid pipeline)
            raise
        except Exception as e:
            return self.fail(str(e), exc=e)

    # Utility
    @staticmethod
    def _doc_to_dict(doc: DocLike) -> Dict:
        """
        Convert a pydantic model or mapping into a plain dict.

        - If the object has ``model_dump(by_alias=True)``, use it (Pydantic v2).
        - Else if it has ``dict(by_alias=True)``, use it (Pydantic v1).
        - Else, return the object as-is (assumed to be a ``dict``).

        Returns
        -------
        dict
        """
        if hasattr(doc, "model_dump"):
            return doc.model_dump(by_alias=True)
        if hasattr(doc, "dict"):
            return doc.dict(by_alias=True)
        return doc

    async def ping(self) -> SafeResult:
        """
        Lightweight server health check.

        Returns
        -------
        SafeResult
            ``{"ok": 1}`` on success.
        """
        try:
            t0 = _now_ms()
            await self.db.command("ping")
            _log_timing("ping", t0)
            return self.ok({"ok": 1})
        except Exception as e:
            return self.fail(str(e), exc=e)
