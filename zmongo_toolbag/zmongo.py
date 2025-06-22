from __future__ import annotations

import hashlib, json, logging, os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel
from pymongo.results import (
    BulkWriteResult, DeleteResult, InsertManyResult,
    InsertOneResult, UpdateResult,
)

from zmongo_toolbag.utils.safe_result import SafeResult
from zmongo_toolbag.utils.ttl_cache import TTLCache

load_dotenv()
DEFAULT_QUERY_LIMIT: int = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zmongo")

JsonDict   = Dict[str, Any]
DocLike    = Union[JsonDict, BaseModel]
DocsLike   = Sequence[DocLike]

_KEYMAP_FIELD = "__keymap"

def _needs_fix(key: str) -> bool:
    return key.startswith("_") and key != "_id"

def _sanitize_dict(d: JsonDict) -> JsonDict:
    if not any(_needs_fix(k) for k in d):
        return d
    fixed, keymap = {}, {}
    for k, v in d.items():
        if _needs_fix(k):
            safe_k = f"u{k.lstrip('_')}"
            while safe_k in d:
                safe_k = f"u{safe_k}"
            keymap[safe_k] = k
            fixed[safe_k] = v
        else:
            fixed[k] = v
    fixed[_KEYMAP_FIELD] = keymap
    return fixed

def _restore_dict(d: JsonDict) -> JsonDict:
    if _KEYMAP_FIELD not in d:
        return d
    keymap = d.pop(_KEYMAP_FIELD, {})
    for safe_k, orig_k in keymap.items():
        if safe_k in d:
            d[orig_k] = d.pop(safe_k)
    return d

def _serialise_doc(doc: DocLike) -> JsonDict:
    if isinstance(doc, BaseModel):
        doc = doc.model_dump(by_alias=True, exclude_none=True)
    return _sanitize_dict(doc)

from typing import Any, Optional, List, Dict, Union
from bson import ObjectId
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult

JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]  # Accepts dicts and pydantic/BaseModel-like objects


def _sanitize_query(query: Any) -> Any:
    """Sanitize query dict keys the same as for storage, recursively."""
    if isinstance(query, dict):
        def fix_key(k):
            # Only fix keys that are field names, not operators like $set, $gte, etc.
            if k.startswith("$"):
                return k
            if _needs_fix(k):
                safe_k = f"u{k.lstrip('_')}"
                # (could add collision avoidance if needed)
                return safe_k
            return k
        return {fix_key(k): _sanitize_query(v) for k, v in query.items()}
    elif isinstance(query, list):
        return [_sanitize_query(x) for x in query]
    return query

class ZMongo:
    def __init__(self, db=None):
        import motor.motor_asyncio
        client = motor.motor_asyncio.AsyncIOMotorClient()
        self.db = client["test"]
        self._cache = defaultdict(lambda: TTLCache(ttl=600))  # 10 min TTL by default

    def _norm(self, coll: str) -> str:
        return coll.lower()

    def _cput(self, coll: str, query: dict, doc: dict):
        norm = self._norm(coll)
        cache = self._cache[norm]
        key = str(query.get("_id"))
        cache.set(key, doc)

    def _cget(self, col: str, q: dict) -> Optional[dict]:
        cache = self._cache[self._norm(col)]
        key = str(q.get("_id"))
        return cache.get(key)

    # For all cache-clearing code, simply:
    def clear_collection_cache(self, collection):
        self._cache[self._norm(collection)].clear()

    async def bulk_write(self, collection: str, ops: List[Any]) -> SafeResult:
        res: BulkWriteResult = await self.db[collection].bulk_write(ops)
        self.clear_collection_cache(collection)
        return SafeResult.ok(res)


    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        raw = _serialise_doc(document)
        res: InsertOneResult = await self.db[collection].insert_one(raw)
        if cache:
            self._cput(collection, {"_id": res.inserted_id}, _restore_dict({**raw, "_id": res.inserted_id}))
        return SafeResult.ok(res)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        if not documents:
            return SafeResult.ok({"inserted_ids": []})
        raws = [_serialise_doc(d) for d in documents]
        res: InsertManyResult = await self.db[collection].insert_many(raws)
        if cache:
            for raw, _id in zip(raws, res.inserted_ids):
                self._cput(collection, {"_id": _id}, _restore_dict({**raw, "_id": _id}))
        return SafeResult.ok(res)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        sanitized_query = _sanitize_query(query)
        doc = await self.db[collection].find_one(sanitized_query)
        if not doc:
            return SafeResult.ok(None)
        doc = _restore_dict(doc)
        if cache and "_id" in doc:
            self._cput(collection, {"_id": doc["_id"]}, doc)
        return SafeResult.ok(doc)

    async def find_documents(self, collection: str, query: JsonDict, *, limit: int = None, sort: list = None) -> SafeResult:
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
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        res: UpdateResult = await self.db[collection].update_one(sanitized_query, upd, upsert=upsert)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False) -> SafeResult:
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        sanitized_query = _sanitize_query(query)
        res: UpdateResult = await self.db[collection].update_many(sanitized_query, upd, upsert=upsert)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        sanitized_query = _sanitize_query(query)
        res: DeleteResult = await self.db[collection].delete_one(sanitized_query)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def delete_documents(self, collection: str, query: JsonDict = None) -> SafeResult:
        sanitized_query = _sanitize_query(query) if query else {}
        res: DeleteResult = await self.db[collection].delete_many(sanitized_query)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        sanitized_query = _sanitize_query(query)
        count = await self.db[collection].count_documents(sanitized_query)
        return SafeResult.ok({"count": count})

    async def list_collections(self) -> SafeResult:
        names = await self.db.list_collection_names()
        return SafeResult.ok(names)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        cur = self.db[collection].aggregate(pipeline)
        out = []
        async for d in cur:
            out.append(_restore_dict(d))
            if len(out) >= limit:
                break
        return SafeResult.ok(out)

    @staticmethod
    def _ckey(query: JsonDict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()

    def _cget(self, col: str, q: JsonDict) -> Optional[dict]:
        return self._cache[self._norm(col)].get(self._ckey(q))

    async def __aenter__(self) -> "ZMongo": return self

    async def __aexit__(self, *_): self._client.close(); logger.info("MongoDB connection closed")
