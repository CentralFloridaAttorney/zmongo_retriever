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

class ZMongo:
    def __init__(self) -> None:
        uri     = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        db_name = os.getenv("MONGO_DATABASE_NAME", "documents")

        self._client = AsyncIOMotorClient(uri, maxPoolSize=200)
        self.db: AsyncIOMotorDatabase = self._client[db_name]
        self._cache: Dict[str, Dict[str, dict]] = defaultdict(dict)

    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _ckey(query: JsonDict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()

    def _cget(self, col: str, q: JsonDict) -> Optional[dict]:
        return self._cache[self._norm(col)].get(self._ckey(q))

    def _cput(self, col: str, q: JsonDict, doc: dict) -> None:
        self._cache[self._norm(col)][self._ckey(q)] = doc

    # ----------------------------- CRUD ops --------------------------------#
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        raw = _serialise_doc(document)
        res: InsertOneResult = await self.db[collection].insert_one(raw)
        restored = _restore_dict({**raw, "_id": res.inserted_id})
        if cache:
            self._cput(collection, {"_id": res.inserted_id}, restored)
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
        if cache and (cached := self._cget(collection, query)):
            return SafeResult.ok(cached)
        doc = await self.db[collection].find_one(query)
        if doc:
            doc = dict(doc)  # <--- force true dict type
            _restore_dict(doc)
            if cache:
                self._cput(collection, query, doc)
        return SafeResult.ok(doc)


    async def find_documents(self, collection: str, query: JsonDict, *, limit: int | None = None, sort: list | None = None) -> SafeResult:
        limit = limit or DEFAULT_QUERY_LIMIT
        cur   = self.db[collection].find(query)
        if sort:
            cur = cur.sort(sort)
        docs = []
        async for d in cur.limit(limit):
            docs.append(_restore_dict(d))
        return SafeResult.ok(docs)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        cur = self.db[collection].aggregate(pipeline)
        out = []
        async for d in cur:
            out.append(_restore_dict(d))
            if len(out) >= limit:
                break
        return SafeResult.ok(out)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False, array_filters: Optional[List[JsonDict]] = None) -> SafeResult:
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        res: UpdateResult = await self.db[collection].update_one(query, upd, upsert=upsert, array_filters=array_filters)
        if res.matched_count or res.upserted_id:
            target = {"_id": res.upserted_id} if res.upserted_id else query
            if doc := await self.db[collection].find_one(target):
                _restore_dict(doc)
                self._cput(collection, {"_id": doc["_id"]}, doc)
        return SafeResult.ok(res)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike, *, upsert: bool = False) -> SafeResult:
        upd = _serialise_doc(update_data)
        if not any(k.startswith("$") for k in upd):
            upd = {"$set": upd}
        res: UpdateResult = await self.db[collection].update_many(query, upd, upsert=upsert)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        res: DeleteResult = await self.db[collection].delete_one(query)
        self._cache[self._norm(collection)].pop(self._ckey(query), None)
        return SafeResult.ok(res)

    async def delete_documents(self, collection: str, query: JsonDict = None) -> SafeResult:
        """
        Efficiently delete many or all documents matching query.
        If query is None or empty, deletes **all documents** in the collection.
        Also clears collection cache.
        """
        query = query or {}
        res: DeleteResult = await self.db[collection].delete_many(query)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def bulk_write(self, collection: str, ops: List[Any]) -> SafeResult:
        res: BulkWriteResult = await self.db[collection].bulk_write(ops)
        self._cache[self._norm(collection)].clear()
        return SafeResult.ok(res)

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        return SafeResult.ok({"count": await self.db[collection].count_documents(query)})

    async def list_collections(self) -> SafeResult:
        return SafeResult.ok(await self.db.list_collection_names())

    async def __aenter__(self) -> "ZMongo": return self
    async def __aexit__(self, *_): self._client.close(); logger.info("MongoDB connection closed")
