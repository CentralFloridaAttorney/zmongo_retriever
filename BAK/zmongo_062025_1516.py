# zmongo_toolbag/zmongo.py
"""
Blazing-fast, flat-metadata ZMongo (2024-06)
-------------------------------------------
*   Every public method returns **SafeResult** with ONE uniform schema.
*   Complete async API (sync fall-backs for CPU-bound writers).
*   In-process cache with automatic invalidation on write / delete.
*   No more “sometimes list, sometimes dict, sometimes int” surprises.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import (DeleteOne, InsertOne, MongoClient, ReplaceOne, UpdateOne)
from pymongo.errors import BulkWriteError, PyMongoError

load_dotenv()
DEFAULT_QUERY_LIMIT: int = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────── SafeResult ──
class SafeResult:
    """A thin helper that always wraps `{ ok, payload, error }`."""

    __slots__ = ("_raw",)

    def __init__(self, ok: bool, payload: Any = None, error: str | None = None) -> None:
        self._raw: Dict[str, Any] = {"ok": ok, "payload": payload, "error": error}

    # convenience helpers
    def model_dump(self) -> Dict[str, Any]:
        return self._raw

    def __getitem__(self, k):  # type: ignore
        return self._raw[k]

    def __iter__(self):
        return iter(self._raw)

    def __bool__(self) -> bool:  # truthiness → ok
        return self._raw["ok"]

    # pretty
    def __repr__(self) -> str:
        return f"SafeResult({self._raw})"


# ─────────────────────────────────────────────────── ZMongo ───────
def _stringify_id(obj: Any) -> Any:
    """Turn ObjectId into str (works recursively on dict / list)."""
    if isinstance(obj, dict):
        return {k: _stringify_id(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_id(v) for v in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj


class ZMongo:
    """High-level async helper around Motor with a small in-process cache."""

    # ----------------------------------------------------- boiler-plate
    def __init__(self) -> None:
        self.MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.MONGO_DB_NAME: str = os.getenv("MONGO_DATABASE_NAME", "documents")

        self.mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(
            self.MONGO_URI, maxPoolSize=200
        )
        self.db: AsyncIOMotorDatabase = self.mongo_client[self.MONGO_DB_NAME]

        # separate sync client for occasional thread-pool calls
        self.sync_client: MongoClient = MongoClient(self.MONGO_URI, maxPoolSize=200)
        self.sync_db = self.sync_client[self.MONGO_DB_NAME]

        # { collection_name → { cache_key → document } }
        self.cache: Dict[str, Dict[str, dict]] = defaultdict(dict)

    # ----------------------------------------------------- helpers
    @staticmethod
    def _normalize_collection_name(coll: str) -> str:
        return coll.strip().lower()

    @staticmethod
    def _cache_key(query: dict) -> str:
        return hashlib.sha256(
            json.dumps(query, sort_keys=True, default=str).encode()
        ).hexdigest()

    # ───────────────────────────── CRUD / META / ETC ──
    # NOTE:  every method returns SafeResult(ok, payload, error)

    # ---------- collection list
    async def list_collections(self) -> SafeResult:
        try:
            names = await self.db.list_collection_names()
            return SafeResult(True, names)
        except Exception as exc:
            logger.error("list_collections: %s", exc)
            return SafeResult(False, [], str(exc))

    # ---------- count
    async def count_documents(self, coll: str) -> SafeResult:
        try:
            n = await self.db[coll].estimated_document_count()
            return SafeResult(True, n)
        except Exception as exc:
            logger.error("count_documents: %s", exc)
            return SafeResult(False, 0, str(exc))

    # ---------- single insert
    async def insert_document(self, coll: str, doc: dict) -> SafeResult:
        try:
            result = await self.db[coll].insert_one(doc)
            doc["_id"] = result.inserted_id
            self._cache_store(coll, doc)
            return SafeResult(True, str(result.inserted_id))
        except Exception as exc:
            logger.error("insert_document: %s", exc)
            return SafeResult(False, None, str(exc))

    # ---------- batch insert (async / sync)
    async def insert_documents(
        self,
        coll: str,
        docs: List[dict],
        *,
        batch_size: int = 1000,
        use_sync: bool = False,
    ) -> SafeResult:
        if not docs:
            return SafeResult(True, [])

        if use_sync:
            loop = asyncio.get_running_loop()
            return SafeResult(
                True,
                await loop.run_in_executor(
                    None, self._insert_documents_sync, coll, docs, batch_size
                ),
            )

        try:
            ids: List[str] = []
            for i in range(0, len(docs), batch_size):
                sub = docs[i : i + batch_size]
                res = await self.db[coll].insert_many(sub, ordered=False)
                for doc, _id in zip(sub, res.inserted_ids):
                    doc["_id"] = _id
                    self._cache_store(coll, doc)
                ids.extend(str(_id) for _id in res.inserted_ids)
            return SafeResult(True, ids)
        except Exception as exc:
            logger.error("insert_documents: %s", exc)
            return SafeResult(False, [], str(exc))

    # helper for sync batch insert
    def _insert_documents_sync(self, coll: str, docs: List[dict], batch: int) -> List[str]:
        ids: List[str] = []
        for i in range(0, len(docs), batch):
            sub = docs[i : i + batch]
            res = self.sync_db[coll].insert_many(sub, ordered=False)
            ids.extend(str(_id) for _id in res.inserted_ids)
        return ids

    # ---------- find one
    async def find_document(self, coll: str, query: dict) -> SafeResult:
        key = self._cache_key(query)
        coll_n = self._normalize_collection_name(coll)
        if key in self.cache[coll_n]:
            return SafeResult(True, _stringify_id(self.cache[coll_n][key]))

        try:
            doc = await self.db[coll].find_one(query)
            if doc:
                self.cache[coll_n][key] = doc
            return SafeResult(True, _stringify_id(doc) if doc else None)
        except Exception as exc:
            logger.error("find_document: %s", exc)
            return SafeResult(False, None, str(exc))

    # ---------- find many
    async def find_documents(self, coll: str, query: dict, *, limit=DEFAULT_QUERY_LIMIT) -> SafeResult:
        try:
            cur = self.db[coll].find(query).limit(limit)
            items = _stringify_id(await cur.to_list(length=limit))
            return SafeResult(True, items)
        except Exception as exc:
            logger.error("find_documents: %s", exc)
            return SafeResult(False, [], str(exc))

    # ---------- delete one
    async def delete_document(self, coll: str, query: dict) -> SafeResult:
        try:
            self._cache_evict(coll, query)
            res = await self.db[coll].delete_one(query)
            return SafeResult(True, res.deleted_count)
        except Exception as exc:
            logger.error("delete_document: %s", exc)
            return SafeResult(False, 0, str(exc))

    # ---------- delete many
    async def delete_documents(self, coll: str, query: dict) -> SafeResult:
        try:
            self._cache_evict(coll, query, many=True)
            res = await self.db[coll].delete_many(query)
            return SafeResult(True, res.deleted_count)
        except Exception as exc:
            logger.error("delete_documents: %s", exc)
            return SafeResult(False, 0, str(exc))

    # ---------- delete all
    async def delete_all_documents(self, coll: str) -> SafeResult:
        try:
            self.cache[self._normalize_collection_name(coll)].clear()
            res = await self.db[coll].delete_many({})
            return SafeResult(True, res.deleted_count)
        except Exception as exc:
            logger.error("delete_all_documents: %s", exc)
            return SafeResult(False, 0, str(exc))

    # ---------- update one
    async def update_document(
        self,
        coll: str,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
        array_filters: Optional[List[dict]] = None,
    ) -> SafeResult:
        try:
            res = await self.db[coll].update_one(
                query, {"$set": update}, upsert=upsert, array_filters=array_filters
            )
            # refresh cache if the doc exists (matched or upserted)
            if res.matched_count or res.upserted_id:
                fresh = await self.db[coll].find_one(query)
                self._cache_store(coll, fresh)
            return SafeResult(
                True,
                {
                    "matched": res.matched_count,
                    "modified": res.modified_count,
                    "upserted_id": str(res.upserted_id) if res.upserted_id else None,
                },
            )
        except Exception as exc:
            logger.error("update_document: %s", exc)
            return SafeResult(False, None, str(exc))

    # ---------- get_field_names
    async def get_field_names(self, coll: str, sample_size: int = 10) -> SafeResult:
        try:
            cur = self.db[coll].find({}, projection={"_id": 0}).limit(sample_size)
            docs = await cur.to_list(length=sample_size)
            fields = sorted({k for d in docs for k in d})
            return SafeResult(True, fields)
        except Exception as exc:
            logger.error("get_field_names: %s", exc)
            return SafeResult(False, [], str(exc))

    # ---------- sample_documents
    async def sample_documents(self, coll: str, sample_size: int = 5) -> SafeResult:
        try:
            cur = self.db[coll].find({}).limit(sample_size)
            docs = _stringify_id(await cur.to_list(length=sample_size))
            return SafeResult(True, docs)
        except Exception as exc:
            logger.error("sample_documents: %s", exc)
            return SafeResult(False, [], str(exc))

    # ---------- text_search
    async def text_search(self, coll: str, text: str, *, limit: int = 10) -> SafeResult:
        try:
            cur = self.db[coll].find({"$text": {"$search": text}}).limit(limit)
            docs = _stringify_id(await cur.to_list(length=limit))
            return SafeResult(True, docs)
        except Exception as exc:
            # often “no text index” → return empty list
            logger.error("text_search: %s", exc)
            return SafeResult(False, [], str(exc))

    # ---------- get by object id
    async def get_document_by_id(self, coll: str, doc_id: Union[str, ObjectId]) -> SafeResult:
        try:
            if isinstance(doc_id, str):
                try:
                    doc_id = ObjectId(doc_id)
                except Exception:
                    return SafeResult(True, None)  # invalid id string

            # ensure we bypass possible stale cache when called explicitly
            self._cache_evict(coll, {"_id": str(doc_id)})

            doc = await self.db[coll].find_one({"_id": doc_id})
            return SafeResult(True, _stringify_id(doc) if doc else None)
        except Exception as exc:
            logger.error("get_document_by_id: %s", exc)
            return SafeResult(False, None, str(exc))

    # ---------- log_training_metrics
    def log_training_metrics(self, metrics: Dict[str, Any]) -> SafeResult:
        try:
            doc = {"ts": datetime.utcnow(), **metrics}
            self.sync_db["training_metrics"].insert_one(doc)
            return SafeResult(True, True)
        except Exception as exc:
            logger.error("log_training_metrics: %s", exc)
            return SafeResult(False, False, str(exc))

    # ---------- save_embedding
    async def save_embedding(
        self,
        coll: str,
        doc_id: ObjectId,
        vector: List[float],
        *,
        field: str = "embedding",
    ) -> SafeResult:
        try:
            await self.db[coll].update_one({"_id": doc_id}, {"$set": {field: vector}}, upsert=True)
            self._cache_evict(coll, {"_id": str(doc_id)})  # ensure fresh on next read
            return SafeResult(True, True)
        except Exception as exc:
            logger.error("save_embedding: %s", exc)
            return SafeResult(False, False, str(exc))

    # ---------- bulk_write
    async def bulk_write(
        self,
        coll: str,
        ops: List[Union[InsertOne, DeleteOne, UpdateOne, ReplaceOne]],
    ) -> SafeResult:
        if not ops:
            return SafeResult(True, {"acknowledged": True})

        try:
            res = await self.db[coll].bulk_write(ops)
            summary = {
                "inserted": getattr(res, "inserted_count", 0),
                "matched": getattr(res, "matched_count", 0),
                "modified": getattr(res, "modified_count", 0),
                "deleted": getattr(res, "deleted_count", 0),
                "upserted": getattr(res, "upserted_count", 0),
            }
            if summary["modified"] or summary["deleted"]:
                self.cache[self._normalize_collection_name(coll)].clear()
            return SafeResult(True, summary)
        except (BulkWriteError, PyMongoError) as exc:
            logger.error("bulk_write: %s", exc)
            return SafeResult(False, {}, str(exc))

    # ---------- cache util
    def _cache_store(self, coll: str, doc: Optional[dict]) -> None:
        if not doc:
            return
        coll_n = self._normalize_collection_name(coll)
        key = self._cache_key({"_id": str(doc["_id"])})
        self.cache[coll_n][key] = doc

    def _cache_evict(self, coll: str, query: dict, *, many: bool = False) -> None:
        coll_n = self._normalize_collection_name(coll)
        if many:
            doomed = [
                k
                for k, v in self.cache[coll_n].items()
                if all(item in v.items() for item in query.items())
            ]
            for k in doomed:
                self.cache[coll_n].pop(k, None)
        else:
            key = self._cache_key(query)
            self.cache[coll_n].pop(key, None)

    # ---------- admin
    async def clear_cache(self) -> SafeResult:
        self.cache.clear()
        return SafeResult(True, True)

    async def close(self) -> SafeResult:
        self.mongo_client.close()
        return SafeResult(True, True)
