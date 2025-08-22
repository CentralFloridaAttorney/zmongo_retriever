# zmongo_toolbag/zmongo.py
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

load_dotenv(Path.home() / "resources" / ".env_local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]
DocsLike = Union[List[DocLike], Any]
MongoOp = Union[InsertOne, DeleteOne, UpdateOne, DeleteMany, UpdateMany]

DEFAULT_QUERY_LIMIT = 100
DEFAULT_CACHE_TTL = 300


class ZMongo:
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
        # Local import to avoid circulars if someone imports ZMongo during package init
        from zmongo_toolbag.buffered_ttl_cache import BufferedAsyncTTLCache
        self.cache = BufferedAsyncTTLCache(ttl=cache_ttl)

    async def __aenter__(self) -> "ZMongo":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---------- Result helpers ----------
    @staticmethod
    def ok(data: Any = None) -> "SafeResult":
        return SafeResult.ok(data)

    @staticmethod
    def fail(error: str, data: Any = None, exc: Optional[Exception] = None) -> "SafeResult":
        return SafeResult.fail(error, data=data, exc=exc)

    # ---------- ObjectId helpers ----------
    @staticmethod
    def _to_objectid(value: Any) -> Any:
        """Return value as ObjectId if it's a 24-hex string or already an ObjectId; otherwise return as-is."""
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
        """Recursively convert _id (and $in/$nin under _id) strings to ObjectId in a Mongo query."""
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
                    # handle {"_id": {"$in": [...]}}, {"$nin": [...]}, or even nested
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
        """Return a shallow copy with _id coerced to str for JSON-friendliness."""
        if not isinstance(doc, dict):
            return doc
        if "_id" in doc and isinstance(doc["_id"], ObjectId):
            newd = dict(doc)
            newd["_id"] = str(newd["_id"])
            return newd
        return doc

    # ---------- Cache helpers (via BufferedTTLCache) ----------
    @staticmethod
    def _ckey(coll: str, key: Union[str, Any]) -> str:
        return f"{coll}:{key}"

    async def _cget(self, coll: str, key: Union[str, Any]) -> Optional[Any]:
        return await self.cache.get(self._ckey(coll, key))

    async def _cput(self, coll: str, key: Union[str, Any], value: Any) -> None:
        # Cache under stringified key (stable across ObjectId/str usage)
        await self.cache.set(self._ckey(coll, str(key)), value, ttl=self._cache_ttl)

    async def _cdelete(self, coll: str, query: Dict) -> None:
        """
        Evict a single cached doc by _id if present, otherwise clear the whole cache to stay correct.
        Note: we normalize _id to ObjectId and then stringify it, matching _cput/_cget keys.
        """
        _id = query.get("_id")
        if _id is not None:
            oid = self._to_objectid(_id)
            await self.cache.delete(self._ckey(coll, str(oid)))
        else:
            # Conservative but correct: multi-updates/deletes likely touched unknown keys
            await self.cache.clear()

    # ---------- Utility ----------
    @staticmethod
    def _doc_to_dict(doc: DocLike) -> Dict:
        if hasattr(doc, "model_dump"):
            return doc.model_dump(by_alias=True)
        if hasattr(doc, "dict"):
            return doc.dict(by_alias=True)
        return doc

    @staticmethod
    def _parse_mongo_result(res: Any) -> Dict[str, Any]:
        """Converts raw PyMongo result objects into clean, serializable dicts."""
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
        """Closes the underlying MongoDB client connection."""
        if self.db is not None and self.db.client is not None:
            self.db.client.close()
            logger.info("MongoDB connection closed.")

    def close_connection(self):
        """Alias for the close() method for improved clarity."""
        self.close()

    # ---------- CRUD ----------
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        try:
            doc_dict = self._doc_to_dict(document)
            res = await self.db[collection].insert_one(doc_dict)
            if cache and res.inserted_id:
                await self._cput(collection, res.inserted_id, {**doc_dict, "_id": res.inserted_id})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
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
        try:
            norm_query = self._normalize_ids_in_query(query)
            # cache lookup only for direct _id lookups
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
        try:
            norm_query = self._normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            # Auto-wrap non-operator updates
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_one(norm_query, update_dict, upsert=upsert)
            await self._cdelete(collection, norm_query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike) -> SafeResult:
        try:
            norm_query = self._normalize_ids_in_query(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(isinstance(k, str) and k.startswith("$") for k in update_dict.keys()):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_many(norm_query, update_dict)
            await self._cdelete(collection, {})  # unknown set of keys: clear cache
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            norm_query = self._normalize_ids_in_query(query)
            res = await self.db[collection].delete_one(norm_query)
            await self._cdelete(collection, norm_query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            norm_query = self._normalize_ids_in_query(query)
            res = await self.db[collection].delete_many(norm_query)
            await self._cdelete(collection, {})  # unknown set of keys: clear cache
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def bulk_write(self, collection: str, ops: List[MongoOp]) -> SafeResult:
        try:
            # We can't reliably normalize every op here; callers should pass ObjectIds for _id in ops
            res = await self.db[collection].bulk_write(ops)
            await self._cdelete(collection, {})  # safest
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    # ---------- Misc ----------
    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            norm_query = self._normalize_ids_in_query(query)
            count = await self.db[collection].count_documents(norm_query)
            return self.ok({"count": count})
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def list_collections(self) -> SafeResult:
        try:
            names = await self.db.list_collection_names()
            return self.ok(names)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        try:
            # Normalize $match stages to handle string _id in pipelines
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
            raise e
        except Exception as e:
            return self.fail(str(e), exc=e)
