import os
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from pymongo.errors import OperationFailure
from pymongo.operations import (DeleteMany, DeleteOne, InsertOne, UpdateMany,
                                UpdateOne)
from pymongo.results import (BulkWriteResult, DeleteResult, InsertManyResult,
                             InsertOneResult, UpdateResult)

load_dotenv(r"C:\Users\iriye\resources\.env_local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
DocLike = Union[dict, Any]
DocsLike = Union[List[DocLike], Any]
MongoOp = Union[InsertOne, DeleteOne, UpdateOne, DeleteMany, UpdateMany]

DEFAULT_QUERY_LIMIT = 100
DEFAULT_CACHE_TTL = 300


class SafeResult:
    """A predictable, serializable wrapper for all MongoDB operation results."""

    def __init__(self, data: Any = None, *, success: bool, error: Optional[str] = None,
                 original_exc: Optional[Exception] = None):
        self.success = success
        self.error = error
        self.data = self._convert_bson(data)
        self._original_exc = original_exc

    @staticmethod
    def _convert_bson(obj: Any) -> Any:
        if isinstance(obj, ObjectId): return str(obj)
        if isinstance(obj, dict): return {k: SafeResult._convert_bson(v) for k, v in obj.items()}
        if isinstance(obj, list): return [SafeResult._convert_bson(x) for x in obj]
        return obj

    @classmethod
    def ok(cls, data: Any = None) -> 'SafeResult':
        return cls(data=data, success=True)

    @classmethod
    def fail(cls, error: str, data: Any = None, exc: Optional[Exception] = None) -> 'SafeResult':
        return cls(data=data, success=False, error=error, original_exc=exc)

    def original(self) -> Union[Exception, None]:
        return self._original_exc

    def __repr__(self):
        return f"SafeResult(success={self.success}, error='{self.error}', data_preview='{str(self.data)[:100]}...')"


class ZMongo:
    def __init__(self, db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None,
                 cache_ttl: int = DEFAULT_CACHE_TTL):
        if db:
            self.db = db
        else:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
            mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
            client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = client[mongo_db]
        self._cache: Dict[str, Dict[str, Tuple[Any, float]]] = {}
        self._cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> 'ZMongo':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()  # Now correctly awaits the async close method

    # Caching methods
    async def _cget(self, coll: str, key: str) -> Optional[Any]:
        async with self._lock:
            coll_cache = self._cache.get(coll, {})
            entry = coll_cache.get(key)
            if not entry: return None
            value, expire_time = entry
            if time.time() > expire_time:
                del coll_cache[key]
                return None
            return value

    async def _cput(self, coll: str, key: str, value: Any):
        async with self._lock:
            if coll not in self._cache: self._cache[coll] = {}
            self._cache[coll][key] = (value, time.time() + self._cache_ttl)

    async def _cdelete(self, coll: str, query: Dict):
        async with self._lock:
            coll_cache = self._cache.get(coll, {})
            if "_id" in query and str(query["_id"]) in coll_cache:
                del coll_cache[str(query["_id"])]
            else:
                self._cache[coll] = {}

    def _doc_to_dict(self, doc: DocLike) -> Dict:
        if hasattr(doc, 'model_dump'): return doc.model_dump(by_alias=True)
        if hasattr(doc, 'dict'): return doc.dict(by_alias=True)
        return doc

    # --- FIX: Made the close method asynchronous ---
    async def close(self):
        """
        Closes the underlying MongoDB client connection.
        """
        if self.db is not None and self.db.client is not None:
            self.db.client.close()
            logger.info("MongoDB connection closed.")

    # CRUD methods
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        try:
            doc_dict = self._doc_to_dict(document)
            res: InsertOneResult = await self.db[collection].insert_one(doc_dict)
            if cache and res.inserted_id:
                await self._cput(collection, str(res.inserted_id), {**doc_dict, "_id": res.inserted_id})
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        if not documents: return SafeResult.ok({"inserted_ids": []})
        try:
            doc_list = [self._doc_to_dict(doc) for doc in documents]
            res: InsertManyResult = await self.db[collection].insert_many(doc_list)
            if cache and res.inserted_ids:
                for doc, doc_id in zip(doc_list, res.inserted_ids):
                    await self._cput(collection, str(doc_id), {**doc, "_id": doc_id})
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        try:
            if cache and "_id" in query:
                cached = await self._cget(collection, str(query["_id"]))
                if cached: return SafeResult.ok(cached)
            doc = await self.db[collection].find_one(query)
            if doc and cache and "_id" in doc:
                await self._cput(collection, str(doc["_id"]), doc)
            return SafeResult.ok(doc)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def find_documents(self, collection: str, query: JsonDict, *, limit: int = DEFAULT_QUERY_LIMIT,
                             sort: Optional[List[Tuple[str, int]]] = None) -> SafeResult:
        try:
            cursor = self.db[collection].find(query)
            if sort: cursor = cursor.sort(sort)
            docs = await cursor.to_list(length=limit)
            return SafeResult.ok(docs)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *,
                              upsert: bool = False) -> SafeResult:
        try:
            update_dict = self._doc_to_dict(update_data)
            if not any(k.startswith("$") for k in update_dict):
                update_dict = {"$set": update_dict}
            res: UpdateResult = await self.db[collection].update_one(query, update_dict, upsert=upsert)
            await self._cdelete(collection, query)
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike) -> SafeResult:
        try:
            update_dict = self._doc_to_dict(update_data)
            if not any(k.startswith("$") for k in update_dict):
                update_dict = {"$set": update_dict}
            res: UpdateResult = await self.db[collection].update_many(query, update_dict)
            await self._cdelete(collection, {})
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            res: DeleteResult = await self.db[collection].delete_one(query)
            await self._cdelete(collection, query)
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def delete_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            res: DeleteResult = await self.db[collection].delete_many(query)
            await self._cdelete(collection, {})
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def bulk_write(self, collection: str, ops: List[MongoOp]) -> SafeResult:
        try:
            res: BulkWriteResult = await self.db[collection].bulk_write(ops)
            await self._cdelete(collection, {})
            return SafeResult.ok(res)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            count = await self.db[collection].count_documents(query)
            return SafeResult.ok({"count": count})
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def list_collections(self) -> SafeResult:
        try:
            names = await self.db.list_collection_names()
            return SafeResult.ok(names)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def aggregate(self, collection: str, pipeline: List[JsonDict], *, limit: int = 1000) -> SafeResult:
        try:
            cursor = self.db[collection].aggregate(pipeline)
            return SafeResult.ok(await cursor.to_list(length=limit))
        except OperationFailure as e:
            raise e
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)