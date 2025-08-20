import os
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import motor.motor_asyncio
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from pymongo.errors import OperationFailure
from pymongo.operations import (DeleteMany, DeleteOne, InsertOne, UpdateMany,
                                UpdateOne)
from pymongo.results import (BulkWriteResult, DeleteResult, InsertManyResult,
                             InsertOneResult, UpdateResult)

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
    def __init__(self, db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None,
                 cache_ttl: int = DEFAULT_CACHE_TTL):
        if db is not None:
            self.db = db
        else:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
            mongo_db = os.getenv("MONGO_DATABASE_NAME", "test")
            client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = client[mongo_db]
        self.cache: Dict[str, Dict[str, Tuple[Any, float]]] = {}
        self.cache_ttl = cache_ttl
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> 'ZMongo':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- ADDED METHODS ---
    @staticmethod
    def ok(data: Any = None) -> 'SafeResult':
        """Convenience wrapper for SafeResult.ok()."""
        return SafeResult.ok(data)

    @staticmethod
    def fail(error: str, data: Any = None, exc: Optional[Exception] = None) -> 'SafeResult':
        """Convenience wrapper for SafeResult.fail()."""
        return SafeResult.fail(error, data=data, exc=exc)

    # Caching methods
    async def _cget(self, coll: str, key: str) -> Optional[Any]:
        async with self.lock:
            coll_cache = self.cache.get(coll, {})
            entry = coll_cache.get(key)
            if not entry: return None
            value, expire_time = entry
            if time.time() > expire_time:
                del coll_cache[key]
                return None
            return value

    async def _cput(self, coll: str, key: str, value: Any):
        async with self.lock:
            if coll not in self.cache: self.cache[coll] = {}
            self.cache[coll][key] = (value, time.time() + self.cache_ttl)

    async def _cdelete(self, coll: str, query: Dict):
        async with self.lock:
            coll_cache = self.cache.get(coll, {})
            # Use a copy to avoid modifying the original query dict
            query_copy = query.copy()
            self._handle_objectid(query_copy)
            if "_id" in query_copy and str(query_copy["_id"]) in coll_cache:
                del coll_cache[str(query_copy["_id"])]
            else:
                self.cache[coll] = {}

    @staticmethod
    def _doc_to_dict(doc: DocLike) -> Dict:
        if hasattr(doc, 'model_dump'): return doc.model_dump(by_alias=True)
        if hasattr(doc, 'dict'): return doc.dict(by_alias=True)
        return doc

    @staticmethod
    def _handle_objectid(query: Dict[str, Any]):
        """
        Intelligently converts a string '_id' to an ObjectId in a query dict.
        Modifies the dictionary in-place.
        """
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except InvalidId:
                # If the string is not a valid ObjectId, proceed without conversion.
                # The query will likely fail or return no results, which is expected.
                logger.warning(f"Value '{query['_id']}' is not a valid ObjectId.")
                pass

    @staticmethod
    def _parse_mongo_result(res: Any) -> Dict[str, Any]:
        """Converts raw Pymongo result objects into clean, serializable dicts."""
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

    # CRUD methods
    async def insert_document(self, collection: str, document: DocLike, *, cache: bool = True) -> SafeResult:
        try:
            doc_dict = self._doc_to_dict(document)
            res = await self.db[collection].insert_one(doc_dict)
            if cache and res.inserted_id:
                await self._cput(collection, str(res.inserted_id), {**doc_dict, "_id": res.inserted_id})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def insert_documents(self, collection: str, documents: DocsLike, *, cache: bool = True) -> SafeResult:
        if not documents: return self.ok({"inserted_ids": [], "acknowledged": True})
        try:
            doc_list = [self._doc_to_dict(doc) for doc in documents]
            res = await self.db[collection].insert_many(doc_list)
            if cache and res.inserted_ids:
                for doc, doc_id in zip(doc_list, res.inserted_ids):
                    await self._cput(collection, str(doc_id), {**doc, "_id": doc_id})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_document(self, collection: str, query: JsonDict, *, cache: bool = True) -> SafeResult:
        try:
            self._handle_objectid(query)
            if cache and "_id" in query:
                cached = await self._cget(collection, str(query["_id"]))
                if cached: return self.ok(cached)
            doc = await self.db[collection].find_one(query)
            if doc and cache and "_id" in doc:
                await self._cput(collection, str(doc["_id"]), doc)
            return self.ok(doc)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def find_documents(self, collection: str, query: JsonDict, *, limit: int = DEFAULT_QUERY_LIMIT,
                             sort: Optional[List[Tuple[str, int]]] = None) -> SafeResult:
        try:
            self._handle_objectid(query)
            cursor = self.db[collection].find(query)
            if sort: cursor = cursor.sort(sort)
            docs = await cursor.to_list(length=limit)
            return self.ok(docs)
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_document(self, collection: str, query: JsonDict, update_data: DocLike, *,
                              upsert: bool = False) -> SafeResult:
        try:
            self._handle_objectid(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(k.startswith("$") for k in update_dict):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_one(query, update_dict, upsert=upsert)
            await self._cdelete(collection, query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def update_documents(self, collection: str, query: JsonDict, update_data: DocLike) -> SafeResult:
        try:
            self._handle_objectid(query)
            update_dict = self._doc_to_dict(update_data)
            if not any(k.startswith("$") for k in update_dict):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_many(query, update_dict)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            self._handle_objectid(query)
            res = await self.db[collection].delete_one(query)
            await self._cdelete(collection, query)
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def delete_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            self._handle_objectid(query)
            res = await self.db[collection].delete_many(query)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def bulk_write(self, collection: str, ops: List[MongoOp]) -> SafeResult:
        try:
            # Note: ObjectId handling in bulk_write ops needs to be done by the caller
            # as it requires iterating through a list of mixed operation types.
            res = await self.db[collection].bulk_write(ops)
            await self._cdelete(collection, {})
            return self.ok(self._parse_mongo_result(res))
        except Exception as e:
            return self.fail(str(e), exc=e)

    async def count_documents(self, collection: str, query: JsonDict) -> SafeResult:
        try:
            self._handle_objectid(query)
            count = await self.db[collection].count_documents(query)
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
            cursor = self.db[collection].aggregate(pipeline)
            docs = await cursor.to_list(length=limit)
            return self.ok(docs)
        except OperationFailure as e:
            raise e
        except Exception as e:
            return self.fail(str(e), exc=e)
