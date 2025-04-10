import os
import asyncio
import hashlib
import json
import logging
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
from bson import ObjectId, json_util
from pymongo import InsertOne, UpdateOne, DeleteOne, ReplaceOne, MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import BulkWriteError, PyMongoError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zdatabase")

DEFAULT_CACHE_ENABLED = True
DEFAULT_BATCH_SIZE = 1000

class ZDatabase:
    def __init__(self, uri: str = None, db_name: str = None, use_cache: bool = DEFAULT_CACHE_ENABLED):
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.db_name = db_name or os.getenv("MONGO_DATABASE_NAME", "documents")
        self.cache_enabled = use_cache

        self.async_client = AsyncIOMotorClient(self.uri)
        self.db = self.async_client[self.db_name]
        self.cache: Dict[str, Dict[str, dict]] = defaultdict(dict)

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _serialize_document(document: dict) -> dict:
        return json.loads(json_util.dumps(document)) if document else {}

    async def insert_documents(self, collection: str, documents: List[dict], batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
        if not documents:
            return {"inserted_count": 0}

        normalized = self._normalize_collection_name(collection)
        inserted_total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = await self.db[normalized].insert_many(batch, ordered=False)
                inserted_total += len(result.inserted_ids)
                if self.cache_enabled:
                    for doc, _id in zip(batch, result.inserted_ids):
                        doc['_id'] = _id
                        cache_key = self._generate_cache_key({'_id': str(_id)})
                        self.cache[normalized][cache_key] = self._serialize_document(doc)
            except Exception as e:
                logger.error(f"Error inserting batch into '{collection}': {e}")

        return {"inserted_count": inserted_total}

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key(query)

        if self.cache_enabled and cache_key in self.cache[normalized]:
            return self.cache[normalized][cache_key]

        doc = await self.db[normalized].find_one(query)
        if doc:
            serialized = self._serialize_document(doc)
            if self.cache_enabled:
                self.cache[normalized][cache_key] = serialized
            return serialized
        return None

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False) -> dict:
        normalized = self._normalize_collection_name(collection)
        try:
            result = await self.db[normalized].update_one(query, {"$set": update_data}, upsert=upsert)
            if result.matched_count > 0 or result.upserted_id:
                updated_doc = await self.db[normalized].find_one(query)
                if updated_doc and self.cache_enabled:
                    cache_key = self._generate_cache_key(query)
                    self.cache[normalized][cache_key] = self._serialize_document(updated_doc)
            return {
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "upserted_id": result.upserted_id,
            }
        except Exception as e:
            logger.error(f"Error updating document in '{collection}': {e}")
            return {"error": str(e)}

    async def delete_document(self, collection: str, query: dict) -> dict:
        normalized = self._normalize_collection_name(collection)
        try:
            result = await self.db[normalized].delete_one(query)
            if self.cache_enabled:
                cache_key = self._generate_cache_key(query)
                self.cache[normalized].pop(cache_key, None)
            return {"deleted_count": result.deleted_count}
        except Exception as e:
            logger.error(f"Error deleting document from '{collection}': {e}")
            return {"error": str(e)}

    async def close(self):
        self.async_client.close()
        logger.info("MongoDB connection closed.")
