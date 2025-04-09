# zmagnum.py
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from bson import json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, errors as pymongo_errors, UpdateOne, InsertOne
from pymongo.errors import BulkWriteError
from cachetools import TTLCache

# Load environment variables
load_dotenv()
DEFAULT_QUERY_LIMIT = int(os.getenv("DEFAULT_QUERY_LIMIT", 100))
DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", 300))
DEFAULT_CACHE_MAXSIZE = int(os.getenv("DEFAULT_CACHE_MAXSIZE", 1024))
MAX_POOL_SIZE = int(os.getenv("MAX_POOL_SIZE", 500))

logger = logging.getLogger("zmagnum")
logger.setLevel(logging.INFO)

@dataclass
class UpdateResponse:
    matched_count: int
    modified_count: int
    upserted_id: Optional[Any] = None

class ZMagnum:

    def __init__(self, disable_cache: bool = False, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.disable_cache = disable_cache
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.db_name = os.getenv("MONGO_DATABASE_NAME", "documents")

        try:
            self.loop = event_loop or asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.mongo_client = AsyncIOMotorClient(self.mongo_uri, maxPoolSize=MAX_POOL_SIZE, io_loop=self.loop)
        self.db = self.mongo_client[self.db_name]
        self.sync_client = MongoClient(self.mongo_uri, maxPoolSize=MAX_POOL_SIZE)
        self.sync_db = self.sync_client[self.db_name]

        self.cache = defaultdict(lambda: TTLCache(maxsize=DEFAULT_CACHE_MAXSIZE, ttl=DEFAULT_CACHE_TTL)) if not self.disable_cache else {}
        if self.disable_cache:
            logger.warning("Fast mode enabled: disabling cache and reducing logging noise.")
            logger.setLevel(logging.WARNING)

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def serialize_document(document: dict) -> dict:
        return json.loads(json_util.dumps(document)) if document else {}

    def _profile(self, operation_name: str, fn: Callable, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except pymongo_errors.PyMongoError as e:
            logger.error(f"[PROFILE][{operation_name}] Mongo error: {e}")
            raise
        except Exception as e:
            logger.error(f"[PROFILE][{operation_name}] error: {e}")
            raise
        elapsed = time.perf_counter() - start
        logger.info(f"[PROFILE] {operation_name} took {elapsed:.4f} seconds")
        return result

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        try:
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            if not self.disable_cache and self.cache[normalized].get(cache_key):
                return self.cache[normalized][cache_key]

            document = await self.db[collection].find_one(query)
            if document:
                serialized = self.serialize_document(document)
                if not self.disable_cache:
                    self.cache[normalized][cache_key] = serialized
                return serialized
        except pymongo_errors.PyMongoError as e:
            logger.error(f"MongoDB error in find_document: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in find_document: {e}")
        return None

    async def insert_documents(self, collection: str, documents: List[dict], batch_size: int = 1000) -> Dict[str, Any]:
        if not documents:
            return {"inserted_count": 0}
        normalized = self._normalize_collection_name(collection)
        inserted = 0
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = await self.db[collection].insert_many(batch, ordered=False)
                inserted += len(result.inserted_ids)
                if not self.disable_cache:
                    for doc, _id in zip(batch, result.inserted_ids):
                        doc['_id'] = _id
                        self.cache[normalized][self._generate_cache_key({'_id': str(_id)})] = self.serialize_document(doc)
            return {"inserted_count": inserted}
        except BulkWriteError as e:
            logger.error(f"Mongo BulkWriteError: {e.details}")
            return {"error": e.details}
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo insert error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return {"error": str(e)}

    async def delete_all_documents(self, collection: str) -> int:
        try:
            result = await self.db[collection].delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents from '{collection}'")
            return result.deleted_count
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo delete_all_documents failed: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error in delete_all_documents: {e}")
            return 0

    async def delete_document(self, collection: str, query: dict) -> Dict[str, Any]:
        try:
            result = await self.db[collection].delete_one(query)
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            if not self.disable_cache:
                self.cache[normalized].pop(cache_key, None)
            return {"deleted_count": result.deleted_count}
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo delete_document failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in delete_document: {e}")
            return {"error": str(e)}

    async def close(self):
        try:
            self.mongo_client.close()
            logger.info("MongoDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

    async def update_document(
            self,
            collection: str,
            query: dict,
            update_data: dict,
            upsert: bool = False,
            array_filters: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.db[collection].update_one(
                filter=query,
                update=update_data,
                upsert=upsert,
                array_filters=array_filters,
            )
            if result.matched_count > 0 or result.upserted_id:
                updated_doc = await self.db[collection].find_one(filter=query)
                if updated_doc and not self.disable_cache:
                    normalized = self._normalize_collection_name(collection)
                    cache_key = self._generate_cache_key(query)
                    self.cache[normalized][cache_key] = self.serialize_document(updated_doc)
            return {
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "upserted_id": result.upserted_id,
            }
        except pymongo_errors.PyMongoError as e:
            logger.error(f"MongoDB error in update_document: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in update_document: {e}")
            return {"error": str(e)}

    # Method Aliases
    find_one = find_document
    delete_one = delete_document
    clear_all_cache = flush_cache = reset_cache = lambda self: self.cache.clear() if not self.disable_cache else None
    insert_documents_alias = insert_documents
    insert_many = insert_bulk = add_documents = insert_documents_alias
    update_document_by_query = update_one = update = update_document