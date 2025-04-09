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

# Load environment variables
load_dotenv()
DEFAULT_QUERY_LIMIT = int(os.getenv("DEFAULT_QUERY_LIMIT", 100))
DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", 300))
MAX_POOL_SIZE = int(os.getenv("MAX_POOL_SIZE", 500))

logger = logging.getLogger("zmagnum")
logger.setLevel(logging.INFO)

@dataclass
class UpdateResponse:
    matched_count: int
    modified_count: int
    upserted_id: Optional[Any] = None

class TTLCache(dict):
    def __init__(self, ttl: int = DEFAULT_CACHE_TTL):
        super().__init__()
        self.ttl = ttl
        self.timestamps: Dict[Any, datetime] = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.timestamps[key] = datetime.utcnow()

    def __getitem__(self, key):
        if key in self.timestamps:
            if (datetime.utcnow() - self.timestamps[key]).total_seconds() > self.ttl:
                self.timestamps.pop(key, None)
                super().__delitem__(key)
                raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def clear(self):
        super().clear()
        self.timestamps.clear()

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

        self.cache = defaultdict(lambda: TTLCache()) if not self.disable_cache else {}
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
            normalized = ZMagnum._normalize_collection_name(collection)
            cache_key = ZMagnum._generate_cache_key(query)
            if not self.disable_cache and self.cache[normalized].get(cache_key):
                return self.cache[normalized][cache_key]

            document = await self.db[collection].find_one(query)
            if document:
                serialized = ZMagnum.serialize_document(document)
                if not self.disable_cache:
                    self.cache[normalized][cache_key] = serialized
                return serialized
        except pymongo_errors.PyMongoError as e:
            logger.error(f"MongoDB error in find_document: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in find_document: {e}")
        return None

    async def insert_documents(self, collection: str, documents: List[dict], batch_size: int = 1000) -> Dict:
        if not documents:
            return {"inserted_count": 0}
        normalized = ZMagnum._normalize_collection_name(collection)
        inserted = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = await self.db[collection].insert_many(batch, ordered=False)
                inserted += len(result.inserted_ids)
                if not self.disable_cache:
                    for doc, _id in zip(batch, result.inserted_ids):
                        doc['_id'] = _id
                        self.cache[normalized][ZMagnum._generate_cache_key({'_id': str(_id)})] = ZMagnum.serialize_document(doc)
            except pymongo_errors.BulkWriteError as e:
                logger.error(f"Mongo BulkWriteError: {e.details}")
                return {"error": e.details}
            except pymongo_errors.PyMongoError as e:
                logger.error(f"Mongo insert error: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
        return {"inserted_count": inserted}

    async def update_document(self, collection: str, query: dict, update_data: dict) -> UpdateResponse:
        try:
            result = await self.db[collection].update_one(query, update_data)
            if result.matched_count > 0:
                updated = await self.db[collection].find_one(query)
                if updated and not self.disable_cache:
                    normalized = ZMagnum._normalize_collection_name(collection)
                    self.cache[normalized][ZMagnum._generate_cache_key(query)] = ZMagnum.serialize_document(updated)
            return UpdateResponse(result.matched_count, result.modified_count, result.upserted_id)
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo update failed: {e}")
            return UpdateResponse(0, 0, None)
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return UpdateResponse(0, 0, None)

    async def recommend_indexes(self, collection: str, sample_size: int = 1000) -> Dict[str, Any]:
        try:
            field_counter = Counter()
            cursor = self.db[collection].find({}, projection=None).limit(sample_size)
            documents = await cursor.to_list(length=sample_size)
            for doc in documents:
                field_counter.update(doc.keys())
            recommendations = {
                field: count for field, count in field_counter.items()
                if count > sample_size * 0.6
            }
            logger.info(f"Index recommendation for '{collection}': {recommendations}")
            return recommendations
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo recommend_indexes failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"recommend_indexes failed: {e}")
            return {"error": str(e)}

    def create_indexes(self, collection: str, fields: List[str]) -> None:
        for field in fields:
            try:
                self.sync_db[collection].create_index(field)
                logger.info(f"Created index on '{field}' in '{collection}'")
            except pymongo_errors.PyMongoError as e:
                logger.error(f"Mongo error creating index on '{field}' in '{collection}': {e}")
            except Exception as e:
                logger.error(f"Failed to create index on '{field}' in '{collection}': {e}")

    async def analyze_embedding_schema(self, collection: str, sample_size: int = 100) -> Dict[str, Any]:
        try:
            cursor = self.db[collection].find({"embedding": {"$exists": True}}).limit(sample_size)
            embeddings = await cursor.to_list(length=sample_size)
            if not embeddings:
                return {"error": "No embeddings found."}
            lengths = [len(doc.get("embedding", [])) for doc in embeddings if isinstance(doc.get("embedding"), list)]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            return {"sampled": len(lengths), "avg_embedding_length": avg_len}
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo analyze_embedding_schema failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"analyze_embedding_schema failed: {e}")
            return {"error": str(e)}

    def is_sharded_cluster(self) -> bool:
        try:
            status = self.sync_client.admin.command("isMaster")
            return status.get("msg") == "isdbgrid"
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo Cluster check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Cluster check failed: {e}")
            return False

    def route_to_shard(self, key: str) -> str:
        return f"shard-{hash(key) % 3}"

    async def close(self):
        try:
            self.mongo_client.close()
            logger.info("MongoDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

    async def bulk_write(self, collection: str, operations: List[Any]) -> Dict[str, Any]:
        if not operations:
            return {
                "inserted_count": 0,
                "matched_count": 0,
                "modified_count": 0,
                "deleted_count": 0,
                "upserted_count": 0,
                "acknowledged": True,
            }

        try:
            normalized = ZMagnum._normalize_collection_name(collection)
            result = await self.db[normalized].bulk_write(operations, ordered=False)

            # Always include the full standard response
            response = {
                "inserted_count": getattr(result, "inserted_count", 0),
                "matched_count": getattr(result, "matched_count", 0),
                "modified_count": getattr(result, "modified_count", 0),
                "deleted_count": getattr(result, "deleted_count", 0),
                "upserted_count": getattr(result, "upserted_count", 0),
                "acknowledged": getattr(result, "acknowledged", True),
            }

            if not self.disable_cache:
                for op in operations:
                    try:
                        if isinstance(op, InsertOne):
                            doc = getattr(op, "_doc", None)
                            _id = doc.get("_id") if doc else None
                            if _id:
                                cache_key = ZMagnum._generate_cache_key({"_id": str(_id)})
                                self.cache[normalized][cache_key] = ZMagnum.serialize_document(doc)

                        elif isinstance(op, UpdateOne):
                            filter_ = getattr(op, "_filter", None)
                            if filter_:
                                updated_doc = await self.db[normalized].find_one(filter_)
                                if updated_doc:
                                    cache_key = ZMagnum._generate_cache_key(filter_)
                                    self.cache[normalized][cache_key] = ZMagnum.serialize_document(updated_doc)
                    except Exception as cache_exc:
                        logger.warning(f"[Cache] Skipped cache write for operation: {cache_exc}")

            return response

        except pymongo_errors.BulkWriteError as e:
            logger.error(f"Mongo BulkWriteError: {e.details}")
            return {"error": e.details}
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Mongo bulk_write error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in bulk_write: {e}")
            return {"error": str(e)}
