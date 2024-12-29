# zmongo_hyper_speed.py

import asyncio
import logging
import functools
import os
from datetime import datetime
from typing import Optional, List, Any

import json
import hashlib

import aioredis
from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from pymongo.results import InsertOneResult, DeleteResult, BulkWriteResult

# Load environment variables
load_dotenv()

# Retrieve and validate environment variables
DEFAULT_QUERY_LIMIT = os.getenv('DEFAULT_QUERY_LIMIT')
if DEFAULT_QUERY_LIMIT is not None:
    try:
        DEFAULT_QUERY_LIMIT = int(DEFAULT_QUERY_LIMIT)
    except ValueError:
        raise ValueError("DEFAULT_QUERY_LIMIT must be an integer.")
else:
    DEFAULT_QUERY_LIMIT = 100  # Set a default value if not provided

TEST_COLLECTION_NAME = os.getenv('TEST_COLLECTION_NAME')
if not TEST_COLLECTION_NAME:
    raise ValueError("TEST_COLLECTION_NAME must be set in the environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZMongoHyperSpeed:
    def __init__(self):
        """
        Initialize the ZMongoHyperSpeed using constants from environment variables.
        Incorporates MongoDB and Redis clients for optimized performance.
        """
        self.mongo_uri = os.getenv('MONGO_URI')
        if not self.mongo_uri:
            raise ValueError("MONGO_URI must be set in the environment variables.")

        self.db_name = os.getenv('MONGO_DATABASE_NAME')
        if not self.db_name or not isinstance(self.db_name, str):
            raise ValueError("MONGO_DATABASE_NAME must be set in the environment variables as a string.")

        # Initialize MongoDB client with optimized connection pooling
        self.mongo_client = AsyncIOMotorClient(
            self.mongo_uri,
            maxPoolSize=500,  # Increased pool size for higher concurrency
            minPoolSize=100,  # Minimum connections to keep
            serverSelectionTimeoutMS=5000,  # Faster server selection
            socketTimeoutMS=10000,  # Socket timeout
            connectTimeoutMS=10000,  # Connection timeout
        )
        self.db = self.mongo_client[self.db_name]

        # Initialize Redis client for external caching
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))

        self.redis = None  # Will be initialized in async context

        # Cache expiration settings
        self.CACHE_EXPIRATION_SECONDS = 300  # 5 minutes

    async def initialize(self):
        """
        Initialize Redis client asynchronously.
        """
        try:
            redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
            self.redis = aioredis.Redis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True,
            )
            # Optionally, ping Redis to ensure connection is established
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}, DB: {self.redis_db}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _normalize_collection_name(self, collection_name: str) -> str:
        return collection_name.strip().lower()

    @functools.lru_cache(maxsize=10000)
    def _generate_cache_key(self, query_string: str) -> str:
        """
        Generate a cache key based on the query string using a hash function.
        """
        return hashlib.sha256(query_string.encode('utf-8')).hexdigest()

    async def fetch_embedding(
            self,
            collection: str,
            document_id: ObjectId,
            embedding_field: str = 'embedding'
    ) -> Optional[List[float]]:
        """
        Fetch the embedding field from a document in the specified collection.
        Utilizes Redis cache for improved performance.
        """
        cache_key = f"embedding:{collection}:{str(document_id)}"
        try:
            # Attempt to fetch from Redis cache
            cached_embedding = await self.redis.get(cache_key)
            if cached_embedding:
                logger.debug(f"Redis cache hit for key '{cache_key}'")
                return json.loads(cached_embedding)
            else:
                logger.debug(f"Redis cache miss for key '{cache_key}'")

            # Fetch from MongoDB if not in cache
            coll = self.db[collection]
            document = await coll.find_one({'_id': document_id}, {embedding_field: 1})
            if document and embedding_field in document:
                embedding_value = document.get(embedding_field)
                # Store in Redis cache
                await self.redis.set(cache_key, json.dumps(embedding_value), ex=self.CACHE_EXPIRATION_SECONDS)
                logger.debug(f"Embedding cached in Redis for key '{cache_key}'")
                return embedding_value
            return None
        except PyMongoError as e:
            logger.error(f"MongoDB error in fetch_embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in fetch_embedding: {e}")
            return None

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """
        Retrieve a single document from the specified MongoDB collection.
        Utilizes Redis cache for improved performance.
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(query, sort_keys=True, default=str)
        cache_key = f"find:{normalized_collection}:{self._generate_cache_key(query_string)}"

        try:
            # Attempt to fetch from Redis cache
            cached_document = await self.redis.get(cache_key)
            if cached_document:
                logger.debug(f"Redis cache hit for key '{cache_key}'")
                return json.loads(cached_document)
            else:
                logger.debug(f"Redis cache miss for key '{cache_key}'")

            # Fetch from MongoDB if not in cache
            coll = self.db[collection]
            document = await coll.find_one(filter=query)
            if document:
                serialized_document = self.serialize_document(document)
                # Store in Redis cache
                await self.redis.set(cache_key, json.dumps(serialized_document), ex=self.CACHE_EXPIRATION_SECONDS)
                logger.debug(f"Document cached in Redis for key '{cache_key}'")
                return serialized_document
            return None
        except PyMongoError as e:
            logger.error(f"MongoDB error in find_document: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in find_document: {e}")
            return None

    async def find_documents(
            self,
            collection: str,
            query: dict,
            limit: int = DEFAULT_QUERY_LIMIT,
            projection: dict = None,
            sort: List[Any] = None,
            skip: int = 0,
    ) -> List[dict]:
        """
        Retrieve multiple documents from a MongoDB collection.
        Utilizes Redis caching for frequently accessed queries.
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(query, sort_keys=True, default=str)
        sort_string = json.dumps(sort) if sort else "None"
        cache_key = f"find_documents:{normalized_collection}:{self._generate_cache_key(query_string)}:{limit}:{skip}:{sort_string}"

        try:
            # Attempt to fetch from Redis cache
            cached_documents = await self.redis.get(cache_key)
            if cached_documents:
                logger.debug(f"Redis cache hit for key '{cache_key}'")
                return json.loads(cached_documents)
            else:
                logger.debug(f"Redis cache miss for key '{cache_key}'")

            # Fetch from MongoDB if not in cache
            coll = self.db[collection]
            cursor = coll.find(filter=query, projection=projection)

            if sort:
                cursor = cursor.sort(sort)

            if skip:
                cursor = cursor.skip(skip)
            cursor = cursor.limit(limit)
            documents = await cursor.to_list(length=limit)

            # Serialize documents for caching
            serialized_documents = [self.serialize_document(doc) for doc in documents]

            # Store in Redis cache
            await self.redis.set(cache_key, json.dumps(serialized_documents), ex=self.CACHE_EXPIRATION_SECONDS)
            logger.debug(f"Documents cached in Redis for key '{cache_key}'")
            return serialized_documents
        except PyMongoError as e:
            logger.error(f"MongoDB error in find_documents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in find_documents: {e}")
            return []

    async def insert_document(self, collection: str, document: dict) -> InsertOneResult:
        """
        Insert a document into the specified MongoDB collection and update the cache.
        """
        coll = self.db[collection]
        try:
            result = await coll.insert_one(document=document)
            document["_id"] = result.inserted_id

            # Normalize collection name
            normalized_collection = self._normalize_collection_name(collection)

            # Exclude 'performance_tests' from caching
            if normalized_collection != "performance_tests":
                # Create a unique cache key for this document
                query_string = json.dumps({"_id": str(result.inserted_id)}, sort_keys=True)
                cache_key = f"find:{normalized_collection}:{self._generate_cache_key(query_string)}"
                serialized_document = self.serialize_document(document)
                # Store in Redis cache
                await self.redis.set(cache_key, json.dumps(serialized_document), ex=self.CACHE_EXPIRATION_SECONDS)
                logger.debug(f"Document cached in Redis for key '{cache_key}'")
            else:
                logger.debug(f"Not caching document in collection: '{normalized_collection}'")

            return result
        except PyMongoError as e:
            logger.error(f"MongoDB error in insert_document: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in insert_document: {e}")
            raise

    async def save_embedding(
            self,
            collection: str,
            document_id: ObjectId,
            embedding: List[float],
            embedding_field: str = 'embedding'
    ):
        """
        Save an embedding to a document in the specified collection.
        Utilizes Redis cache for improved performance.
        """
        coll = self.db[collection]
        cache_key = f"embedding:{collection}:{str(document_id)}"
        try:
            await coll.update_one(
                {'_id': document_id},
                {'$set': {embedding_field: embedding}},
                upsert=True
            )
            logger.debug(f"Embedding saved in MongoDB for document '{document_id}' in collection '{collection}'.")

            # Update Redis cache
            await self.redis.set(cache_key, json.dumps(embedding), ex=self.CACHE_EXPIRATION_SECONDS)
            logger.debug(f"Embedding cached in Redis for key '{cache_key}'.")
        except PyMongoError as e:
            logger.error(f"MongoDB error in save_embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in save_embedding: {e}")
            raise

    async def update_document(
            self,
            collection: str,
            update_data: dict,
            query: dict,
            upsert: bool = False
    ) -> bool:
        """
        Perform a partial update ($set, $push, etc.) on a document
        matching 'query' in 'collection', and update the Redis cache
        for speed. Returns True if a doc was modified or upserted.
        """
        try:
            # Apply the update in MongoDB
            result = await self.db[collection].update_one(
                filter=query,
                update=update_data,
                upsert=upsert
            )

            # Determine if the operation was successful
            success = (result.modified_count > 0) or (result.upserted_id is not None)

            if success:
                # Normalize collection name and generate cache key
                normalized_coll = self._normalize_collection_name(collection)
                query_str = json.dumps(query, sort_keys=True, default=str)
                cache_key = f"find:{normalized_coll}:{self._generate_cache_key(query_str)}"

                # Fetch the updated document
                coll = self.db[collection]
                document = await coll.find_one(filter=query)
                if document:
                    serialized_document = self.serialize_document(document)
                    # Update Redis cache
                    await self.redis.set(cache_key, json.dumps(serialized_document), ex=self.CACHE_EXPIRATION_SECONDS)
                    logger.debug(f"Cache updated in Redis for key '{cache_key}'")
            else:
                logger.debug(f"No changes made for query '{query}' in collection '{collection}'")

            return success

        except PyMongoError as e:
            logger.error(f"MongoDB error in update_document: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in update_document: {e}")
            raise

    async def delete_document(self, collection: str, query: dict) -> Optional[DeleteResult]:
        """
        Delete a document from the specified MongoDB collection, updating the cache.
        """
        coll = self.db[collection]
        try:
            result = await coll.delete_one(query)
            if result.deleted_count > 0:
                normalized_collection = self._normalize_collection_name(collection)
                query_string = json.dumps(query, sort_keys=True, default=str)
                cache_key = f"find:{normalized_collection}:{self._generate_cache_key(query_string)}"
                await self.redis.delete(cache_key)
                logger.debug(f"Cache invalidated in Redis for key '{cache_key}'")
            return result
        except PyMongoError as e:
            logger.error(f"MongoDB error in delete_document: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in delete_document: {e}")
            raise

    @staticmethod
    def serialize_document(document: dict) -> dict:
        """
        Converts ObjectId fields in a document to strings for JSON serialization.
        """
        if document is None:
            return None
        return json.loads(json_util.dumps(document))

    async def aggregate_documents(
            self, collection: str, pipeline: list, limit: int = DEFAULT_QUERY_LIMIT
    ) -> List[dict]:
        """
        Perform an aggregation operation on the specified MongoDB collection.
        Utilizes Redis cache for frequently used aggregation pipelines.
        """
        normalized_collection = self._normalize_collection_name(collection)
        pipeline_string = json.dumps(pipeline, sort_keys=True, default=str)
        cache_key = f"aggregate:{normalized_collection}:{hashlib.sha256(pipeline_string.encode()).hexdigest()}:{limit}"

        try:
            # Attempt to fetch from Redis cache
            cached_aggregation = await self.redis.get(cache_key)
            if cached_aggregation:
                logger.debug(f"Redis cache hit for key '{cache_key}'")
                return json.loads(cached_aggregation)
            else:
                logger.debug(f"Redis cache miss for key '{cache_key}'")

            # Perform aggregation in MongoDB
            coll = self.db[collection]
            cursor = coll.aggregate(pipeline)
            documents = await cursor.to_list(length=limit)

            # Serialize documents for caching
            serialized_documents = [ZMongoHyperSpeed.serialize_document(doc) for doc in documents]

            # Store in Redis cache
            await self.redis.set(cache_key, json.dumps(serialized_documents), ex=self.CACHE_EXPIRATION_SECONDS)
            logger.debug(f"Aggregation results cached in Redis for key '{cache_key}'")
            return serialized_documents
        except PyMongoError as e:
            logger.error(f"MongoDB error in aggregate_documents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in aggregate_documents: {e}")
            return []

    async def bulk_write(self, collection: str, operations: list) -> Optional[BulkWriteResult]:
        """
        Perform bulk write operations (insert and update), updating the cache.
        Optimized for higher throughput and lower latency.
        """
        coll = self.db[collection]
        try:
            # **1. Segregate Operations**
            insert_ops = [op["document"] for op in operations if op.get("action") == "insert" and "document" in op]
            update_ops = [
                UpdateOne(
                    filter=op["filter"],
                    update=op["update"],
                    upsert=op.get("upsert", False)
                )
                for op in operations if op.get("action") == "update" and "filter" in op and "update" in op
            ]

            # **2. Validate Operations**
            for idx, op in enumerate(operations):
                if op.get("action") == "insert" and "document" not in op:
                    logger.error(f"Insert operation at index {idx} missing 'document': {op}")
                if op.get("action") == "update" and ("filter" not in op or "update" not in op):
                    logger.error(f"Update operation at index {idx} missing 'filter' or 'update' attribute: {op}")
                if op.get("action") not in ["insert", "update"]:
                    logger.warning(f"Unsupported operation type at index {idx}: {op}")

            # **3. Perform Bulk Write Operations**
            if insert_ops or update_ops:
                logger.info(f"Performing bulk write operations on collection '{collection}'.")
                bulk_operations = []
                if insert_ops:
                    bulk_operations.extend([InsertOne(doc) for doc in insert_ops])
                if update_ops:
                    bulk_operations.extend(update_ops)

                # Execute bulk write with ordered=False for better performance
                bulk_result = await coll.bulk_write(bulk_operations, ordered=False)

                logger.info(f"Bulk write completed: {bulk_result.bulk_api_result}")

                # **4. Update Redis Cache**
                # For insertions
                for doc in insert_ops:
                    inserted_id = doc.get("_id")
                    if not inserted_id:
                        continue  # Skip if no _id
                    normalized_collection = self._normalize_collection_name(collection)
                    query_string = json.dumps({"_id": str(inserted_id)}, sort_keys=True)
                    cache_key = f"find:{normalized_collection}:{self._generate_cache_key(query_string)}"
                    serialized_document = self.serialize_document(doc)
                    await self.redis.set(cache_key, json.dumps(serialized_document), ex=self.CACHE_EXPIRATION_SECONDS)
                    logger.debug(f"Document cached in Redis for key '{cache_key}'")

                # For updates
                for op in operations:
                    if op.get("action") != "update":
                        continue
                    query = op.get("filter")
                    if not query:
                        continue
                    normalized_coll = self._normalize_collection_name(collection)
                    query_str = json.dumps(query, sort_keys=True, default=str)
                    cache_key = f"find:{normalized_coll}:{self._generate_cache_key(query_str)}"
                    # Fetch the updated document
                    document = await self.db[collection].find_one(filter=query)
                    if document:
                        serialized_document = self.serialize_document(document)
                        await self.redis.set(cache_key, json.dumps(serialized_document), ex=self.CACHE_EXPIRATION_SECONDS)
                        logger.debug(f"Updated document cached in Redis for key '{cache_key}'")

            else:
                logger.warning("No valid insert or update operations found to perform.")

            return None  # Modify as needed to return relevant information

        except BulkWriteError as e:
            logger.error(f"Bulk write error in '{collection}': {e.details}")
            raise
        except PyMongoError as e:
            logger.error(f"MongoDB error in bulk_write: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in bulk_write: {e}")
            raise

    async def clear_cache(self):
        """
        Clear the entire Redis cache by flushing all keys.
        Use with caution in production environments.
        """
        try:
            await self.redis.flushdb()
            logger.info("Redis cache has been flushed.")
        except Exception as e:
            logger.error(f"Failed to flush Redis cache: {e}")
            raise

    async def log_performance(self, operation: str, duration: float, num_operations: int):
        """
        Log performance results into a MongoDB collection for analysis.
        Excludes 'performance_tests' from being cached.
        """
        performance_data = {
            "operation": operation,
            "num_operations": num_operations,
            "duration_seconds": duration,
            "avg_duration_per_operation": duration / num_operations if num_operations else 0,
            "timestamp": datetime.utcnow(),
        }
        await self.insert_document("performance_tests", performance_data)
        logger.info(f"Performance log inserted: {performance_data}")

    async def close(self):
        """
        Close the MongoDB and Redis client connections.
        """
        try:
            self.mongo_client.close()
            logger.info("MongoDB client connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB client: {e}")

        try:
            await self.redis.close()
            logger.info("Redis client connection closed.")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
