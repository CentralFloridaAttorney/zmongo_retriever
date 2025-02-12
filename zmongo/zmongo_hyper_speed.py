# zmongo_hyper_speed.py

import asyncio
import logging
import functools
import os
from datetime import datetime
from typing import Optional, List, Any, Dict

import json
import hashlib

from bson import json_util
from bson.objectid import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from pymongo.results import InsertOneResult, DeleteResult, BulkWriteResult

# Load environment variables from .env file
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

# Removed CACHE_EXPIRATION_SECONDS as Redis cache is being removed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ZMongoHyperSpeed:
    def __init__(self):
        """
        Initialize the ZMongoHyperSpeed using constants from environment variables.
        Incorporates only MongoDB client for optimized performance.
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

        # Removed Redis client initialization

    # Removed the `initialize` method as Redis is no longer used

    def _normalize_collection_name(self, collection_name: str) -> str:
        return collection_name.strip().lower()

    @functools.lru_cache(maxsize=10000)
    def _generate_cache_key(self, query_string: str) -> str:
        """
        Generate a cache key based on the query string using a hash function.
        This method can remain for MongoDB-based caching or other purposes.
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
        Removed Redis caching for this method.
        """
        try:
            # Fetch from MongoDB directly
            coll = self.db[collection]
            document = await coll.find_one({'_id': document_id}, {embedding_field: 1})
            if document and embedding_field in document:
                embedding_value = document.get(embedding_field)
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
        Find a single document in the specified collection based on the query.
        Removed Redis caching to fetch directly from MongoDB.
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(query, sort_keys=True, default=str)
        # Removed cache_key generation and Redis interactions

        try:
            # Fetch directly from MongoDB
            coll = self.db[collection]
            document = await coll.find_one(filter=query)
            if document:
                serialized_document = self.serialize_document(document)
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
        Removed Redis caching for this method.
        """
        normalized_collection = self._normalize_collection_name(collection)
        # Removed cache_key generation and Redis interactions

        try:
            # Fetch directly from MongoDB
            coll = self.db[collection]
            cursor = coll.find(filter=query, projection=projection)

            if sort:
                cursor = cursor.sort(sort)

            if skip:
                cursor = cursor.skip(skip)
            cursor = cursor.limit(limit)
            documents = await cursor.to_list(length=limit)

            # Serialize documents for consistency
            serialized_documents = [self.serialize_document(doc) for doc in documents]

            return serialized_documents
        except PyMongoError as e:
            logger.error(f"MongoDB error in find_documents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in find_documents: {e}")
            return []

    async def insert_document(self, collection: str, document: dict) -> Optional[str]:
        """
        Insert a document into the specified MongoDB collection.
        Removed Redis caching and cache key management.
        Returns the conversation_id as a string if successful, else None.
        """
        coll = self.db[collection]
        try:
            result = await coll.insert_one(document=document)
            document["_id"] = result.inserted_id

            # Exclude 'performance_tests' from any additional processing if needed
            normalized_collection = self._normalize_collection_name(collection)

            # Removed Redis caching logic

            # Return the conversation_id
            return document.get("conversation_id")
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
        Removed Redis caching for this method.
        """
        coll = self.db[collection]
        try:
            await coll.update_one(
                {'_id': document_id},
                {'$set': {embedding_field: embedding}},
                upsert=True
            )
            logger.debug(f"Embedding saved in MongoDB for document '{document_id}' in collection '{collection}'.")
            # Removed Redis caching logic
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
        matching 'query' in 'collection'.
        Removed Redis caching logic.
        Returns True if a doc was modified or upserted.
        """
        try:
            # Apply the update in MongoDB
            result = await self.db[collection].update_one(
                filter=query,
                update=update_data,
                upsert=upsert
            )

            # Determine if the operation was successful
            # This captures both modifications and upsert insertions
            success = (result.modified_count > 0) or (result.upserted_id is not None)

            if success:
                # Optionally, you can fetch and return the updated document
                document = await self.db[collection].find_one(filter=query)
                if document:
                    logger.debug(f"Document updated successfully for query '{query}' in collection '{collection}'.")
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
        Delete a document from the specified MongoDB collection.
        Removed Redis cache invalidation.
        """
        coll = self.db[collection]
        try:
            result = await coll.delete_one(query)
            if result.deleted_count > 0:
                normalized_collection = self._normalize_collection_name(collection)
                # Removed cache_key generation and Redis interactions
                logger.debug(f"Document deleted successfully for query '{query}' in collection '{collection}'.")
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
        Removed Redis caching for this method.
        """
        normalized_collection = self._normalize_collection_name(collection)
        # Removed cache_key generation and Redis interactions

        try:
            # Perform aggregation in MongoDB
            coll = self.db[collection]
            cursor = coll.aggregate(pipeline)
            documents = await cursor.to_list(length=limit)

            # Serialize documents for consistency
            serialized_documents = [self.serialize_document(doc) for doc in documents]

            return serialized_documents
        except PyMongoError as e:
            logger.error(f"MongoDB error in aggregate_documents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in aggregate_documents: {e}")
            return []

    async def bulk_write(self, collection: str, operations: list) -> Optional[BulkWriteResult]:
        """
        Perform bulk write operations (insert and update).
        Removed Redis caching logic.
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

                # **4. Update MongoDB Directly (No Redis Cache)**
                # Since caching is removed, no need to update Redis cache
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

    # Removed the `clear_cache` method as Redis is no longer used

    async def log_performance(self, operation: str, duration: float, num_operations: int):
        """
        Log performance results into a MongoDB collection for analysis.
        Removed Redis caching considerations.
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
        Close the MongoDB client connection.
        Removed Redis client closure.
        """
        try:
            self.mongo_client.close()
            logger.info("MongoDB client connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB client: {e}")
