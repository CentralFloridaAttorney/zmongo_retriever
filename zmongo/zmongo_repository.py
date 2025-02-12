# zmongo_repository.py

import asyncio
import logging
import functools
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Any

import json
import hashlib
from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import BulkWriteError
from pymongo.results import (
    InsertOneResult,
    UpdateResult,
    DeleteResult,
    BulkWriteResult,
)

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


class ZMongoRepository:
    def __init__(self):
        """
        Initialize the ZMongoRepository using constants from environment variables.
        Includes an in-memory cache for improved performance.
        """
        self.mongo_uri = os.getenv('MONGO_URI')
        if not self.mongo_uri:
            raise ValueError("MONGO_URI must be set in the environment variables.")

        self.db_name = os.getenv('MONGO_DATABASE_NAME')
        if not self.db_name or not isinstance(self.db_name, str):
            raise ValueError("MONGO_DATABASE_NAME must be set in the environment variables as a string.")

        self.mongo_client = AsyncIOMotorClient(
            self.mongo_uri, maxPoolSize=200  # Adjusted pool size as needed
        )
        self.db = self.mongo_client[self.db_name]
        self.cache = defaultdict(dict)  # Cache structure: {collection: {cache_key: document}}

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
        """
        coll = self.db[collection]
        document = await coll.find_one({'_id': document_id}, {embedding_field: 1})
        if document:
            embedding_value = document.get(embedding_field)
            return embedding_value
        return None

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """
        Retrieve a single document from the specified MongoDB collection.
        Uses cache if available, otherwise fetches from MongoDB.
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(query, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)

        if cache_key in self.cache[normalized_collection]:
            logger.debug(f"Cache hit for collection '{normalized_collection}' with key '{cache_key}'")
            return self.cache[normalized_collection][cache_key]
        else:
            logger.debug(f"Cache miss for collection '{normalized_collection}' with key '{cache_key}'")

        coll = self.db[collection]
        document = await coll.find_one(filter=query)
        if document:
            serialized_document = self.serialize_document(document)
            self.cache[normalized_collection][cache_key] = serialized_document
            logger.debug(f"Document cached for collection '{normalized_collection}' with key '{cache_key}'")
            return serialized_document
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
        """
        coll = self.db[collection]
        cursor = coll.find(filter=query, projection=projection)

        if sort:
            cursor = cursor.sort(sort)

        cursor = cursor.skip(skip).limit(limit)
        documents = await cursor.to_list(length=limit)
        return documents

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
                logger.debug(f"Caching document in collection: '{normalized_collection}'")
                query_string = json.dumps({"_id": str(result.inserted_id)}, sort_keys=True)
                cache_key = self._generate_cache_key(query_string)
                self.cache[normalized_collection][cache_key] = self.serialize_document(document)
            else:
                logger.debug(f"Not caching document in collection: '{normalized_collection}'")

            return result
        except Exception as e:
            logger.error(f"Error inserting document into '{collection}': {e}")
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
        """
        coll = self.db[collection]
        try:
            await coll.update_one(
                {'_id': document_id},
                {'$set': {embedding_field: embedding}},
                upsert=True
            )
            logger.debug(f"Embedding saved for document '{document_id}' in collection '{collection}'.")
        except Exception as e:
            logger.error(f"Error saving embedding for document '{document_id}': {e}")
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
        matching 'query' in 'collection', and update the in-memory cache
        for speed. Returns True if a doc was modified or upserted.
        """
        try:
            # 1) Apply the update in MongoDB
            result = await self.db[collection].update_one(
                filter=query,
                update=update_data,
                upsert=upsert
            )

            # 2) We consider the operation 'successful' if something was modified or upserted
            success = (result.modified_count > 0) or (result.upserted_id is not None)

            if success:
                # 3) Normalize collection name and generate cache key
                normalized_coll = self._normalize_collection_name(collection)
                query_str = json.dumps(query, sort_keys=True, default=str)
                cache_key = self._generate_cache_key(query_str)

                if cache_key in self.cache[normalized_coll]:
                    # 4) Apply the update operators to the cached document
                    self._apply_update_operator(
                        self.cache[normalized_coll][cache_key],
                        update_data
                    )
                    logger.debug(f"Cache updated for collection '{normalized_coll}' with key '{cache_key}'")

                    # Log the updated part of the document for verification
                    updated_part = self.cache[normalized_coll][cache_key]
                    logger.debug(f"Updated document: {updated_part}")
                else:
                    logger.debug(f"No cache entry found for collection '{normalized_coll}' with key '{cache_key}'")

            return success

        except Exception as e:
            error_detail = getattr(e, 'details', str(e))
            logger.error(
                f"Error updating document in '{collection}': {e}, full error: {error_detail}"
            )
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
                cache_key = self._generate_cache_key(query_string)
                self.cache[normalized_collection].pop(cache_key, None)
                logger.debug(f"Cache updated: Document with query '{query}' removed from cache.")
            return result
        except Exception as e:
            logger.error(f"Error deleting document from '{collection}': {e}")
            raise

    @staticmethod
    def serialize_document(document: dict) -> dict:
        """
        Converts ObjectId fields in a document to strings for JSON serialization.
        """
        if document is None:
            return None
        return json.loads(json_util.dumps(document))

    @staticmethod
    def _apply_update_operator(document: dict, update_data: dict):
        """
        Apply MongoDB update operators to the cached document.
        Supports $set, $unset, $inc, $push, and $addToSet with nested keys.
        """
        for operator, fields in update_data.items():
            if operator == "$set":
                for key_path, value in fields.items():
                    ZMongoRepository._set_nested_value(document, key_path, value)
                    logger.debug(f"$set applied on '{key_path}' with value '{value}'")
            elif operator == "$unset":
                for key_path in fields.keys():
                    ZMongoRepository._unset_nested_key(document, key_path)
                    logger.debug(f"$unset applied on '{key_path}'")
            elif operator == "$inc":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, value)
                        logger.debug(f"$inc applied on '{key_path}' with value '{value}' (initialized)")
                    else:
                        ZMongoRepository._set_nested_value(document, key_path, current + value)
                        logger.debug(f"$inc applied on '{key_path}' with value '{value}' (incremented)")
            elif operator == "$push":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logger.debug(f"$push applied on '{key_path}' with value '{value}' (initialized list)")
                    elif isinstance(current, list):
                        current.append(value)
                        logger.debug(f"$push applied on '{key_path}' with value '{value}' (appended)")
                    else:
                        # Handle error: trying to push to a non-list field
                        logger.warning(f"Cannot push to non-list field: '{key_path}'")
            elif operator == "$addToSet":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logger.debug(f"$addToSet applied on '{key_path}' with value '{value}' (initialized list)")
                    elif isinstance(current, list) and value not in current:
                        current.append(value)
                        logger.debug(f"$addToSet applied on '{key_path}' with value '{value}' (added to set)")
            # Implement other operators as needed

    @staticmethod
    def _set_nested_value(document: dict, key_path: str, value: Any):
        """
        Set a value in a nested dictionary or list based on the dot-separated key path.
        """
        keys = key_path.split('.')
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
                if not isinstance(document, list):
                    logger.warning(f"Expected list at key: '{key}'")
                    return
                while len(document) <= key:
                    document.append({})
                document = document[key]
            else:
                if key not in document or not isinstance(document[key], dict):
                    document[key] = {}
                document = document[key]
        last_key = keys[-1]
        if last_key.isdigit():
            last_key = int(last_key)
            if not isinstance(document, list):
                logger.warning(f"Expected list at key: '{last_key}'")
                return
            while len(document) <= last_key:
                document.append({})
            document[last_key] = value
        else:
            document[last_key] = value

    @staticmethod
    def _unset_nested_key(document: dict, key_path: str):
        """
        Unset a value in a nested dictionary or list based on the dot-separated key path.
        """
        keys = key_path.split('.')
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
                if not isinstance(document, list) or key >= len(document):
                    return
                document = document[key]
            else:
                if key not in document:
                    return
                document = document[key]
        last_key = keys[-1]
        if last_key.isdigit():
            last_key = int(last_key)
            if isinstance(document, list) and last_key < len(document):
                document.pop(last_key)
                logger.debug(f"Unset list element at index '{last_key}'")
        else:
            document.pop(last_key, None)
            logger.debug(f"Unset field '{last_key}'")

    @staticmethod
    def _get_nested_value(document: dict, key_path: str) -> Optional[Any]:
        """
        Retrieve a value from a nested dictionary or list based on the dot-separated key path.
        """
        keys = key_path.split('.')
        for key in keys:
            if key.isdigit():
                key = int(key)
                if not isinstance(document, list) or key >= len(document):
                    return None
                document = document[key]
            else:
                if key not in document:
                    return None
                document = document[key]
        return document

    async def aggregate_documents(
            self, collection: str, pipeline: list, limit: int = DEFAULT_QUERY_LIMIT
    ) -> List[dict]:
        """
        Perform an aggregation operation on the specified MongoDB collection.
        """
        coll = self.db[collection]
        try:
            cursor = coll.aggregate(pipeline)
            documents = await cursor.to_list(length=limit)
            return documents
        except Exception as e:
            logger.error(f"Error during aggregation on '{collection}': {e}")
            raise

    async def bulk_write(self, collection: str, operations: list) -> Optional[BulkWriteResult]:
        """
        Perform bulk write operations (insert and update), updating the cache.

        Each operation in the list should be a dictionary with the following structure:

        - Insert Operation:
          {
              "action": "insert",
              "document": { ... }  # Document to insert
          }

        - Update Operation:
          {
              "action": "update",
              "filter": { ... },    # Filter to select documents
              "update": { ... },    # Update operations (e.g., {"$set": {"age": 30}})
              "upsert": True        # Optional: Perform upsert
          }
        """
        coll = self.db[collection]
        try:
            # **1. Segregate Operations**
            insert_docs = [op["document"] for op in operations if op.get("action") == "insert" and "document" in op]
            update_ops = [op for op in operations if op.get("action") == "update" and "filter" in op and "update" in op]

            # **2. Log Any Malformed Operations**
            for idx, op in enumerate(operations):
                if op.get("action") == "insert" and "document" not in op:
                    logger.error(f"Insert operation at index {idx} missing 'document': {op}")
                if op.get("action") == "update" and ("filter" not in op or "update" not in op):
                    logger.error(f"Update operation at index {idx} missing 'filter' or 'update' attribute: {op}")
                if op.get("action") not in ["insert", "update"]:
                    logger.warning(f"Unsupported operation type at index {idx}: {op}")

            # **3. Perform Insert Operations**
            if insert_docs:
                logger.info(f"Performing {len(insert_docs)} insert operations on collection '{collection}'.")
                insert_result = await coll.insert_many(insert_docs)

                # Update the cache with inserted documents
                insert_tasks = [
                    self._update_cache_with_insert(collection, doc)
                    for doc in insert_docs
                ]
                await asyncio.gather(*insert_tasks)
            else:
                logger.warning("No valid insert operations found to perform.")

            # **4. Perform Update Operations**
            if update_ops:
                logger.info(f"Performing {len(update_ops)} update operations on collection '{collection}'.")
                update_results = []
                for op in update_ops:
                    result = await coll.update_one(
                        filter=op["filter"],
                        update=op["update"],
                        upsert=op.get("upsert", False)
                    )
                    update_results.append(result)

                    # Update the cache for each update operation
                    await self._update_cache_with_update(collection, op["filter"], op["update"])
            else:
                logger.warning("No valid update operations found to perform.")

            return None  # Return value can be adjusted as needed

        except BulkWriteError as e:
            logger.error(f"Bulk write error in '{collection}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during bulk write in '{collection}': {e}")
            raise

    async def _update_cache_with_insert(self, collection: str, doc: dict):
        """
        Helper method to update cache after an insert operation.
        """
        # Ensure the document has an '_id'
        if "_id" not in doc:
            logger.error(f"Inserted document missing '_id': {doc}")
            return

        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps({"_id": str(doc.get("_id"))}, sort_keys=True)
        cache_key = self._generate_cache_key(query_string)
        self.cache[normalized_collection][cache_key] = self.serialize_document(doc)
        logger.debug(f"Cache updated with inserted document in '{normalized_collection}' with key '{cache_key}'")

    async def _update_cache_with_update(self, collection: str, filter_query: dict, update_data: dict):
        """
        Helper method to update cache after an update operation.
        """
        # Generate cache key based on the filter
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(filter_query, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)

        if cache_key in self.cache[normalized_collection]:
            # Apply the update operators to the cached document
            self._apply_update_operator(
                self.cache[normalized_collection][cache_key],
                update_data
            )
            logger.debug(f"Cache updated with bulk update in '{normalized_collection}' with key '{cache_key}'")

            # Log the updated part of the document for verification
            updated_part = self.cache[normalized_collection][cache_key]
            logger.debug(f"Updated document: {updated_part}")
        else:
            logger.debug(f"No cache entry found for collection '{normalized_collection}' with key '{cache_key}'")

    async def clear_cache(self):
        """
        Clear the entire cache by reinitializing the defaultdict.
        """
        self.cache = defaultdict(dict)
        logger.info("Cache has been reinitialized.")

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
        Close the MongoDB client connection.
        """
        self.mongo_client.close()
