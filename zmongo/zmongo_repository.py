"""
ZMongoRepository
================

This module provides a professionally documented and cleaned-up implementation
of an asynchronous MongoDB repository with an in-memory caching mechanism for
improved performance. It also supports custom MongoDB update operators and
bulk write operations.
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Any

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import BulkWriteError
from pymongo.results import (
    InsertOneResult,
    DeleteResult,
    BulkWriteResult,
)

# Load environment variables
load_dotenv()

# Retrieve and validate environment variables
DEFAULT_QUERY_LIMIT = os.getenv("DEFAULT_QUERY_LIMIT")
if DEFAULT_QUERY_LIMIT is not None:
    try:
        DEFAULT_QUERY_LIMIT = int(DEFAULT_QUERY_LIMIT)
    except ValueError:
        raise ValueError("DEFAULT_QUERY_LIMIT must be an integer.")
else:
    DEFAULT_QUERY_LIMIT = 100  # Default if not provided

TEST_COLLECTION_NAME = os.getenv("TEST_COLLECTION_NAME")
if not TEST_COLLECTION_NAME:
    raise ValueError("TEST_COLLECTION_NAME must be set in the environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZMongoRepository:
    """
    Asynchronous MongoDB repository providing an in-memory cache and
    support for various MongoDB operations.

    Attributes:
        mongo_uri (str): URI for connecting to MongoDB.
        db_name (str): Name of the MongoDB database.
        mongo_client (AsyncIOMotorClient): Async MongoDB client instance.
        db (Database): The MongoDB database object.
        cache (defaultdict): In-memory cache storing {collection: {cache_key: document}}.
    """

    def __init__(self):
        """
        Initialize the ZMongoRepository using environment variables. Verifies that
        MONGO_URI and MONGO_DATABASE_NAME are set. An in-memory cache is maintained
        to reduce subsequent database queries for the same query parameters.
        """
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("MONGO_URI must be set in the environment variables.")

        self.db_name = os.getenv("MONGO_DATABASE_NAME")
        if not self.db_name or not isinstance(self.db_name, str):
            raise ValueError(
                "MONGO_DATABASE_NAME must be set in the environment variables as a string."
            )

        self.mongo_client = AsyncIOMotorClient(self.mongo_uri, maxPoolSize=200)
        self.db = self.mongo_client[self.db_name]
        self.cache = defaultdict(dict)

    def _normalize_collection_name(self, collection_name: str) -> str:
        """
        Normalize a collection name by stripping whitespace and converting to lowercase.

        Args:
            collection_name (str): The original collection name.

        Returns:
            str: The normalized collection name.
        """
        return collection_name.strip().lower()

    @functools.lru_cache(maxsize=10000)
    def _generate_cache_key(self, query_string: str) -> str:
        """
        Generate a unique cache key from a query string using a SHA-256 hash.

        Args:
            query_string (str): A JSON-serialized representation of a MongoDB query.

        Returns:
            str: The SHA-256 hash of the query string.
        """
        return hashlib.sha256(query_string.encode("utf-8")).hexdigest()

    async def fetch_embedding(
        self,
        collection: str,
        document_id: ObjectId,
        embedding_field: str = "embedding",
    ) -> Optional[List[float]]:
        """
        Retrieve an embedding list from a document in the specified collection.

        Args:
            collection (str): The name of the MongoDB collection.
            document_id (ObjectId): The document ID to fetch.
            embedding_field (str, optional): The field containing the embedding.
                Defaults to "embedding".

        Returns:
            Optional[List[float]]: The embedding if present, otherwise None.
        """
        coll = self.db[collection]
        document = await coll.find_one({"_id": document_id}, {embedding_field: 1})
        if document:
            embedding_value = document.get(embedding_field)
            return embedding_value
        return None

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """
        Find a single document in the given collection matching the specified query.

        This method first checks the in-memory cache for the query result. If not
        found, it queries MongoDB. The result is then cached for subsequent queries.

        Args:
            collection (str): The name of the MongoDB collection.
            query (dict): The MongoDB query to filter documents.

        Returns:
            Optional[dict]: The matching document, or None if not found.
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(query, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)

        if cache_key in self.cache[normalized_collection]:
            logger.debug(
                f"Cache hit for collection '{normalized_collection}' with key '{cache_key}'"
            )
            return self.cache[normalized_collection][cache_key]
        else:
            logger.debug(
                f"Cache miss for collection '{normalized_collection}' with key '{cache_key}'"
            )

        coll = self.db[collection]
        document = await coll.find_one(filter=query)
        if document:
            serialized_document = self.serialize_document(document)
            self.cache[normalized_collection][cache_key] = serialized_document
            logger.debug(
                f"Document cached for collection '{normalized_collection}' with key '{cache_key}'"
            )
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

        Args:
            collection (str): The name of the MongoDB collection.
            query (dict): The MongoDB query to filter documents.
            limit (int, optional): The maximum number of documents to return.
                Defaults to the environment variable DEFAULT_QUERY_LIMIT.
            projection (dict, optional): Fields to include or exclude. Defaults to None.
            sort (List[Any], optional): Sort specification. Defaults to None.
            skip (int, optional): Number of documents to skip. Defaults to 0.

        Returns:
            List[dict]: A list of retrieved documents.
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
        Insert a document into a MongoDB collection and update the in-memory cache.

        Args:
            collection (str): The name of the MongoDB collection.
            document (dict): The document to insert.

        Returns:
            InsertOneResult: The result of the insert operation, containing the inserted_id.
        """
        coll = self.db[collection]
        try:
            result = await coll.insert_one(document=document)
            document["_id"] = result.inserted_id

            normalized_collection = self._normalize_collection_name(collection)
            # Exclude 'performance_tests' from caching
            if normalized_collection != "performance_tests":
                logger.debug(f"Caching document in collection: '{normalized_collection}'")
                query_string = json.dumps({"_id": str(result.inserted_id)}, sort_keys=True)
                cache_key = self._generate_cache_key(query_string)
                self.cache[normalized_collection][cache_key] = self.serialize_document(
                    document
                )
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
        embedding_field: str = "embedding",
    ):
        """
        Save an embedding list to a document in the specified collection.

        Args:
            collection (str): The name of the MongoDB collection.
            document_id (ObjectId): The ID of the document to update.
            embedding (List[float]): The embedding data to store.
            embedding_field (str, optional): The field name for the embedding.
                Defaults to "embedding".
        """
        coll = self.db[collection]
        try:
            await coll.update_one(
                {"_id": document_id}, {"$set": {embedding_field: embedding}}, upsert=True
            )
            logger.debug(
                f"Embedding saved for document '{document_id}' in collection '{collection}'."
            )
        except Exception as e:
            logger.error(f"Error saving embedding for document '{document_id}': {e}")
            raise

    async def update_document(
        self,
        collection: str,
        update_data: dict,
        query: dict,
        upsert: bool = False,
    ) -> bool:
        """
        Perform a partial update (e.g., $set, $push) on a document and update the cache.

        Args:
            collection (str): The name of the MongoDB collection.
            update_data (dict): The update specification (operators and fields).
            query (dict): The MongoDB query to match a document.
            upsert (bool, optional): Whether to insert if no document matches.
                Defaults to False.

        Returns:
            bool: True if a document was modified or upserted, False otherwise.
        """
        try:
            # Apply the update in MongoDB
            result = await self.db[collection].update_one(
                filter=query, update=update_data, upsert=upsert
            )

            # Successful if something was modified or upserted
            success = (result.modified_count > 0) or (result.upserted_id is not None)

            if success:
                normalized_coll = self._normalize_collection_name(collection)
                query_str = json.dumps(query, sort_keys=True, default=str)
                cache_key = self._generate_cache_key(query_str)

                if cache_key in self.cache[normalized_coll]:
                    # Apply update operators to the cached document
                    self._apply_update_operator(
                        self.cache[normalized_coll][cache_key],
                        update_data
                    )
                    logger.debug(
                        f"Cache updated for collection '{normalized_coll}' with key '{cache_key}'"
                    )

                    updated_part = self.cache[normalized_coll][cache_key]
                    logger.debug(f"Updated document: {updated_part}")
                else:
                    logger.debug(
                        f"No cache entry found for collection '{normalized_coll}' "
                        f"with key '{cache_key}'"
                    )

            return success
        except Exception as e:
            error_detail = getattr(e, "details", str(e))
            logger.error(
                f"Error updating document in '{collection}': {e}, full error: {error_detail}"
            )
            raise

    async def delete_document(self, collection: str, query: dict) -> Optional[DeleteResult]:
        """
        Delete a document from the specified MongoDB collection and update the cache.

        Args:
            collection (str): The name of the MongoDB collection.
            query (dict): The MongoDB query to match the document to delete.

        Returns:
            Optional[DeleteResult]: The result of the delete operation, or None on error.
        """
        coll = self.db[collection]
        try:
            result = await coll.delete_one(query)
            if result.deleted_count > 0:
                normalized_collection = self._normalize_collection_name(collection)
                query_string = json.dumps(query, sort_keys=True, default=str)
                cache_key = self._generate_cache_key(query_string)
                self.cache[normalized_collection].pop(cache_key, None)
                logger.debug(
                    f"Cache updated: Document with query '{query}' removed from cache."
                )
            return result
        except Exception as e:
            logger.error(f"Error deleting document from '{collection}': {e}")
            raise

    @staticmethod
    def serialize_document(document: dict) -> dict:
        """
        Serialize a MongoDB document by converting special types (like ObjectId)
        into JSON-friendly formats.

        Args:
            document (dict): The MongoDB document to serialize.

        Returns:
            dict: The JSON-serializable form of the document.
        """
        if document is None:
            return None
        return json.loads(json_util.dumps(document))

    @staticmethod
    def _apply_update_operator(document: dict, update_data: dict):
        """
        Apply MongoDB-like update operators ($set, $unset, $inc, $push, $addToSet)
        to a cached document in-memory.

        Args:
            document (dict): The cached document to update.
            update_data (dict): The update specification, e.g. {"$set": {...}}.
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
                        logger.debug(
                            f"$inc applied on '{key_path}' with value '{value}' (initialized)"
                        )
                    else:
                        ZMongoRepository._set_nested_value(
                            document, key_path, current + value
                        )
                        logger.debug(
                            f"$inc applied on '{key_path}' with value '{value}' (incremented)"
                        )

            elif operator == "$push":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logger.debug(
                            f"$push applied on '{key_path}' with value '{value}' (initialized list)"
                        )
                    elif isinstance(current, list):
                        current.append(value)
                        logger.debug(
                            f"$push applied on '{key_path}' with value '{value}' (appended)"
                        )
                    else:
                        logger.warning(
                            f"Cannot push to non-list field: '{key_path}'"
                        )

            elif operator == "$addToSet":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logger.debug(
                            f"$addToSet applied on '{key_path}' with value '{value}' (initialized list)"
                        )
                    elif isinstance(current, list) and value not in current:
                        current.append(value)
                        logger.debug(
                            f"$addToSet applied on '{key_path}' with value '{value}' (added to set)"
                        )

            # Additional MongoDB operators can be added as needed.

    @staticmethod
    def _set_nested_value(document: dict, key_path: str, value: Any):
        """
        Set a nested value in a dict or list based on a dot-separated key path.

        Args:
            document (dict): The document to modify.
            key_path (str): Dot-separated path to the value, e.g. "address.street".
            value (Any): The value to set.
        """
        keys = key_path.split(".")
        for key in keys[:-1]:
            if key.isdigit():
                key_index = int(key)
                if not isinstance(document, list):
                    logger.warning(f"Expected a list at key '{key}'")
                    return
                while len(document) <= key_index:
                    document.append({})
                document = document[key_index]
            else:
                if key not in document or not isinstance(document[key], dict):
                    document[key] = {}
                document = document[key]

        last_key = keys[-1]
        if last_key.isdigit():
            last_index = int(last_key)
            if not isinstance(document, list):
                logger.warning(f"Expected a list at key '{last_key}'")
                return
            while len(document) <= last_index:
                document.append({})
            document[last_index] = value
        else:
            document[last_key] = value

    @staticmethod
    def _unset_nested_key(document: dict, key_path: str):
        """
        Unset (remove) a nested key in a dict or list based on a dot-separated path.

        Args:
            document (dict): The document to modify.
            key_path (str): Dot-separated path to the value, e.g. "address.street".
        """
        keys = key_path.split(".")
        for key in keys[:-1]:
            if key.isdigit():
                key_index = int(key)
                if not isinstance(document, list) or key_index >= len(document):
                    return
                document = document[key_index]
            else:
                if key not in document:
                    return
                document = document[key]

        last_key = keys[-1]
        if last_key.isdigit():
            last_index = int(last_key)
            if isinstance(document, list) and last_index < len(document):
                document.pop(last_index)
                logger.debug(f"Unset list element at index '{last_index}'")
        else:
            document.pop(last_key, None)
            logger.debug(f"Unset field '{last_key}'")

    @staticmethod
    def _get_nested_value(document: dict, key_path: str) -> Optional[Any]:
        """
        Retrieve a nested value from a dict or list based on a dot-separated key path.

        Args:
            document (dict): The document to read.
            key_path (str): Dot-separated path to the value, e.g. "address.street".

        Returns:
            Optional[Any]: The retrieved value, or None if not found.
        """
        keys = key_path.split(".")
        for key in keys:
            if key.isdigit():
                key_index = int(key)
                if not isinstance(document, list) or key_index >= len(document):
                    return None
                document = document[key_index]
            else:
                if key not in document:
                    return None
                document = document[key]
        return document

    async def aggregate_documents(
        self, collection: str, pipeline: list, limit: int = DEFAULT_QUERY_LIMIT
    ) -> List[dict]:
        """
        Perform an aggregation operation on a MongoDB collection.

        Args:
            collection (str): The name of the MongoDB collection.
            pipeline (list): List of aggregation pipeline stages.
            limit (int, optional): The maximum number of documents to return.
                Defaults to DEFAULT_QUERY_LIMIT.

        Returns:
            List[dict]: The aggregation results.
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
        Perform multiple insert or update operations in a single bulk write,
        updating the in-memory cache.

        Each operation in `operations` should be a dictionary of the form:
            - Insert Operation:
                {
                    "action": "insert",
                    "document": { ... }
                }
            - Update Operation:
                {
                    "action": "update",
                    "filter": { ... },
                    "update": { ... },
                    "upsert": True
                }

        Args:
            collection (str): The name of the MongoDB collection.
            operations (list): A list of bulk operations.

        Returns:
            Optional[BulkWriteResult]: The result of the bulk write, or None if all
            operations are performed individually without using BulkWriteResult.
        """
        coll = self.db[collection]
        try:
            # 1. Separate insert and update operations
            insert_docs = [
                op["document"]
                for op in operations
                if op.get("action") == "insert" and "document" in op
            ]
            update_ops = [
                op
                for op in operations
                if op.get("action") == "update"
                and "filter" in op
                and "update" in op
            ]

            # 2. Log any malformed operations
            for idx, op in enumerate(operations):
                if op.get("action") == "insert" and "document" not in op:
                    logger.error(
                        f"Insert operation at index {idx} missing 'document': {op}"
                    )
                if op.get("action") == "update" and (
                    "filter" not in op or "update" not in op
                ):
                    logger.error(
                        f"Update operation at index {idx} missing 'filter' or 'update': {op}"
                    )
                if op.get("action") not in ["insert", "update"]:
                    logger.warning(f"Unsupported operation type at index {idx}: {op}")

            # 3. Perform insert operations
            if insert_docs:
                logger.info(
                    f"Performing {len(insert_docs)} insert operations on '{collection}'."
                )
                await coll.insert_many(insert_docs)

                # Update cache with inserted documents
                insert_tasks = [
                    self._update_cache_with_insert(collection, doc)
                    for doc in insert_docs
                ]
                await asyncio.gather(*insert_tasks)
            else:
                logger.warning("No valid insert operations found to perform.")

            # 4. Perform update operations
            if update_ops:
                logger.info(
                    f"Performing {len(update_ops)} update operations on '{collection}'."
                )
                for op in update_ops:
                    result = await coll.update_one(
                        filter=op["filter"],
                        update=op["update"],
                        upsert=op.get("upsert", False),
                    )
                    logger.debug(
                        f"Update result for filter={op['filter']}: {result.raw_result}"
                    )
                    await self._update_cache_with_update(collection, op["filter"], op["update"])
            else:
                logger.warning("No valid update operations found to perform.")

            # Here you could return a consolidated BulkWriteResult if needed
            return None

        except BulkWriteError as e:
            logger.error(f"Bulk write error in '{collection}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during bulk write in '{collection}': {e}")
            raise

    async def _update_cache_with_insert(self, collection: str, doc: dict):
        """
        Update the in-memory cache after an insert operation.

        Args:
            collection (str): The name of the MongoDB collection.
            doc (dict): The inserted document.
        """
        if "_id" not in doc:
            logger.error(f"Inserted document missing '_id': {doc}")
            return

        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps({"_id": str(doc.get("_id"))}, sort_keys=True)
        cache_key = self._generate_cache_key(query_string)
        self.cache[normalized_collection][cache_key] = self.serialize_document(doc)
        logger.debug(
            f"Cache updated with inserted document in '{normalized_collection}' with key '{cache_key}'"
        )

    async def _update_cache_with_update(self, collection: str, filter_query: dict, update_data: dict):
        """
        Update the in-memory cache after an update operation.

        Args:
            collection (str): The name of the MongoDB collection.
            filter_query (dict): The MongoDB filter used to match the document.
            update_data (dict): The update specification (operators and fields).
        """
        normalized_collection = self._normalize_collection_name(collection)
        query_string = json.dumps(filter_query, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)

        if cache_key in self.cache[normalized_collection]:
            self._apply_update_operator(
                self.cache[normalized_collection][cache_key], update_data
            )
            logger.debug(
                f"Cache updated with bulk update in '{normalized_collection}' with key '{cache_key}'"
            )
            updated_part = self.cache[normalized_collection][cache_key]
            logger.debug(f"Updated document: {updated_part}")
        else:
            logger.debug(
                f"No cache entry found for collection '{normalized_collection}' with key '{cache_key}'"
            )

    async def clear_cache(self):
        """
        Clear the entire in-memory cache, removing all cached documents.
        """
        self.cache = defaultdict(dict)
        logger.info("Cache has been reinitialized.")

    async def log_performance(self, operation: str, duration: float, num_operations: int):
        """
        Log performance metrics into a dedicated MongoDB collection (performance_tests).

        Args:
            operation (str): A descriptive name for the operation performed.
            duration (float): Total duration of the operation in seconds.
            num_operations (int): The number of operations performed in this batch.
        """
        performance_data = {
            "operation": operation,
            "num_operations": num_operations,
            "duration_seconds": duration,
            "avg_duration_per_operation": duration / num_operations if num_operations else 0,
            "timestamp": datetime.now(),
        }
        await self.insert_document("performance_tests", performance_data)
        logger.info(f"Performance log inserted: {performance_data}")

    async def close(self):
        """
        Close the MongoDB client connection.
        """
        self.mongo_client.close()
        logger.info("MongoDB client connection closed.")

import asyncio
import logging
import random
from bson import ObjectId

# Import your ZMongoRepository from where you've placed it
# For example:
# from zmongo_repository import ZMongoRepository

async def main():
    # Adjust the logging level as needed for demonstration
    logging.getLogger().setLevel(logging.DEBUG)

    # Instantiate the repository
    repo = ZMongoRepository()

    # This collection name should exist in your MongoDB, or you can specify a new one
    test_collection = "test_collection"

    try:
        # 1. Insert a document
        doc_to_insert = {
            "name": "Alice",
            "role": "Engineer",
            "skills": ["Python", "Databases"],
            "score": 10,
        }
        insert_result = await repo.insert_document(test_collection, doc_to_insert)
        print(f"Inserted document ID: {insert_result.inserted_id}")

        # 2. Find a single document
        found_doc = await repo.find_document(
            test_collection,
            {"_id": insert_result.inserted_id}
        )
        print("Found document:", found_doc)

        # 3. Find multiple documents
        # (We can insert another one so that we have multiple documents)
        await repo.insert_document(test_collection, {"name": "Bob", "role": "Designer"})
        documents = await repo.find_documents(test_collection, query={}, limit=10)
        print("All documents in collection (limit=10):", documents)

        # 4. Update a document
        update_query = {"_id": insert_result.inserted_id}
        update_data = {"$set": {"score": 20}, "$push": {"skills": "MongoDB"}}
        update_success = await repo.update_document(
            test_collection,
            update_data=update_data,
            query=update_query
        )
        print(f"Update success: {update_success}")

        # Check the updated document
        updated_doc = await repo.find_document(test_collection, {"_id": insert_result.inserted_id})
        print("Updated document:", updated_doc)

        # 5. Save and fetch an embedding (simulate a 768-dimensional embedding)
        fake_embedding = [random.random() for _ in range(4)]  # shortened for demo
        await repo.save_embedding(test_collection, insert_result.inserted_id, fake_embedding)
        embedding_result = await repo.fetch_embedding(test_collection, insert_result.inserted_id)
        print("Fetched embedding:", embedding_result)

        # 6. Aggregate documents
        # Example pipeline: match all, then limit
        pipeline = [
            {"$match": {}},
            {"$limit": 5},
        ]
        aggregated_docs = await repo.aggregate_documents(test_collection, pipeline)
        print("Aggregated documents (up to 5):", aggregated_docs)

        # 7. Bulk write operations
        bulk_ops = [
            {
                "action": "insert",
                "document": {"name": "Charlie", "role": "DevOps"},
            },
            {
                "action": "update",
                "filter": {"name": "Alice"},
                "update": {"$inc": {"score": 5}},
                "upsert": False,
            },
        ]
        await repo.bulk_write(test_collection, bulk_ops)
        # Check results
        charlie_doc = await repo.find_document(test_collection, {"name": "Charlie"})
        print("Inserted via bulk_write:", charlie_doc)
        alice_after_bulk = await repo.find_document(test_collection, {"name": "Alice"})
        print("Alice after bulk update:", alice_after_bulk)

        # 8. Log performance (example values)
        # Typically you'd measure real start/stop times around your ops
        await repo.log_performance("demo_operations", duration=2.5, num_operations=3)

        # 9. Delete a document
        delete_result = await repo.delete_document(test_collection, {"name": "Charlie"})
        print(f"Deleted count for Charlie: {delete_result.deleted_count}")

        # 10. Clear the cache (optional)
        await repo.clear_cache()

    finally:
        # 11. Close the MongoDB client connection
        await repo.close()


if __name__ == "__main__":
    asyncio.run(main())
