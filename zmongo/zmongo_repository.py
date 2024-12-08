import asyncio
import logging
import functools
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, List

import time
import json
import hashlib
from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import InsertOne, UpdateOne
from pymongo.errors import BulkWriteError
from pymongo.results import InsertOneResult, UpdateResult

# Load environment variables
load_dotenv('.env')

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
            self.mongo_uri, maxPoolSize=200  # Adjusted pool size
        )
        self.db = self.mongo_client[self.db_name]
        self.cache = defaultdict(dict)  # Cache structure: {collection: {cache_key: document}}

    def _normalize_collection_name(self, collection_name: str) -> str:
        return collection_name.strip().lower()

    @functools.lru_cache(maxsize=10000)
    def _generate_cache_key(self, query_string: str):
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

    async def find_document(self, collection: str, query: dict) -> dict:
        """
        Retrieve a single document from the specified MongoDB collection.
        Uses cache if available, otherwise fetches from MongoDB.
        """
        query_string = json.dumps(query, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)
        if cache_key in self.cache[collection]:
            logging.debug(f"Cache hit for collection {collection} with key {cache_key}")
            return self.cache[collection][cache_key]
        else:
            logging.debug(f"Cache miss for collection {collection} with key {cache_key}")

        coll = self.db[collection]
        document = await coll.find_one(filter=query)
        if document:
            serialized_document = self.serialize_document(document)
            self.cache[collection][cache_key] = serialized_document
            return serialized_document
        return None

    async def find_documents(
            self,
            collection: str,
            query: dict,
            limit: int = DEFAULT_QUERY_LIMIT,
            projection: dict = None,
            sort: list = None,
            skip: int = 0,
    ) -> list:
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
                logging.debug(f"Caching document in collection: '{collection}'")
                query_string = json.dumps({"_id": str(result.inserted_id)}, sort_keys=True)
                cache_key = self._generate_cache_key(query_string)
                self.cache[normalized_collection][cache_key] = self.serialize_document(document)
            else:
                logging.debug(f"Not caching document in collection: '{collection}'")

            return result
        except Exception as e:
            logging.error(f"Error inserting document into {collection}: {e}")
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
        except Exception as e:
            logging.error(f"Error saving embedding for document {document_id}: {e}")
            raise

    async def update_document(
            self, collection: str, query: dict, update_data: dict, upsert: bool = True
    ) -> UpdateResult:
        """
        Update a document in the specified MongoDB collection.
        Updates the cache accordingly.
        """
        coll = self.db[collection]
        try:
            update_result = await coll.update_one(
                filter=query, update=update_data, upsert=upsert
            )

            # Update cache if document was modified
            if update_result.modified_count > 0:
                query_string = json.dumps(query, sort_keys=True, default=str)
                cache_key = self._generate_cache_key(query_string)
                if cache_key in self.cache[collection]:
                    self._apply_update_operator(
                        self.cache[collection][cache_key], update_data
                    )
            return update_result
        except Exception as e:
            logging.error(f"Error updating document in {collection}: {e}")
            raise

    async def delete_document(self, collection: str, query: dict):
        """
        Delete a document from the specified MongoDB collection, updating the cache.
        """
        coll = self.db[collection]
        try:
            result = await coll.delete_one(query)
            if result.deleted_count > 0:
                query_string = json.dumps(query, sort_keys=True, default=str)
                cache_key = self._generate_cache_key(query_string)
                self.cache[collection].pop(cache_key, None)
            return result
        except Exception as e:
            logging.error(f"Error deleting document from {collection}: {e}")
            raise

    @staticmethod
    def serialize_document(document):
        """
        Converts ObjectId fields in a document to strings for JSON serialization.
        """
        if document is None:
            return None
        return json.loads(json_util.dumps(document))

    @staticmethod
    def _apply_update_operator(document, update_data):
        """
        Apply MongoDB update operators to the cached document.
        Supports $set, $unset, $inc, $push, and $addToSet.
        """
        for operator, fields in update_data.items():
            if operator == "$set":
                for key, value in fields.items():
                    document[key] = value
            elif operator == "$unset":
                for key in fields.keys():
                    document.pop(key, None)
            elif operator == "$inc":
                for key, value in fields.items():
                    document[key] = document.get(key, 0) + value
            elif operator == "$push":
                for key, value in fields.items():
                    if key not in document:
                        document[key] = []
                    document[key].append(value)
            elif operator == "$addToSet":
                for key, value in fields.items():
                    if key not in document:
                        document[key] = []
                    if value not in document[key]:
                        document[key].append(value)
            # Implement other operators as needed

    async def aggregate_documents(
            self, collection: str, pipeline: list, limit: int = DEFAULT_QUERY_LIMIT
    ) -> list:
        """
        Perform an aggregation operation on the specified MongoDB collection.
        """
        coll = self.db[collection]
        try:
            cursor = coll.aggregate(pipeline)
            documents = await cursor.to_list(length=limit)
            return documents
        except Exception as e:
            logging.error(f"Error during aggregation on {collection}: {e}")
            raise

    async def bulk_write(self, collection: str, operations: list):
        """
        Perform bulk write operations, updating the cache.
        """
        coll = self.db[collection]
        try:
            result = await coll.bulk_write(operations)

            # Separate Insert and Update operations
            # Access protected members because public attributes are not available
            insert_docs = [op._doc for op in operations if isinstance(op, InsertOne)]
            update_ops = [op for op in operations if isinstance(op, UpdateOne)]

            # Process Insert operations concurrently
            insert_tasks = [
                self._update_cache_with_insert(collection, doc)
                for doc in insert_docs
            ]
            await asyncio.gather(*insert_tasks)

            # Process Update operations concurrently
            update_tasks = [
                self._update_cache_with_update(collection, op)
                for op in update_ops
            ]
            await asyncio.gather(*update_tasks)

            return result
        except BulkWriteError as e:
            logging.error(f"Bulk write error in {collection}: {e.details}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during bulk write in {collection}: {e}")
            raise

    async def _update_cache_with_insert(self, collection: str, doc: dict):
        """
        Helper method to update cache after an InsertOne operation.
        """
        query_string = json.dumps({"_id": str(doc.get("_id"))}, sort_keys=True)
        cache_key = self._generate_cache_key(query_string)
        self.cache[collection][cache_key] = self.serialize_document(doc)

    async def _update_cache_with_update(self, collection: str, op: UpdateOne):
        """
        Helper method to update cache after an UpdateOne operation.
        """
        filter_dict = op._filter
        query_string = json.dumps(filter_dict, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)
        if cache_key in self.cache[collection]:
            self._apply_update_operator(
                self.cache[collection][cache_key], op._update
            )

    async def clear_cache(self):
        """
        Clear the entire cache by reinitializing the defaultdict.
        """
        self.cache = defaultdict(dict)
        logging.info("Cache has been reinitialized.")

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
        logging.info(f"Performance log inserted: {performance_data}")

    async def close(self):
        """
        Close the MongoDB client connection.
        """
        self.mongo_client.close()


# Performance Tests and High-Load Testing

async def high_load_test(repository: ZMongoRepository, num_operations=1000):
    """
    Perform a high-load test on the ZMongoRepository by running concurrent operations.
    """
    semaphore = asyncio.Semaphore(1000)  # Limit concurrency to prevent overwhelming the event loop

    inserted_ids = []

    # Define async tasks for each method

    async def insert_test_user(i):
        """Insert a test user."""
        async with semaphore:
            document = {"name": f"Test User {i}", "age": 20 + i, "creator": "admin"}
            result = await repository.insert_document(
                collection=TEST_COLLECTION_NAME, document=document
            )
            return result.inserted_id

    async def find_test_user(i):
        """Find a test user."""
        async with semaphore:
            query = {"name": f"Test User {i}"}
            document = await repository.find_document(
                collection=TEST_COLLECTION_NAME, query=query
            )
            return document

    async def update_test_user(i):
        """Update a test user."""
        async with semaphore:
            query = {"name": f"Test User {i}"}
            update_data = {"$set": {"age": 30 + i}}
            result = await repository.update_document(
                collection=TEST_COLLECTION_NAME, query=query, update_data=update_data
            )
            return result

    async def delete_test_user(document_id):
        """Delete a test user by document ID."""
        async with semaphore:
            query = {"_id": document_id}
            result = await repository.delete_document(
                collection=TEST_COLLECTION_NAME,
                query=query
            )
            return result

    async def fetch_embedding_test(document_id):
        """Fetch embedding for a test user."""
        async with semaphore:
            embedding = await repository.fetch_embedding(
                collection=TEST_COLLECTION_NAME,
                document_id=document_id
            )
            return embedding

    async def save_embedding_test(document_id, embedding):
        """Save embedding for a test user."""
        async with semaphore:
            await repository.save_embedding(
                collection=TEST_COLLECTION_NAME,
                document_id=document_id,
                embedding=embedding
            )

    async def aggregate_test():
        """Perform an aggregation on test users."""
        pipeline = [
            {"$match": {"creator": "admin"}},
            {"$group": {"_id": "$creator", "average_age": {"$avg": "$age"}}}
        ]
        result = await repository.aggregate_documents(
            collection=TEST_COLLECTION_NAME,
            pipeline=pipeline
        )
        return result

    async def bulk_write_test(start_index):
        """Perform bulk insert and update operations."""
        operations = []
        for i in range(start_index, start_index + 100):
            operations.append(InsertOne({"name": f"Bulk User {i}", "age": 25 + i}))
            operations.append(UpdateOne(
                {"name": f"Bulk User {i}"},
                {"$set": {"age": 35 + i}},
                upsert=True
            ))
        await repository.bulk_write(TEST_COLLECTION_NAME, operations)

    async def clear_cache_test():
        """Clear the repository cache."""
        await repository.clear_cache()

    # Execute Tests in Sequence with Performance Logging

    # 1. Insert Users Concurrently
    logging.info(f"Starting insert operations for {num_operations} users...")
    start_time = time.time()
    insert_tasks = [insert_test_user(i) for i in range(num_operations)]
    inserted_ids = await asyncio.gather(*insert_tasks)
    insert_duration = time.time() - start_time
    await repository.log_performance("insert", insert_duration, num_operations)
    logging.info(f"Insert operations completed in {insert_duration:.2f} seconds.")

    # 2. Find Users Concurrently
    logging.info(f"Starting find operations for {num_operations} users...")
    start_time = time.time()
    find_tasks = [find_test_user(i) for i in range(num_operations)]
    find_results = await asyncio.gather(*find_tasks)
    find_duration = time.time() - start_time
    await repository.log_performance("find", find_duration, num_operations)
    logging.info(f"Find operations completed in {find_duration:.2f} seconds.")

    # 3. Update Users Concurrently
    logging.info(f"Starting update operations for {num_operations} users...")
    start_time = time.time()
    update_tasks = [update_test_user(i) for i in range(num_operations)]
    update_results = await asyncio.gather(*update_tasks)
    update_duration = time.time() - start_time
    await repository.log_performance("update", update_duration, num_operations)
    logging.info(f"Update operations completed in {update_duration:.2f} seconds.")

    # 4. Fetch Embeddings Concurrently
    logging.info(f"Starting fetch_embedding operations for {num_operations} users...")
    start_time = time.time()
    fetch_tasks = [fetch_embedding_test(doc_id) for doc_id in inserted_ids]
    embeddings = await asyncio.gather(*fetch_tasks)
    fetch_duration = time.time() - start_time
    await repository.log_performance("fetch_embedding", fetch_duration, num_operations)
    logging.info(f"Fetch_embedding operations completed in {fetch_duration:.2f} seconds.")

    # 5. Save Embeddings Concurrently
    logging.info(f"Starting save_embedding operations for {num_operations} users...")
    start_time = time.time()
    save_tasks = [
        save_embedding_test(doc_id, [0.1, 0.2, 0.3, 0.4, 0.5])
        for doc_id in inserted_ids
    ]
    await asyncio.gather(*save_tasks)
    save_duration = time.time() - start_time
    await repository.log_performance("save_embedding", save_duration, num_operations)
    logging.info(f"Save_embedding operations completed in {save_duration:.2f} seconds.")

    # 6. Aggregate Documents
    logging.info("Starting aggregation operations...")
    start_time = time.time()
    aggregate_results = await aggregate_test()
    aggregate_duration = time.time() - start_time
    await repository.log_performance("aggregate", aggregate_duration, 1)
    logging.info(f"Aggregation operation completed in {aggregate_duration:.2f} seconds. Result: {aggregate_results}")

    # 7. Bulk Write Operations
    bulk_operations = num_operations // 100  # 100 operations per bulk_write
    logging.info(f"Starting bulk_write operations ({bulk_operations} batches)...")
    start_time = time.time()
    bulk_tasks = [bulk_write_test(i * 100) for i in range(bulk_operations)]
    await asyncio.gather(*bulk_tasks)
    bulk_duration = time.time() - start_time
    await repository.log_performance("bulk_write", bulk_duration,
                                     num_operations * 2)  # Each bulk_write handles 2 operations per iteration
    logging.info(f"Bulk_write operations completed in {bulk_duration:.2f} seconds.")

    # 8. Delete Users Concurrently
    logging.info(f"Starting delete operations for {num_operations} users...")
    start_time = time.time()
    delete_tasks = [delete_test_user(document_id) for document_id in inserted_ids]
    await asyncio.gather(*delete_tasks)
    delete_duration = time.time() - start_time
    await repository.log_performance("delete", delete_duration, num_operations)
    logging.info(f"Delete operations completed in {delete_duration:.2f} seconds.")

    # 9. Clear Cache
    logging.info("Starting cache clearing operation...")
    start_time = time.time()
    await clear_cache_test()
    clear_cache_duration = time.time() - start_time
    await repository.log_performance("clear_cache", clear_cache_duration, 1)
    logging.info(f"Cache cleared in {clear_cache_duration:.2f} seconds.")

    # 10. Verify Cache is Empty
    logging.info("Verifying cache is empty...")
    is_cache_empty = all(not cache for cache in repository.cache.values())
    if is_cache_empty:
        logging.info("Cache verification successful: Cache is empty.")
    else:
        logging.warning("Cache verification failed: Cache is not empty.")


# Main execution for testing
if __name__ == "__main__":
    async def main():
        repository = ZMongoRepository()
        await repository.clear_cache()
        try:
            # Run high-load test with 5,000 concurrent operations
            await high_load_test(repository, num_operations=5000)
        except Exception as e:
            logging.error(f"Error during high-load test: {e}")
        finally:
            await repository.close()


    asyncio.run(main())
