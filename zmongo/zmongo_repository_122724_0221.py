import asyncio
import logging
import functools
import os
import uuid
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
                    logging.debug(f"Cache updated for collection '{collection}' with key '{cache_key}'")
                else:
                    logging.debug(f"No cache entry found for collection '{collection}' with key '{cache_key}'")

            return success

        except Exception as e:
            logging.error(
                f"Error updating document in {collection}: {e}, "
                f"full error: {e.details if hasattr(e, 'details') else e}"
            )
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
    def _apply_update_operator(document: dict, update_data: dict):
        """
        Apply MongoDB update operators to the cached document.
        Supports $set, $unset, $inc, $push, and $addToSet with nested keys.
        """
        for operator, fields in update_data.items():
            if operator == "$set":
                for key_path, value in fields.items():
                    ZMongoRepository._set_nested_value(document, key_path, value)
                    logging.debug(f"$set applied on '{key_path}' with value '{value}'")
            elif operator == "$unset":
                for key_path in fields.keys():
                    ZMongoRepository._unset_nested_key(document, key_path)
                    logging.debug(f"$unset applied on '{key_path}'")
            elif operator == "$inc":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, value)
                        logging.debug(f"$inc applied on '{key_path}' with value '{value}' (initialized)")
                    else:
                        ZMongoRepository._set_nested_value(document, key_path, current + value)
                        logging.debug(f"$inc applied on '{key_path}' with value '{value}' (incremented)")
            elif operator == "$push":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logging.debug(f"$push applied on '{key_path}' with value '{value}' (initialized list)")
                    elif isinstance(current, list):
                        current.append(value)
                        logging.debug(f"$push applied on '{key_path}' with value '{value}' (appended)")
                    else:
                        # Handle error: trying to push to a non-list field
                        logging.warning(f"Cannot push to non-list field: {key_path}")
            elif operator == "$addToSet":
                for key_path, value in fields.items():
                    current = ZMongoRepository._get_nested_value(document, key_path)
                    if current is None:
                        ZMongoRepository._set_nested_value(document, key_path, [value])
                        logging.debug(f"$addToSet applied on '{key_path}' with value '{value}' (initialized list)")
                    elif isinstance(current, list) and value not in current:
                        current.append(value)
                        logging.debug(f"$addToSet applied on '{key_path}' with value '{value}' (added to set)")
            # Implement other operators as needed

    @staticmethod
    def _set_nested_value(document: dict, key_path: str, value):
        """
        Set a value in a nested dictionary or list based on the dot-separated key path.
        """
        keys = key_path.split('.')
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
                if not isinstance(document, list):
                    logging.warning(f"Expected list at key: {key}")
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
                logging.warning(f"Expected list at key: {last_key}")
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
                logging.debug(f"Unset list element at index {last_key}")
        else:
            document.pop(last_key, None)
            logging.debug(f"Unset field '{last_key}'")

    @staticmethod
    def _get_nested_value(document: dict, key_path: str):
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
        logging.debug(f"Cache updated with inserted document in '{collection}' with key '{cache_key}'")

    async def _update_cache_with_update(self, collection: str, op: UpdateOne):
        """
        Helper method to update cache after an UpdateOne operation.
        """
        # Safely access protected attributes using getattr
        filter_dict = getattr(op, '_filter', None)
        update_data = getattr(op, '_update', None)

        if filter_dict is None or update_data is None:
            logging.error("UpdateOne operation missing '_filter' or '_update' attribute.")
            return

        query_string = json.dumps(filter_dict, sort_keys=True, default=str)
        cache_key = self._generate_cache_key(query_string)
        if cache_key in self.cache[collection]:
            self._apply_update_operator(
                self.cache[collection][cache_key],
                update_data
            )
            logging.debug(f"Cache updated with bulk update in '{collection}' with key '{cache_key}'")
        else:
            logging.debug(f"No cache entry found for collection '{collection}' with key '{cache_key}'")

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

# Placeholder for LlamaModel class
class LlamaModel:
    def __init__(self):
        # Initialize your model here
        pass


# Main function for interactive testing
async def main_interactive():
    logging.basicConfig(level=logging.INFO)
    verification = LlamaInteractiveVerification()

    while True:
        print("\nChoose an action:")
        print("1. Create a new conversation")
        print("2. Add an interaction to a conversation")
        print("3. Edit an interaction")
        print("4. Clear conversation history")
        print("5. Mark an interaction as verified")
        print("6. Retrieve conversation history")
        print("7. Show available conversation IDs (all)")
        print("8. Delete a conversation")
        print("9. Show conversation IDs for a username")
        print("10. Exit")

        try:
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                username = input("Enter the username for this conversation: ").strip()
                conversation_id = await verification.create_new_conversation(username)
                print(f"New conversation created. conversation_id: {conversation_id}, user: '{username}'")

            elif choice == "2":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                ai_question = input("Enter the AI question: ").strip()
                user_response = input("Enter the user response: ").strip()
                verified_input = input("Is the response verified? (yes/no): ").strip().lower()
                verified = verified_input == "yes"
                await verification.save_interaction(conversation_id, ai_question, user_response, verified)
                print(f"Interaction added to conversation_id {conversation_id}.")

            elif choice == "3":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                interaction_id_input = input("Enter the interaction ID (1-based): ").strip()
                if not interaction_id_input.isdigit():
                    logging.error("Invalid interaction ID. It must be a positive integer.")
                    continue
                interaction_id = int(interaction_id_input)
                print("Enter the fields to update (leave blank to skip):")
                ai_question = input("New AI question: ").strip()
                user_response = input("New user response: ").strip()
                verified_str = input("Is the response verified? (yes/no): ").strip().lower()
                new_data = {}
                if ai_question:
                    new_data["ai_question"] = ai_question
                if user_response:
                    new_data["user_response"] = user_response
                if verified_str in {"yes", "no"}:
                    new_data["verified"] = (verified_str == "yes")

                if not new_data:
                    print("No valid fields to update. Operation skipped.")
                    continue

                await verification.edit_interaction(conversation_id, interaction_id, new_data)

            elif choice == "4":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                await verification.clear_conversation_history(conversation_id)
                print(f"Conversation history cleared for conversation_id {conversation_id}.")

            elif choice == "5":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                interaction_id_input = input("Enter the interaction ID (1-based): ").strip()
                if not interaction_id_input.isdigit():
                    logging.error("Invalid interaction ID. It must be a positive integer.")
                    continue
                interaction_id = int(interaction_id_input)
                await verification.mark_interaction_verified(conversation_id, interaction_id)
                print(f"Interaction {interaction_id} marked as verified in conversation_id {conversation_id}.")

            elif choice == "6":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                history = await verification.get_conversation_history(conversation_id)
                if history:
                    print("Conversation History:")
                    for i, interaction in enumerate(history, 1):
                        print(f"{i}. {interaction}")
                else:
                    print("No history found.")

            elif choice == "7":
                limit_str = input("Enter the limit for conversation IDs to retrieve (default is 100): ").strip()
                limit = int(limit_str) if limit_str.isdigit() else 100
                ids_list = await verification.get_all_conversation_ids(limit)
                if ids_list:
                    print("Available Conversation UUIDs:")
                    for cid in ids_list:
                        print(cid)
                else:
                    print("No conversations found.")

            elif choice == "8":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                await verification.delete_conversation(conversation_id)
                print(f"Conversation {conversation_id} deleted.")

            elif choice == "9":
                username = input("Enter the username to list conversation UUIDs for: ").strip()
                limit_str = input("Enter the limit (default is 100): ").strip()
                limit = int(limit_str) if limit_str.isdigit() else 100
                user_conversations = await verification.get_all_conversation_ids_for_username(username, limit)
                if user_conversations:
                    print(f"Conversations for user '{username}':")
                    for cid in user_conversations:
                        print(cid)
                else:
                    print(f"No conversations found for username '{username}'.")

            elif choice == "10":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

        except Exception as e:
            logging.error(f"An error occurred during operation: {e}")


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
        try:
            await repository.bulk_write(TEST_COLLECTION_NAME, operations)
            logging.debug(f"Bulk write operations from index {start_index} to {start_index + 99} completed.")
        except AttributeError as e:
            logging.error(f"Bulk write operation failed due to missing attribute: {e}")
        except Exception as e:
            logging.error(f"Bulk write operation failed: {e}")

    async def clear_cache_test():
        """Clear the repository cache."""
        await repository.clear_cache()

    # Execute Tests in Sequence with Performance Logging

    # 1. Insert Users Concurrently
    logging.info(f"Starting insert operations for {num_operations} users...")
    start_time = time.time()
    insert_tasks = [insert_test_user(i) for i in range(num_operations)]
    inserted_ids = await asyncio.gather(*insert_tasks, return_exceptions=True)
    # Filter out exceptions and keep only successful inserts
    inserted_ids = [doc_id for doc_id in inserted_ids if isinstance(doc_id, ObjectId)]
    insert_duration = time.time() - start_time
    await repository.log_performance("insert", insert_duration, num_operations)
    logging.info(f"Insert operations completed in {insert_duration:.2f} seconds.")

    # 2. Find Users Concurrently
    logging.info(f"Starting find operations for {num_operations} users...")
    start_time = time.time()
    find_tasks = [find_test_user(i) for i in range(num_operations)]
    find_results = await asyncio.gather(*find_tasks, return_exceptions=True)
    find_duration = time.time() - start_time
    await repository.log_performance("find", find_duration, num_operations)
    logging.info(f"Find operations completed in {find_duration:.2f} seconds.")

    # 3. Update Users Concurrently
    logging.info(f"Starting update operations for {num_operations} users...")
    start_time = time.time()
    update_tasks = [update_test_user(i) for i in range(num_operations)]
    update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
    update_duration = time.time() - start_time
    await repository.log_performance("update", update_duration, num_operations)
    logging.info(f"Update operations completed in {update_duration:.2f} seconds.")

    # 4. Fetch Embeddings Concurrently
    logging.info(f"Starting fetch_embedding operations for {num_operations} users...")
    start_time = time.time()
    fetch_tasks = [fetch_embedding_test(doc_id) for doc_id in inserted_ids]
    embeddings = await asyncio.gather(*fetch_tasks, return_exceptions=True)
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
    await asyncio.gather(*save_tasks, return_exceptions=True)
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
    await asyncio.gather(*bulk_tasks, return_exceptions=True)
    bulk_duration = time.time() - start_time
    await repository.log_performance("bulk_write", bulk_duration,
                                     num_operations * 2)  # Each bulk_write handles 2 operations per iteration
    logging.info(f"Bulk_write operations completed in {bulk_duration:.2f} seconds.")

    # 8. Delete Users Concurrently
    logging.info(f"Starting delete operations for {num_operations} users...")
    start_time = time.time()
    delete_tasks = [delete_test_user(document_id) for document_id in inserted_ids]
    await asyncio.gather(*delete_tasks, return_exceptions=True)
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
    async def run_main():
        repository = ZMongoRepository()
        await repository.clear_cache()
        try:
            # Run high-load test with 5,000 concurrent operations
            await high_load_test(repository, num_operations=5000)
        except Exception as e:
            logging.error(f"Error during high-load test: {e}")
        finally:
            await repository.close()

    asyncio.run(run_main())



# Performance Tests and High-Load Testing


async def test_update_document_cache():
    repository = ZMongoRepository()

    # Insert a test document
    conversation_id = str(uuid.uuid4())
    document = {
        "conversation_id": conversation_id,
        "username": "testuser",
        "created_at": datetime.now(),
        "interactions": [
            {"ai_question": "Q1", "user_response": "A1", "verified": False, "timestamp": datetime.now()},
            {"ai_question": "Q2", "user_response": "A2", "verified": False, "timestamp": datetime.now()},
        ],
    }
    await repository.insert_document("conversations", document)

    # Ensure it's cached
    cached_doc = await repository.find_document("conversations", {"conversation_id": conversation_id})
    assert cached_doc is not None
    assert len(cached_doc["interactions"]) == 2

    # Update an interaction
    update_data = {
        "$set": {
            "interactions.1.user_response": "Updated Answer",
            "interactions.1.verified": True
        }
    }
    success = await repository.update_document(
        collection="conversations",
        update_data=update_data,
        query={"conversation_id": conversation_id},
        upsert=False
    )
    assert success

    # Retrieve from cache and verify
    updated_cached_doc = repository.cache["conversations"][
        repository._generate_cache_key(json.dumps({"conversation_id": conversation_id}, sort_keys=True, default=str))]
    assert updated_cached_doc["interactions"][1]["user_response"] == "Updated Answer"
    assert updated_cached_doc["interactions"][1]["verified"] == True

    # Cleanup
    await repository.delete_document("conversations", {"conversation_id": conversation_id})
    await repository.close()


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
        test_result = test_update_document_cache()
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
