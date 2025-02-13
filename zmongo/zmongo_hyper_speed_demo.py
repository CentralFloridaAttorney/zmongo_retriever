# zmongo_hyper_speed_demo.py

import logging
import os
import time
from bson import ObjectId
from dotenv import load_dotenv

from zmongo.demos.llama_interactive_verification import LlamaInteractiveVerification
from zmongo.zmongo_repository import ZMongoRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
TEST_COLLECTION_NAME = os.getenv("TEST_COLLECTION_NAME")


async def high_load_test(repository: ZMongoRepository, num_operations=1000):
    """
    Perform a high-load test on the ZMongoHyperSpeed by running concurrent operations.
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
            operations.append({
                "action": "insert",
                "document": {"name": f"Bulk User {i}", "age": 25 + i}
            })
            operations.append({
                "action": "update",
                "filter": {"name": f"Bulk User {i}"},
                "update": {"$set": {"age": 35 + i}},
                "upsert": True
            })
        try:
            await repository.bulk_write(TEST_COLLECTION_NAME, operations)
            logger.debug(f"Bulk write operations from index {start_index} to {start_index + 99} completed.")
        except AttributeError as e:
            logger.error(f"Bulk write operation failed due to missing attribute: {e}")
        except Exception as e:
            logger.error(f"Bulk write operation failed: {e}")

    async def clear_cache_test():
        """Clear the repository cache."""
        await repository.clear_cache()

    # Execute Tests in Sequence with Performance Logging

    # 1. Insert Users Concurrently
    logger.info(f"Starting insert operations for {num_operations} users...")
    start_time = time.time()
    insert_tasks = [insert_test_user(i) for i in range(num_operations)]
    inserted_ids = await asyncio.gather(*insert_tasks, return_exceptions=True)
    # Filter out exceptions and keep only successful inserts
    inserted_ids = [doc_id for doc_id in inserted_ids if isinstance(doc_id, ObjectId)]
    insert_duration = time.time() - start_time
    await repository.log_performance("insert", insert_duration, num_operations)
    logger.info(f"Insert operations completed in {insert_duration:.2f} seconds.")

    # 2. Find Users Concurrently
    logger.info(f"Starting find operations for {num_operations} users...")
    start_time = time.time()
    find_tasks = [find_test_user(i) for i in range(num_operations)]
    find_results = await asyncio.gather(*find_tasks, return_exceptions=True)
    find_duration = time.time() - start_time
    await repository.log_performance("find", find_duration, num_operations)
    logger.info(f"Find operations completed in {find_duration:.2f} seconds.")

    # 3. Update Users Concurrently
    logger.info(f"Starting update operations for {num_operations} users...")
    start_time = time.time()
    update_tasks = [update_test_user(i) for i in range(num_operations)]
    update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
    update_duration = time.time() - start_time
    await repository.log_performance("update", update_duration, num_operations)
    logger.info(f"Update operations completed in {update_duration:.2f} seconds.")

    # 4. Fetch Embeddings Concurrently
    logger.info(f"Starting fetch_embedding operations for {num_operations} users...")
    start_time = time.time()
    fetch_tasks = [fetch_embedding_test(doc_id) for doc_id in inserted_ids]
    embeddings = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    fetch_duration = time.time() - start_time
    await repository.log_performance("fetch_embedding", fetch_duration, num_operations)
    logger.info(f"Fetch_embedding operations completed in {fetch_duration:.2f} seconds.")

    # 5. Save Embeddings Concurrently
    logger.info(f"Starting save_embedding operations for {num_operations} users...")
    start_time = time.time()
    save_tasks = [
        save_embedding_test(doc_id, [0.1, 0.2, 0.3, 0.4, 0.5])
        for doc_id in inserted_ids
    ]
    await asyncio.gather(*save_tasks, return_exceptions=True)
    save_duration = time.time() - start_time
    await repository.log_performance("save_embedding", save_duration, num_operations)
    logger.info(f"Save_embedding operations completed in {save_duration:.2f} seconds.")

    # 6. Aggregate Documents
    logger.info("Starting aggregation operations...")
    start_time = time.time()
    aggregate_results = await aggregate_test()
    aggregate_duration = time.time() - start_time
    await repository.log_performance("aggregate", aggregate_duration, 1)
    logger.info(f"Aggregation operation completed in {aggregate_duration:.2f} seconds. Result: {aggregate_results}")

    # 7. Bulk Write Operations
    bulk_batches = num_operations // 100  # 100 operations per bulk_write
    logger.info(f"Starting bulk_write operations ({bulk_batches} batches)...")
    start_time = time.time()
    bulk_tasks = [bulk_write_test(i * 100) for i in range(bulk_batches)]
    await asyncio.gather(*bulk_tasks, return_exceptions=True)
    bulk_duration = time.time() - start_time
    await repository.log_performance("bulk_write", bulk_duration, num_operations * 2)  # Each bulk_write handles 2 operations per iteration
    logger.info(f"Bulk_write operations completed in {bulk_duration:.2f} seconds.")

    # 8. Delete Users Concurrently
    logger.info(f"Starting delete operations for {num_operations} users...")
    start_time = time.time()
    delete_tasks = [delete_test_user(document_id) for document_id in inserted_ids]
    await asyncio.gather(*delete_tasks, return_exceptions=True)
    delete_duration = time.time() - start_time
    await repository.log_performance("delete", delete_duration, num_operations)
    logger.info(f"Delete operations completed in {delete_duration:.2f} seconds.")

    # 9. Clear Cache
    logger.info("Starting cache clearing operation...")
    start_time = time.time()
    await clear_cache_test()
    clear_cache_duration = time.time() - start_time
    await repository.log_performance("clear_cache", clear_cache_duration, 1)
    logger.info(f"Cache cleared in {clear_cache_duration:.2f} seconds.")

    # 10. Verify Cache is Empty
    logger.info("Verifying cache is empty...")
    # Since Redis does not have a direct method to check if cache is empty, we'll perform a simple dbsize
    try:
        cache_size = await repository.redis.dbsize()
        if cache_size == 0:
            logger.info("Cache verification successful: Cache is empty.")
        else:
            logger.warning(f"Cache verification failed: {cache_size} keys found in cache.")
    except Exception as e:
        logger.error(f"Failed to verify cache status: {e}")


async def test_bulk_write(num_bulk_write):
    repository = ZMongoHyperSpeed()
    test_collection = TEST_COLLECTION_NAME
    if not test_collection:
        logger.error("TEST_COLLECTION_NAME environment variable is not set.")
        return

    try:
        # **1. Initialize Redis Client**
        await repository.initialize()

        # **2. Clear the Collection**
        logger.info(f"Clearing all documents from collection '{test_collection}'.")
        await repository.db[test_collection].delete_many({})
        logger.info(f"All documents cleared from collection '{test_collection}'.")

        # **3. Prepare Bulk Operations**
        operations = []
        for i in range(num_bulk_write):
            operations.append({
                "action": "insert",
                "document": {"name": f"Test User {i}", "age": 25 + i}
            })
            operations.append({
                "action": "update",
                "filter": {"name": f"Test User {i}"},
                "update": {"$set": {"age": 35 + i}},
                "upsert": True
            })

        # **4. Perform Bulk Write**
        logger.info(f"Performing bulk write operations for collection '{test_collection}'.")
        await repository.bulk_write(test_collection, operations)
        logger.info(f"Bulk write operations for collection '{test_collection}' completed successfully.")

        # **5. Log Performance**
        duration = 0.0  # Placeholder, implement timing if needed
        await repository.log_performance("bulk_write_test", duration, num_bulk_write * 2)
    except Exception as e:
        logger.error(f"Bulk write operation failed: {e}")
    finally:
        # **6. Close Connections**
        await repository.close()


async def interactive_cli():
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
                verified = input("Is the response verified? (yes/no): ").lower() == "yes"
                await verification.save_interaction(conversation_id, ai_question, user_response, verified)
                print(f"Interaction added to conversation_id {conversation_id}.")

            elif choice == "3":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                interaction_id = int(input("Enter the interaction ID (1-based): "))
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

                await verification.edit_interaction(conversation_id, interaction_id, new_data)

            elif choice == "4":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                await verification.clear_conversation_history(conversation_id)

            elif choice == "5":
                conversation_id = input("Enter the conversation_id (UUID): ").strip()
                interaction_id = int(input("Enter the interaction ID (1-based): "))
                await verification.mark_interaction_verified(conversation_id, interaction_id)

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
            logger.error(f"An error occurred during operation: {e}")


# zmongo_repository_main.py

import asyncio
import logging
from zmongo_hyper_speed import ZMongoHyperSpeed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_all_tests():
    """
    Runs all tests sequentially:
    1. Initialize repository
    2. Perform bulk write operations
    3. Clear cache
    4. Close connections
    """
    repository = ZMongoHyperSpeed()

    try:
        # **1. Initialize Redis Client**
        await repository.initialize()

        # **2. Clear 'user' Collection (for testing purposes)**
        logger.info("Clearing all documents from collection 'user'.")
        await repository.db['user'].delete_many({})
        logger.info("All documents cleared from collection 'user'.")

        # **3. Perform Bulk Write Operations**
        logger.info("Performing bulk write operations on collection 'user'.")
        operations = [
            {"action": "insert", "document": {"name": f"User{i}", "age": 20 + i}} for i in range(5)
        ] + [
            {"action": "update", "filter": {"name": f"User{i}"}, "update": {"$set": {"age": 30 + i}}} for i in range(5)
        ]
        await repository.bulk_write('user', operations)
        logger.info("Bulk write operations completed successfully.")

        # **4. Clear Cache**
        logger.info("Clearing Redis cache.")
        await repository.clear_cache()
        logger.info("Redis cache cleared successfully.")

    except Exception as e:
        logger.error(f"Bulk write operation failed: {e}")
    finally:
        # **5. Close Connections**
        logger.info("Closing connections.")
        await repository.close()


async def main():
    # Run initial tests
    await run_all_tests()

    # Create a new repository instance for high load test
    high_load_repository = ZMongoHyperSpeed()
    await high_load_repository.initialize()
    await high_load_test(high_load_repository)

    # Create another repository instance for bulk write test
    bulk_write_repository = ZMongoHyperSpeed()
    await bulk_write_repository.initialize()
    await test_bulk_write(5000)


if __name__ == "__main__":
    asyncio.run(main())
