import asyncio
import logging
import os
import time
import uuid
from bson import ObjectId
from dotenv import load_dotenv

from zmongo.BAK.zmongo_repository import ZMongoRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
TEST_COLLECTION_NAME = os.getenv("TEST_COLLECTION_NAME")


# --- Minimal Stub for LlamaInteractiveVerification ---
class LlamaInteractiveVerification:
    """
    A minimal in‑memory implementation of an interactive verification module.
    This stub supports creating a conversation, saving/editing interactions,
    clearing conversation history, and retrieving conversation IDs.
    """
    def __init__(self):
        self.conversations = {}  # conversation_id -> {username, interactions}

    async def create_new_conversation(self, username: str) -> str:
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = {"username": username, "interactions": []}
        logger.info(f"Created conversation {conv_id} for user '{username}'.")
        return conv_id

    async def save_interaction(self, conversation_id: str, ai_question: str, user_response: str, verified: bool):
        if conversation_id not in self.conversations:
            raise Exception("Conversation not found")
        conv = self.conversations[conversation_id]
        interaction = {
            "id": len(conv["interactions"]) + 1,
            "ai_question": ai_question,
            "user_response": user_response,
            "verified": verified
        }
        conv["interactions"].append(interaction)
        logger.info(f"Saved interaction #{interaction['id']} in conversation {conversation_id}.")

    async def edit_interaction(self, conversation_id: str, interaction_id: int, new_data: dict):
        if conversation_id not in self.conversations:
            raise Exception("Conversation not found")
        interactions = self.conversations[conversation_id]["interactions"]
        if interaction_id < 1 or interaction_id > len(interactions):
            raise Exception("Interaction not found")
        interactions[interaction_id - 1].update(new_data)
        logger.info(f"Edited interaction #{interaction_id} in conversation {conversation_id}.")

    async def clear_conversation_history(self, conversation_id: str):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["interactions"] = []
            logger.info(f"Cleared conversation history for {conversation_id}.")

    async def mark_interaction_verified(self, conversation_id: str, interaction_id: int):
        await self.edit_interaction(conversation_id, interaction_id, {"verified": True})
        logger.info(f"Marked interaction #{interaction_id} as verified in conversation {conversation_id}.")

    async def get_conversation_history(self, conversation_id: str):
        return self.conversations.get(conversation_id, {}).get("interactions", [])

    async def get_all_conversation_ids(self, limit: int = 100):
        return list(self.conversations.keys())[:limit]

    async def delete_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation {conversation_id}.")

    async def get_all_conversation_ids_for_username(self, username: str, limit: int = 100):
        result = [cid for cid, conv in self.conversations.items() if conv.get("username") == username]
        return result[:limit]


# --- High-Load Test Functions for ZMongoRepository ---
async def high_load_test(repository: ZMongoRepository, num_operations=1000):
    """
    Perform a high-load test on the ZMongoRepository by running concurrent operations.
    """
    semaphore = asyncio.Semaphore(1000)  # Limit concurrency

    inserted_ids = []

    async def insert_test_user(i):
        async with semaphore:
            document = {"name": f"Test User {i}", "age": 20 + i, "creator": "admin"}
            result = await repository.insert_document(
                collection=TEST_COLLECTION_NAME, document=document
            )
            return result.inserted_id

    async def find_test_user(i):
        async with semaphore:
            query = {"name": f"Test User {i}"}
            document = await repository.find_document(
                collection=TEST_COLLECTION_NAME, query=query
            )
            return document

    async def update_test_user(i):
        async with semaphore:
            query = {"name": f"Test User {i}"}
            update_data = {"$set": {"age": 30 + i}}
            result = await repository.update_document(
                collection=TEST_COLLECTION_NAME, query=query, update_data=update_data
            )
            return result

    async def delete_test_user(document_id):
        async with semaphore:
            query = {"_id": document_id}
            result = await repository.delete_document(
                collection=TEST_COLLECTION_NAME, query=query
            )
            return result

    async def fetch_embedding_test(document_id):
        async with semaphore:
            embedding = await repository.fetch_embedding(
                collection=TEST_COLLECTION_NAME, document_id=document_id
            )
            return embedding

    async def save_embedding_test(document_id, embedding):
        async with semaphore:
            await repository.save_embedding(
                collection=TEST_COLLECTION_NAME, document_id=document_id, embedding=embedding
            )

    async def aggregate_test():
        pipeline = [
            {"$match": {"creator": "admin"}},
            {"$group": {"_id": "$creator", "average_age": {"$avg": "$age"}}}
        ]
        result = await repository.aggregate_documents(
            collection=TEST_COLLECTION_NAME, pipeline=pipeline
        )
        return result

    async def bulk_write_test(start_index):
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
        except Exception as e:
            logger.error(f"Bulk write operation failed: {e}")

    async def clear_cache_test():
        await repository.clear_cache()

    logger.info(f"Starting insert operations for {num_operations} users...")
    start_time = time.time()
    insert_tasks = [insert_test_user(i) for i in range(num_operations)]
    inserted_ids = await asyncio.gather(*insert_tasks, return_exceptions=True)
    inserted_ids = [doc_id for doc_id in inserted_ids if isinstance(doc_id, ObjectId)]
    insert_duration = time.time() - start_time
    await repository.log_performance("insert", insert_duration, num_operations)
    logger.info(f"Insert operations completed in {insert_duration:.2f} seconds.")

    logger.info(f"Starting find operations for {num_operations} users...")
    start_time = time.time()
    find_tasks = [find_test_user(i) for i in range(num_operations)]
    find_results = await asyncio.gather(*find_tasks, return_exceptions=True)
    find_duration = time.time() - start_time
    await repository.log_performance("find", find_duration, num_operations)
    logger.info(f"Find operations completed in {find_duration:.2f} seconds.")

    logger.info(f"Starting update operations for {num_operations} users...")
    start_time = time.time()
    update_tasks = [update_test_user(i) for i in range(num_operations)]
    update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
    update_duration = time.time() - start_time
    await repository.log_performance("update", update_duration, num_operations)
    logger.info(f"Update operations completed in {update_duration:.2f} seconds.")

    logger.info(f"Starting fetch_embedding operations for {num_operations} users...")
    start_time = time.time()
    fetch_tasks = [fetch_embedding_test(doc_id) for doc_id in inserted_ids]
    embeddings = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    fetch_duration = time.time() - start_time
    await repository.log_performance("fetch_embedding", fetch_duration, num_operations)
    logger.info(f"Fetch_embedding operations completed in {fetch_duration:.2f} seconds.")

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

    logger.info("Starting aggregation operations...")
    start_time = time.time()
    aggregate_results = await aggregate_test()
    aggregate_duration = time.time() - start_time
    await repository.log_performance("aggregate", aggregate_duration, 1)
    logger.info(f"Aggregation operation completed in {aggregate_duration:.2f} seconds. Result: {aggregate_results}")

    bulk_batches = num_operations // 100
    logger.info(f"Starting bulk_write operations ({bulk_batches} batches)...")
    start_time = time.time()
    bulk_tasks = [bulk_write_test(i * 100) for i in range(bulk_batches)]
    await asyncio.gather(*bulk_tasks, return_exceptions=True)
    bulk_duration = time.time() - start_time
    await repository.log_performance("bulk_write", bulk_duration, num_operations * 2)
    logger.info(f"Bulk_write operations completed in {bulk_duration:.2f} seconds.")

    logger.info(f"Starting delete operations for {num_operations} users...")
    start_time = time.time()
    delete_tasks = [delete_test_user(document_id) for document_id in inserted_ids]
    await asyncio.gather(*delete_tasks, return_exceptions=True)
    delete_duration = time.time() - start_time
    await repository.log_performance("delete", delete_duration, num_operations)
    logger.info(f"Delete operations completed in {delete_duration:.2f} seconds.")

    logger.info("Starting cache clearing operation...")
    start_time = time.time()
    await clear_cache_test()
    clear_cache_duration = time.time() - start_time
    await repository.log_performance("clear_cache", clear_cache_duration, 1)
    logger.info(f"Cache cleared in {clear_cache_duration:.2f} seconds.")

    logger.info("Verifying cache is empty...")
    is_cache_empty = all(not cache for cache in repository.cache.values())
    if is_cache_empty:
        logger.info("Cache verification successful: Cache is empty.")
    else:
        logger.warning("Cache verification failed: Cache is not empty.")


async def test_bulk_write(num_bulk_write):
    repository = ZMongoRepository()
    test_collection = TEST_COLLECTION_NAME
    if not test_collection:
        logger.error("TEST_COLLECTION_NAME environment variable is not set.")
        return

    await repository.db[test_collection].delete_many({})
    logger.info(f"Cleared all documents from collection '{test_collection}'.")

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

    try:
        await repository.bulk_write(test_collection, operations)
        logger.info(f"Bulk write operations for collection '{test_collection}' completed successfully.")
    except Exception as e:
        logger.error(f"Bulk write operation failed: {e}")
    finally:
        await repository.close()


async def interactive_cli():
    """
    A simple interactive CLI that demonstrates conversation verification functions.
    """
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
                print(f"New conversation created. ID: {conversation_id}, User: '{username}'")
            elif choice == "2":
                conversation_id = input("Enter the conversation ID: ").strip()
                ai_question = input("Enter the AI question: ").strip()
                user_response = input("Enter the user response: ").strip()
                verified = input("Is the response verified? (yes/no): ").strip().lower() == "yes"
                await verification.save_interaction(conversation_id, ai_question, user_response, verified)
                print(f"Interaction added to conversation {conversation_id}.")
            elif choice == "3":
                conversation_id = input("Enter the conversation ID: ").strip()
                interaction_id = int(input("Enter the interaction ID (1-based): "))
                print("Enter new fields (leave blank to skip):")
                ai_question = input("New AI question: ").strip()
                user_response = input("New user response: ").strip()
                verified_str = input("Verified? (yes/no): ").strip().lower()
                new_data = {}
                if ai_question:
                    new_data["ai_question"] = ai_question
                if user_response:
                    new_data["user_response"] = user_response
                if verified_str in {"yes", "no"}:
                    new_data["verified"] = (verified_str == "yes")
                await verification.edit_interaction(conversation_id, interaction_id, new_data)
            elif choice == "4":
                conversation_id = input("Enter the conversation ID: ").strip()
                await verification.clear_conversation_history(conversation_id)
            elif choice == "5":
                conversation_id = input("Enter the conversation ID: ").strip()
                interaction_id = int(input("Enter the interaction ID (1-based): "))
                await verification.mark_interaction_verified(conversation_id, interaction_id)
            elif choice == "6":
                conversation_id = input("Enter the conversation ID: ").strip()
                history = await verification.get_conversation_history(conversation_id)
                print("Conversation History:")
                for i, interaction in enumerate(history, 1):
                    print(f"{i}. {interaction}")
            elif choice == "7":
                limit_str = input("Enter the limit (default 100): ").strip()
                limit = int(limit_str) if limit_str.isdigit() else 100
                ids_list = await verification.get_all_conversation_ids(limit)
                print("Available Conversation IDs:")
                for cid in ids_list:
                    print(cid)
            elif choice == "8":
                conversation_id = input("Enter the conversation ID: ").strip()
                await verification.delete_conversation(conversation_id)
                print(f"Conversation {conversation_id} deleted.")
            elif choice == "9":
                username = input("Enter username: ").strip()
                limit_str = input("Enter limit (default 100): ").strip()
                limit = int(limit_str) if limit_str.isdigit() else 100
                conv_ids = await verification.get_all_conversation_ids_for_username(username, limit)
                print(f"Conversations for '{username}':")
                for cid in conv_ids:
                    print(cid)
            elif choice == "10":
                print("Exiting interactive CLI.")
                break
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            logger.error(f"Error during interactive CLI: {e}")


async def run_all_tests():
    """
    Runs repository tests sequentially.
    """
    repository = ZMongoRepository()
    try:
        logger.info("Clearing collection 'user'.")
        await repository.db['user'].delete_many({})
        logger.info("Collection 'user' cleared.")

        logger.info("Performing bulk write operations on 'user'.")
        operations = (
            [{"action": "insert", "document": {"name": f"User{i}", "age": 20 + i}} for i in range(5)]
            +
            [{"action": "update", "filter": {"name": f"User{i}"}, "update": {"$set": {"age": 30 + i}}} for i in range(5)]
        )
        await repository.bulk_write('user', operations)
        logger.info("Bulk write on 'user' completed.")

        logger.info("Clearing cache.")
        await repository.clear_cache()
        logger.info("Cache cleared.")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        logger.info("Closing repository connections.")
        await repository.close()


if __name__ == "__main__":
    # Run repository high-load test
    this_zmongo_repository = ZMongoRepository()
    asyncio.run(high_load_test(this_zmongo_repository))
    asyncio.run(test_bulk_write(5000))
    # Run repository tests sequentially
    asyncio.run(run_all_tests())
    # Start the interactive CLI for conversation verification
    asyncio.run(interactive_cli())
