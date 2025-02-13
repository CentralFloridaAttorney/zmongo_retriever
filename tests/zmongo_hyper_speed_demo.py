# zmongo_hyper_speed_demo.py

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from zmongo.BAK.zmongo_repository import ZMongoRepository
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

CACHE_EXPIRATION_SECONDS = os.getenv('CACHE_EXPIRATION_SECONDS')
if CACHE_EXPIRATION_SECONDS is not None:
    try:
        CACHE_EXPIRATION_SECONDS = int(CACHE_EXPIRATION_SECONDS)
    except ValueError:
        raise ValueError("CACHE_EXPIRATION_SECONDS must be an integer.")
else:
    CACHE_EXPIRATION_SECONDS = 300  # Default to 5 minutes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def main():
    # Initialize the Mongo Repository
    zmongo = ZMongoRepository()
    # await zmongo.initialize()

    # Example: Insert a document
    document = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "created_at": datetime.utcnow()
    }
    insert_result = await zmongo.insert_document("users", document)
    print(f"Inserted document ID: {insert_result.inserted_id}")

    # Example: Find a document
    found_doc = await zmongo.find_document("users", {"_id": insert_result.inserted_id})
    print(f"Found document: {found_doc}")

    # Example: Update a document
    update_success = await zmongo.update_document(
        collection="users",
        update_data={"$set": {"email": "john.new@example.com"}},
        query={"_id": insert_result.inserted_id},
        upsert=False
    )
    print(f"Update successful: {update_success}")

    # Example: Find documents
    users = await zmongo.find_documents("users", {"name": "John Doe"})
    print(f"Users found: {users}")

    # Example: Delete a document
    delete_result = await zmongo.delete_document("users", {"_id": insert_result.inserted_id})
    print(f"Documents deleted: {delete_result.deleted_count}")

    # Close connections
    await zmongo.close()

if __name__ == "__main__":
    asyncio.run(main())
