import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import the necessary classes from zmongo_service.py
from zmongo_service import ZMongoService, SafeResult, Document

# Apply the nest_asyncio patch for running nested event loops
import nest_asyncio

nest_asyncio.apply()
load_dotenv(Path.home() / "resources" / ".env_local")


async def main():
    """
    This script demonstrates how to use the ZMongoService class to perform
    a basic add and search operation on a MongoDB collection.
    """

    # Load environment variables from a .env file
    # This assumes MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY are set
    # in a file at the user's home directory under "resources/.env_local"
    # as specified in the original zmongo_service.py file.
    # Get environment variables
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Check if required environment variables are set
    if not all([MONGO_URI, MONGO_DATABASE_NAME, GEMINI_API_KEY]):
        print("Please set MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY in your .env file.")
        return

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the ZMongoService instance
    service = ZMongoService(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DATABASE_NAME,
        gemini_api_key=GEMINI_API_KEY
    )

    # Define the collection and the field containing the text to embed
    collection = "my_knowledge_base"
    content_field = "content"

    try:
        # --- Add a document ---
        print("\n--- Attempting to add a document ---")
        doc_to_add = {
            "title": "Pasta Pizza and More",
            "content": "Gino' Restaurant is cooking a giant pizza tomorrow evening."
        }
        res: SafeResult = await service.add_and_embed(collection, doc_to_add, text_field=content_field)

        if res.success:
            print(f"Successfully added document with ID: {res.data['inserted_id']}")
        else:
            print(f"Failed to add document: {res.error}")

        # --- Perform a semantic search ---
        print("\n--- Performing a search ---")
        query = "Where can a take my wife to eat?"

        # Set the minimum similarity score percentage (e.g., 70%)
        min_score_threshold = 0.70
        print(f"Searching for documents with a score of {min_score_threshold * 100}% or higher.")

        # The search method uses the retriever to perform a vector search
        search_results: list[Document] = await service.search(
            collection,
            query,
            content_field=content_field,
            similarity_threshold=min_score_threshold
        )

        print(f"\nFound {len(search_results)} results for the query: '{query}'")
        for doc in search_results:
            print(f"\n  Score: {doc.metadata.get('retrieval_score'):.4f}")
            print(f"  Content: {doc.page_content}")
            print(f"  Source Title: {doc.metadata.get('title')}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Close the connection when done
        print("\n--- Closing Connection ---")
        await service.close_connection()


if __name__ == "__main__":
    asyncio.run(main())

