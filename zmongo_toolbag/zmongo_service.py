import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from bson.objectid import ObjectId
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
from langchain.schema import Document
import motor.motor_asyncio

# We keep SafeResult imported here as it's a fundamental class used across the module.
from zmongo_toolbag.safe_result import SafeResult
# The following imports have been moved inside the methods to break the circular dependency.
# from zmongo_toolbag.zmongo_retriever import ZMongoRetriever
# from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
# from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder


# User's provided path for environment variables
load_dotenv(Path.home() / "resources" / ".env_local")



class ZMongoService:
    """
    A high-level service class that provides a simplified interface for
    interacting with MongoDB, handling text embeddings, and performing
    semantic searches.
    """

    def __init__(self, mongo_uri: str, db_name: str, gemini_api_key: str):
        # We import the classes inside the method to break the circular dependency.
        from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
        from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = self.client[db_name]
            self.repository = ZMongoAtlas(self.db)
            self.embedder = ZMongoEmbedder(repository=self.repository, collection="_embedding_cache", page_content_key="content", gemini_api_key=gemini_api_key)
            self.gemini_api_key = gemini_api_key
            logging.info(f"ZMongoService initialized for database '{db_name}'.")
        except Exception as e:
            logging.error(f"Failed to initialize ZMongoService: {e}")
            raise

    def get_retriever(self, collection_name: str, **kwargs):
        # The retriever is also imported and instantiated here.
        from zmongo_toolbag.zmongo_retriever import ZMongoRetriever
        return ZMongoRetriever(
            repository=self.repository,
            embedder=self.embedder,
            collection_name=collection_name,
            **kwargs
        )

    async def add_and_embed(
            self,
            collection_name: str,
            document: Dict[str, Any],
            text_field: str,
            embedding_field: str = "embeddings"
    ) -> SafeResult:
        """
        Checks for duplicate content before inserting. If the content is unique,
        it inserts the document and adds embeddings.
        """
        text_to_embed = document.get(text_field)
        if not text_to_embed or not isinstance(text_to_embed, str):
            return SafeResult.fail(f"Document must contain a non-empty string in the '{text_field}' field.")

        existing_doc_res = await self.repository.find_document(collection_name, {text_field: text_to_embed})
        if existing_doc_res.success and existing_doc_res.data:
            existing_id = existing_doc_res.data.get('_id')
            logging.warning(f"Document with the same content already exists with ID {existing_id}. Skipping.")
            return SafeResult.ok(
                {"message": "Document already exists.", "inserted_id": str(existing_id), "existed": True})

        try:
            insert_res = await self.repository.insert_document(collection_name, document)
        except DuplicateKeyError:
            return await self.add_and_embed(collection_name, document, text_field, embedding_field)

        if not insert_res.success:
            return insert_res

        doc_id_from_insert = insert_res.data.inserted_id
        if not doc_id_from_insert:
            return SafeResult.fail("Failed to retrieve document ID after insertion.")

        try:
            doc_id_obj = ObjectId(doc_id_from_insert)
        except Exception:
            return SafeResult.fail(f"Invalid document ID format returned from insert: {doc_id_from_insert}")

        try:
            embeddings = await self.embedder.embed_text(text_to_embed)
            if not embeddings:
                await self.repository.delete_document(collection_name, {"_id": doc_id_obj})
                return SafeResult.fail("Embedding generation returned no vectors.")
        except Exception as e:
            logging.error(f"Embedding generation failed for doc_id {doc_id_obj}: {e}")
            await self.repository.delete_document(collection_name, {"_id": doc_id_obj})
            return SafeResult.fail(f"An exception occurred during embedding generation: {e}")

        update_res = await self.repository.update_document(
            collection_name,
            {"_id": doc_id_obj},
            {"$set": {embedding_field: embeddings}}
        )

        # --- THIS IS THE CORRECTED LINE ---
        if not update_res.success or update_res.data.modified_count == 0:
            logging.error(f"Failed to embed document with ID {doc_id_obj}.")
            error_msg = update_res.error or f"Failed to save embeddings for document ID {doc_id_obj}."
            return SafeResult.fail(error_msg)

        logging.info(f"Successfully inserted and embedded document {doc_id_obj}.")
        return SafeResult.ok({"inserted_id": str(doc_id_obj), "existed": False})

    async def search(self, collection_name: str, query_text: str, **kwargs) -> List[Document]:
        retriever = self.get_retriever(collection_name, **kwargs)
        return await retriever._get_relevant_documents(query_text)

    async def close_connection(self):
        if hasattr(self, 'repository') and hasattr(self.repository, 'close'):
            await self.repository.close()


async def main():
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not all([MONGO_URI, MONGO_DATABASE_NAME, GEMINI_API_KEY]):
        print("Please set MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY in your .env file.")
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = ZMongoService(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DATABASE_NAME,
        gemini_api_key=GEMINI_API_KEY
    )

    collection = "my_knowledge_base"
    content_field = "content"

    try:
        print("\n--- Attempting to add a document ---")
        doc_to_add = {"title": "Python Basics",
                      "content": "Python is a versatile and popular programming language known for its simplicity."}
        res = await service.add_and_embed(collection, doc_to_add, text_field=content_field)
        if res.success:
            if res.data.get("existed"):
                print(f"Document already existed with ID: {res.data['inserted_id']}")
            else:
                print(f"Successfully added new document with ID: {res.data['inserted_id']}")
        else:
            print(f"Failed to add document: {res.error}")

        print("\n--- Attempting to add the same document again ---")
        res_again = await service.add_and_embed(collection, doc_to_add, text_field=content_field)
        if res_again.success:
            if res_again.data.get("existed"):
                print(f"Document already existed with ID: {res_again.data['inserted_id']}")
            else:
                print(f"Successfully added new document with ID: {res_again.data['inserted_id']}")
        else:
            print(f"Failed to add document: {res_again.error}")

        print("\n--- Performing Search ---")
        query = "What is a good language for beginners?"

        search_results = await service.search(
            collection,
            query,
            content_field=content_field
        )

        print(f"\nFound {len(search_results)} results for the query: '{query}'")
        for doc in search_results:
            print(f"\n  Score: {doc.metadata.get('retrieval_score'):.4f}")
            print(f"  Content: {doc.page_content}")
            print(f"  Source Title: {doc.metadata.get('title')}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n--- Closing Connection ---")
        await service.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
