import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from bson import ObjectId
from dotenv import load_dotenv
from langchain.schema import Document

# Assuming your toolbag is structured as a package or in the same directory
from zmongo_toolbag.data_processing import DataProcessor
from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo_retriever import ZMongoRetriever

# --- Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZMongoProcessor:
    """
    A high-level wrapper to process and search documents in a MongoDB collection.

    This class simplifies the workflow of embedding text from specified fields
    within documents and performing semantic searches on them.
    """

    def __init__(
            self,
            collection_name: str,
            text_field_keys: List[str],
            mongo_atlas: Optional[ZMongoAtlas] = None,
            gemini_api_key: Optional[str] = None
    ):
        """
        Initializes the processor.

        Args:
            collection_name (str): The name of the MongoDB collection to work with.
            text_field_keys (List[str]): A list of dot-separated keys to access the text fields
                                         (e.g., ['case_text', 'case_name']).
            mongo_atlas (Optional[ZMongoAtlas]): An existing ZMongoAtlas instance.
                                                 If None, a new one is created.
            gemini_api_key (Optional[str]): Your Google Gemini API key. If None,
                                            it's read from the environment.
        """
        if not collection_name or not text_field_keys:
            raise ValueError("collection_name and a list of text_field_keys must be provided.")

        self.collection_name = collection_name
        self.text_field_keys = text_field_keys
        self.repository = mongo_atlas or ZMongoAtlas()

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in your environment or passed to the constructor.")

        self.embedder = ZMongoEmbedder(
            repository=self.repository,
            collection=self.collection_name,
            page_content_key="content",
            gemini_api_key=api_key
        )

    def _get_embedding_field_name(self, text_field_key: str) -> str:
        """Creates a dynamic embedding field name from a text field key."""
        return text_field_key.replace('.', '_') + '_embedding'

    async def process_and_embed_collection(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 1000
    ) -> Dict[str, int]:
        """
        Iterates through documents and text fields, extracts text, generates
        embeddings, and stores them back into the document in dynamically named fields.

        This method is idempotent; it will skip documents that already have embeddings
        for a specific text field.

        Args:
            query (Optional[Dict[str, Any]]): A MongoDB query to filter which
                                              documents to process.
            limit (int): The maximum number of records to process for each text field
                         in a single run. Defaults to 1000.

        Returns:
            Dict[str, int]: A summary of the operation.
        """
        summary = {"total_docs_checked": 0, "embeddings_created": 0, "total_failures": 0, "total_skipped": 0}

        for key in self.text_field_keys:
            embedding_field = self._get_embedding_field_name(key)

            # Query for documents that do not have the specific embedding field yet.
            process_query = {embedding_field: {"$exists": False}}
            if query:
                process_query.update(query)

            logger.info(f"Processing field '{key}' -> '{embedding_field}' with query: {process_query} and limit: {limit}")

            find_result = await self.repository.find_documents(self.collection_name, process_query, limit=limit)
            if not find_result.success:
                logger.error(f"Failed to retrieve documents for field '{key}': {find_result.error}")
                continue

            documents_to_process = find_result.data
            if not documents_to_process:
                logger.info(f"No documents to process for field '{key}'.")
                continue

            logger.info(f"Found {len(documents_to_process)} documents to process for field '{key}'.")
            summary["total_docs_checked"] += len(documents_to_process)

            for doc in documents_to_process:
                doc_id = doc.get("_id")
                if not doc_id:
                    logger.warning("Skipping document with no _id.")
                    summary["total_skipped"] += 1
                    continue

                try:
                    text_to_embed = DataProcessor.get_value(doc, key)

                    if not text_to_embed or not isinstance(text_to_embed, str):
                        logger.warning(f"Skipping doc {doc_id} for field '{key}': text not found or not a string.")
                        summary["total_skipped"] += 1
                        continue

                    logger.info(f"Embedding text from '{key}' for document {doc_id}...")
                    await self.embedder.embed_and_store(
                        document_id=doc_id,
                        text=text_to_embed,
                        embedding_field=embedding_field
                    )
                    summary["embeddings_created"] += 1

                except Exception as e:
                    logger.error(f"Failed to process document {doc_id} for field '{key}': {e}")
                    summary["total_failures"] += 1

        logger.info(f"Processing complete. Summary: {summary}")
        return summary

    async def search(self, query_text: str, search_field: str, top_k: int = 5, similarity_threshold: float = 0.7) -> \
    List[Document]:
        """
        Performs a semantic vector search on a specific embedding field.

        Args:
            query_text (str): The text to search for.
            search_field (str): The original text field key whose embedding you want to search against.
            top_k (int): The number of top results to retrieve before filtering.
            similarity_threshold (float): The minimum relevance score to return.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        if search_field not in self.text_field_keys:
            raise ValueError(f"'{search_field}' is not one of the configured text fields for this processor.")

        embedding_field = self._get_embedding_field_name(search_field)
        logger.info(f"Performing search for: '{query_text}' on embedding field '{embedding_field}'")

        retriever = ZMongoRetriever(
            repository=self.repository,
            embedder=self.embedder,
            collection_name=self.collection_name,
            embedding_field=embedding_field,
            content_field=search_field,  # Use the original key for content
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        return await retriever.ainvoke(query_text)


async def main():
    """Example usage of the ZMongoProcessor."""
    collection_to_process = "cases"
    # Process both the prompt and response fields
    keys_to_process = ["case_text"]

    try:
        # 1. Initialize the processor with multiple keys
        processor = ZMongoProcessor(
            collection_name=collection_to_process,
            text_field_keys=keys_to_process
        )

        # 2. Process the collection, but only a maximum of 50 records for each field
        summary = await processor.process_and_embed_collection(limit=10)
        print(f"\nEmbedding process summary: {summary}")

        # 3. Perform a search on the 'response' embeddings
        search_query = "lost promissory note in foreclosure case"
        search_results = await processor.search(query_text=search_query,
                                                search_field=keys_to_process[0],
                                                top_k=5,
                                                similarity_threshold=0.7)

        print(f"\nFound {len(search_results)} results for the query: '{search_query}'")
        for doc in search_results:
            print("-" * 40)
            print(f"Score: {doc.metadata.get('retrieval_score'):.4f}")
            print(f"Source Document ID: {doc.metadata.get('_id')}")
            print(f"Prompt: {DataProcessor.get_value(doc.metadata, 'prompt')}")
            print(f"Response Snippet: {doc.page_content[:250]}...")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}", exc_info=True)
    finally:
        # Cleanly close the connection if a new one was created
        if 'processor' in locals() and hasattr(processor.repository, 'close'):
            await processor.repository.close()


if __name__ == "__main__":
    asyncio.run(main())
