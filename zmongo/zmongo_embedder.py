import asyncio
import logging
import numpy as np
import openai
from bson import ObjectId
from bson.errors import InvalidId
from openai import OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_exponential
from typing import List

from zmongo.zmongo_repository import ZMongoRepository
from zmongo.zmongo_chunker import ZMongoChunker
import zconstants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZMongoEmbedder:
    def __init__(
        self,
        page_content_keys: List[str],
        collection_name: str,
        embedding_model: str = zconstants.EMBEDDING_MODEL,
        max_tokens_per_chunk: int = 4096,
        overlap_prior_chunks: int = 0,
        encoding_name: str = zconstants.EMBEDDING_ENCODING,
        openai_api_key: str = zconstants.OPENAI_API_KEY,
    ):
        """
        Initialize the ZMongoEmbedder with the necessary parameters.
        """
        self.page_content_keys = page_content_keys
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.zmongo_repository = ZMongoRepository()
        self.chunker = ZMongoChunker(
            page_content_keys=page_content_keys,
            overlap_prior_chunks=overlap_prior_chunks,
            max_tokens_per_set=max_tokens_per_chunk,
            encoding_name=encoding_name,
            db_name=self.zmongo_repository.db_name,
            collection_name=collection_name,
        )
        openai.api_key = openai_api_key

    async def embed_collection(self) -> None:
        """
        Generate and save embeddings for all documents in the specified collection.
        Processes documents in batches to handle large datasets efficiently.
        """
        try:
            batch_size = 1000  # Adjust based on memory and performance requirements
            skip = 0
            total_processed = 0

            while True:
                # Fetch a batch of document IDs
                documents = await self.zmongo_repository.find_documents(
                    collection=self.collection_name,
                    query={},  # Fetch all documents
                    projection={"_id": 1},  # Only fetch the _id field
                    limit=batch_size,
                    skip=skip  # Added skip for pagination
                )

                if not documents:
                    # No more documents to process
                    break

                logger.info(f"Processing batch of {len(documents)} documents starting from skip={skip}")

                # Collect object IDs
                object_ids = [str(doc["_id"]) for doc in documents]

                await self.process_documents(object_ids)

                total_processed += len(object_ids)
                skip += batch_size

            logger.info(f"Finished processing {total_processed} documents in collection '{self.collection_name}'.")

        except Exception as e:
            logger.error(f"Error while embedding collection '{self.collection_name}': {e}")
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using OpenAI's API.
        """
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = self.get_embedding_from_response(response)
            return embedding
        except OpenAIError as e:
            logger.error(f"OpenAIError during get_embedding: {e}")
            raise

    @staticmethod
    def get_embedding_from_response(response) -> List[float]:
        """
        Extracts the embedding vector from the OpenAI API response.
        """
        # Access the data using attribute access
        embedding = response.data[0].embedding
        return embedding

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    async def process_documents(self, object_ids: List[str]) -> None:
        """
        Processes a list of document ObjectIDs to generate and save their embeddings.
        """
        valid_object_ids = []
        for oid in object_ids:
            try:
                valid_object_ids.append(ObjectId(oid))
            except InvalidId:
                logger.error(f"Invalid ObjectId: {oid}")

        if not valid_object_ids:
            logger.warning("No valid ObjectIds after parsing.")
            return

        # Use chunker to get chunked documents
        documents = await self.chunker.invoke(
            object_ids=object_ids,
            existing_metadata=None,  # You can pass existing metadata if needed
        )

        if not documents:
            logger.warning("No documents found or all documents are empty.")
            return

        # Group chunks by document_id and content_key
        documents_by_id_and_key = {}
        for doc in documents:
            document_id = doc.metadata.get('document_id')
            content_key = doc.metadata.get('page_content_key')
            if document_id not in documents_by_id_and_key:
                documents_by_id_and_key[document_id] = {}
            if content_key not in documents_by_id_and_key[document_id]:
                documents_by_id_and_key[document_id][content_key] = []
            documents_by_id_and_key[document_id][content_key].append(doc)

        for doc_id_str, content_dict in documents_by_id_and_key.items():
            try:
                doc_id = ObjectId(doc_id_str)
            except InvalidId:
                logger.error(f"Invalid ObjectId in document metadata: {doc_id_str}")
                continue

            for content_key, doc_chunks in content_dict.items():
                # Define the embedding field dynamically based on content_key
                embedding_field = f"embeddings.{content_key.replace('.', '_')}"

                # Check if embedding already exists
                existing_embedding = await self.zmongo_repository.fetch_embedding(
                    collection=self.collection_name,
                    document_id=doc_id,
                    embedding_field=embedding_field
                )
                if existing_embedding:
                    logger.info(f"Embedding already exists for document ID {doc_id} and content key '{content_key}'. Skipping API call.")
                    continue  # Skip to the next content_key

                # Proceed to generate embeddings since they don't exist
                embeddings = []
                for doc in doc_chunks:
                    chunk = doc.page_content
                    try:
                        embedding = await self.get_embedding(chunk)
                        embeddings.append(embedding)
                    except OpenAIError as e:
                        logger.error(f"Error generating embedding for chunk in document ID {doc_id} and content key '{content_key}': {e}")
                        continue

                if embeddings:
                    # Convert embeddings to numpy array of float64
                    embeddings_array = np.array(embeddings, dtype=float)
                    avg_embedding = np.mean(embeddings_array, axis=0)

                    # Check for NaN or Infinity values
                    if np.any(np.isnan(avg_embedding)) or np.any(np.isinf(avg_embedding)):
                        logger.error(f"Embedding contains NaN or Infinity values for document ID {doc_id} and content key '{content_key}'. Skipping.")
                        continue

                    avg_embedding = avg_embedding.tolist()
                    # Ensure that the embedding is a list of Python floats
                    avg_embedding = [float(x) for x in avg_embedding]

                    # Save embedding under the dynamic field
                    await self.zmongo_repository.save_embedding(
                        collection=self.collection_name,
                        document_id=doc_id,
                        embedding=avg_embedding,
                        embedding_field=embedding_field
                    )
                    logger.info(f"Saved embedding for document ID {doc_id} and content key '{content_key}'.")
                else:
                    logger.warning(f"No embeddings generated for document ID {doc_id} and content key '{content_key}'.")

async def main():
    # List of content keys (dot-separated paths)
    page_content_keys = [
        'meaning_upright',
        'meaning_reversed'
        # Add any other content keys as needed
    ]

    # Initialize ZMongoEmbedder with multiple content keys
    embedder = ZMongoEmbedder(
        page_content_keys=page_content_keys,
        collection_name='tarot_cards',        # Replace with your collection name
        embedding_model=zconstants.EMBEDDING_MODEL,
        max_tokens_per_chunk=128,
        overlap_prior_chunks=50,
        encoding_name=zconstants.EMBEDDING_ENCODING,
        openai_api_key=zconstants.OPENAI_API_KEY,
    )

    # Embed the entire collection
    await embedder.embed_collection()


if __name__ == "__main__":
    asyncio.run(main())
