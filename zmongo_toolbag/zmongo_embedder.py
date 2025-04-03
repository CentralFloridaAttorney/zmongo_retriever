import os
from typing import List
import logging
from bson.objectid import ObjectId
from dotenv import load_dotenv
from openai import AsyncOpenAI  # Assuming this is correctly installed

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

logger = logging.getLogger(__name__)

load_dotenv()

class ZMongoEmbedder:
    def __init__(self, repository: ZMongo, collection: str) -> None:
        """
        Initialize the ZMongoEmbedder with a repository and collection.

        Args:
            repository (ZMongo): The repository for database operations.
            collection (str): The name of the collection to operate on.
        """
        self.repository = repository
        self.collection = collection
        this_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(
            api_key= this_key # Fetch API key from environment variables
        )
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using OpenAI API.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector for the input text.

        Raises:
            ValueError: If the API response is invalid or malformed.
            Exception: If other errors occur while generating embeddings.
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model, input=[text]
            )
            # Validate response structure
            if not isinstance(response.data, list) or len(response.data) == 0:
                raise ValueError("Invalid response format from OpenAI API: missing embedding data")
            if not hasattr(response.data[0], "embedding"):
                raise ValueError("Invalid response format from OpenAI API: 'embedding' field is missing")

            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings for text: {e}")
            raise

    async def embed_and_store(self, document_id: ObjectId, text: str, embedding_field: str = "embedding") -> None:
        """
        Generate embeddings for the given text and store it in the database.
        """
        if not isinstance(document_id, ObjectId):
            raise ValueError("document_id must be an instance of ObjectId")
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        try:
            embedding = await self.embed_text(text)
            await self.repository.save_embedding(self.collection, document_id, embedding, embedding_field)
        except Exception as e:
            logger.error(f"Failed to embed and store text for document {document_id}: {e}")
            raise