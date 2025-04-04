import hashlib
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
        this_key = os.getenv("OPENAI_API_KEY_APP")
        self.openai_client = AsyncOpenAI(
            api_key= this_key # Fetch API key from environment variables
        )
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using OpenAI API or return cached version from DB.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector for the input text.
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        # Generate a stable hash of the text to use as a lookup key
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        try:
            # Check if embedding is already in the DB using a special _embedding_cache collection
            cached = await self.repository.find_document(
                collection="_embedding_cache",
                query={"text_hash": text_hash},
            )
            if cached and "embedding" in cached:
                logger.info(f"🔁 Reusing cached embedding for text hash: {text_hash}")
                return cached["embedding"]

            # If not cached, call OpenAI API
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )

            if not isinstance(response.data, list) or len(response.data) == 0:
                raise ValueError("Invalid response format from OpenAI API: missing embedding data")
            if not hasattr(response.data[0], "embedding"):
                raise ValueError("Invalid response format from OpenAI API: 'embedding' field is missing")

            embedding = response.data[0].embedding

            # Cache embedding in MongoDB
            await self.repository.insert_document("_embedding_cache", {
                "text_hash": text_hash,
                "embedding": embedding,
                "source_text": text
            })

            return embedding

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