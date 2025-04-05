import hashlib
import os
import logging
from typing import List, Tuple

from bson.objectid import ObjectId
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken

from zmongo_toolbag.zmongo import ZMongo

logger = logging.getLogger(__name__)
load_dotenv()

class ZMongoEmbedder:
    def __init__(self, collection: str, repository: ZMongo = None) -> None:
        """
        Initialize the ZMongoEmbedder with a repository and collection.
        """
        self.repository = repository or ZMongo()
        self.collection = collection
        api_key = os.getenv("OPENAI_API_KEY_APP")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.max_tokens = int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192"))
        self.encoding_name = os.getenv("EMBEDDING_ENCODING", "cl100k_base")

    def _truncate_text_to_max_tokens(self, text: str) -> str:
        """
        Truncate the input text to fit within the model's token limit.
        """
        encoding = tiktoken.get_encoding(self.encoding_name)
        encoded = encoding.encode(text)
        if len(encoded) > self.max_tokens:
            logger.warning(f"⚠️ Input text exceeds {self.max_tokens} tokens. Truncating.")
            encoded = encoded[:self.max_tokens]
        return encoding.decode(encoded)

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using OpenAI API or return cached version from DB.
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        safe_text = self._truncate_text_to_max_tokens(text)
        text_hash = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()

        try:
            cached_doc = await self.repository.find_document("_embedding_cache", {"text_hash": text_hash})
            if cached_doc and "embedding" in cached_doc:
                logger.info(f"🔁 Reusing cached embedding for text hash: {text_hash}")
                logger.debug("Source: MongoDB cache")
                return cached_doc["embedding"]

            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[safe_text]
            )

            # If 'response.data' is empty, or missing 'embedding', raise ValueError:
            if not response.data:
                # e.g. data = []
                raise ValueError("OpenAI embedding response is empty; expected at least one embedding.")

            first_record = response.data[0]
            if not hasattr(first_record, "embedding"):
                # e.g. data = [NoEmbedding()]
                raise ValueError("OpenAI response is missing embedding data.")

            embedding = first_record.embedding

            await self.repository.insert_document("_embedding_cache", {
                "text_hash": text_hash,
                "embedding": embedding,
                "source_text": safe_text
            })

            logger.info(f"✅ Generated new embedding for text hash: {text_hash}")
            logger.debug("Source: OpenAI API")
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
