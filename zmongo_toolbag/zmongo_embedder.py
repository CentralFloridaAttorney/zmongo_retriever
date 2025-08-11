import hashlib
import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional

from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
import google.generativeai as genai

from zmongo_toolbag.zmongo import ZMongo, SafeResult

logger = logging.getLogger(__name__)
load_dotenv(Path.home() / "resources" / ".env_local")


class ZMongoEmbedder:
    """
    Generates and stores text embeddings using the Google Gemini API, with a
    MongoDB-backed cache to avoid redundant API calls.
    """

    def __init__(self, repository: ZMongo, collection: str, page_content_key: str, gemini_api_key: Optional[str] = None):
        """
        Initializes the embedder with the repository, collection, and API key.

        Args:
            repository (ZMongo): The ZMongo repository instance.
            collection (str): The collection name to use for embedding-related operations.
            gemini_api_key (Optional[str]): The Gemini API key.
        """
        if not isinstance(repository, ZMongo):
            raise TypeError("repository must be an instance of ZMongo")

        self.repository = repository
        self.collection = collection
        self.page_content_key = page_content_key
        self.embedding_model_name = "models/embedding-001"

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set or passed to the constructor.")
        genai.configure(api_key=api_key)

    def _split_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text): break
            start += chunk_size - overlap
        return chunks

    async def _get_embedding_from_api(self, chunk: str) -> List[float]:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: genai.embed_content(
                model=self.embedding_model_name,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            ))
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_text(self, text: str) -> List[List[float]]:
        if not text: raise ValueError("text must be a non-empty string")
        chunks = self._split_chunks(text)
        if not chunks: return []

        embeddings = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            cached = await self.repository.find_document("_embedding_cache", {"text_hash": chunk_hash})

            if cached.success and cached.data:
                embeddings.append(cached.data["embedding"])
            else:
                embedding = await self._get_embedding_from_api(chunk)
                await self.repository.insert_document("_embedding_cache", {
                    "text_hash": chunk_hash, "embedding": embedding, "source_text": chunk
                })
                embeddings.append(embedding)
        return embeddings


    async def embed_and_store(self, document_id: str | ObjectId, text: str,
                              embedding_field: str = "embeddings") -> None:
        """
        Embeds text and stores the resulting vector in a specified document.
        Handles both string and ObjectId for the document_id.
        """
        try:
            # Ensure document_id is an ObjectId
            if isinstance(document_id, str):
                document_id = ObjectId(document_id)
        except InvalidId:
            # Handle cases where the string is not a valid ObjectId
            print(f"Error: Provided string '{document_id}' is not a valid ObjectId.")
            return  # Or raise a more specific error

        embeddings = await self.embed_text(text)
        if embeddings:
            await self.repository.update_document(
                self.collection,
                {"_id": document_id},
                {"$set": {embedding_field: embeddings}},
                upsert=True
            )


async def main(text_to_embed: str) -> SafeResult:
      zmongo = ZMongoEmbedder(repository=ZMongo(), collection="text", page_content_key="content", gemini_api_key=os.getenv("GEMINI_API_KEY"))
if __name__ == "__main__":
    text_to_embed = "Funny Joke about a lawyer and a pineapple."
    safe_result_embeddings = asyncio.run(main(text_to_embed))