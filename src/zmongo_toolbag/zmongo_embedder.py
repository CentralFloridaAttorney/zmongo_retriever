import hashlib
import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from itertools import chain

from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
import google.generativeai as genai

from zmongo_toolbag.zmongo import ZMongo, SafeResult

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(Path.home() / "resources" / ".env_local")


class ZMongoEmbedder:
    """
    Generates and stores text embeddings using the Google Gemini API,
    with a highly efficient, cache-first batching workflow.
    """

    def __init__(self, collection: str, gemini_api_key: Optional[str] = None):

        self.repository = ZMongo()
        self.collection = collection
        self.embedding_model_name = "models/embedding-001"

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        genai.configure(api_key=api_key)

    # --- Core Private Methods ---

    def _split_text_into_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
        """Splits a single text into manageable, overlapping chunks."""
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text): break
            start += chunk_size - overlap
        return chunks

    async def _embed_chunks_via_api(self, chunks: List[str]) -> Dict[str, List[float]]:
        """Sends a batch of text chunks to the Gemini API in a single request."""
        if not chunks: return {}
        logger.info(f"Making API call to embed {len(chunks)} new chunk(s).")
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: genai.embed_content(
                model=self.embedding_model_name, content=chunks, task_type="RETRIEVAL_DOCUMENT"
            ))
            return dict(zip(chunks, result['embedding']))
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def _get_embeddings_from_chunks(self, chunks: List[str]) -> Dict[str, List[float]]:
        """
        **Cache-First Implementation**: Gets embeddings for chunks by first
        querying ZMongo, then batching any misses for an API call.
        """
        if not chunks: return {}
        hashes = [hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
        hash_to_chunk = dict(zip(hashes, chunks))

        cache_results = await self.repository.find_documents("_embedding_cache", {"text_hash": {"$in": hashes}})

        # --- FIX: Safely handle cases where cache_results.data is None ---
        found_embeddings = {
            res["source_text"]: res["embedding"]
            for res in (cache_results.data or []) if cache_results.success
        }
        logger.info(f"Found {len(found_embeddings)} of {len(chunks)} chunks in ZMongo cache.")

        cached_hashes = {res["text_hash"] for res in (cache_results.data or [])}
        missing_hashes = set(hashes) - cached_hashes
        chunks_to_embed = [hash_to_chunk[h] for h in missing_hashes]

        if chunks_to_embed:
            api_embeddings = await self._embed_chunks_via_api(chunks_to_embed)
            new_cache_entries = [
                {"text_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(), "embedding": emb, "source_text": chunk}
                for chunk, emb in api_embeddings.items()]
            if new_cache_entries:
                await self.repository.insert_documents("_embedding_cache", new_cache_entries)
            found_embeddings.update(api_embeddings)

        return found_embeddings

    # --- Public-Facing Methods ---

    async def embed_text(self, text: str) -> List[List[float]]:
        """
        Embeds a single text, splitting it into chunks if necessary,
        using the efficient cache-first workflow.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        chunks = self._split_text_into_chunks(text)
        if not chunks:
            return []

        embedding_map = await self._get_embeddings_from_chunks(chunks)
        return [embedding_map[chunk] for chunk in chunks if chunk in embedding_map]

    async def embed_texts_batched(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        """
        Efficiently embeds a batch of texts, preserving the relationship
        between each original text and its chunked embeddings.
        """
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("texts must be a non-empty list of non-empty strings")

        text_to_chunks_map = {text: self._split_text_into_chunks(text) for text in texts}
        all_chunks = list(chain.from_iterable(text_to_chunks_map.values()))
        unique_chunks = sorted(list(set(all_chunks)))

        embedding_map = await self._get_embeddings_from_chunks(unique_chunks)

        result = {}
        for text, chunks in text_to_chunks_map.items():
            text_embeddings = [embedding_map[chunk] for chunk in chunks if chunk in embedding_map]
            result[text] = text_embeddings
        return result

    async def embed_and_store(self, document_id: str | ObjectId, text: str,
                              embedding_field: str = "embeddings") -> SafeResult:
        """
        Embeds a single text and stores its chunked embeddings in a specified document.
        """
        try:
            obj_id = ObjectId(document_id) if isinstance(document_id, str) else document_id
        except InvalidId:
            error_msg = f"Error: Provided string '{document_id}' is not a valid ObjectId."
            logger.error(error_msg)
            return self.repository.fail(error_msg)

        embeddings = await self.embed_text(text)
        if embeddings:
            return await self.repository.update_document(
                self.collection,
                {"_id": obj_id},
                {"$set": {embedding_field: embeddings}},
                upsert=True
            )
        return self.repository.ok(data={"message": "No embeddings were generated or stored."})
