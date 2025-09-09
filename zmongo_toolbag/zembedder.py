"""
ZEmbedder_Llama — Deterministic, Async Text Embedder for ZMongo using a Local Llama Model
========================================================================================

This module provides a compact, deterministic embedding utility that uses a local,
GGUF-compatible Llama model for generating embeddings. It:

- Splits text into **chunks** (sentence / paragraph / fixed-size window).
- **Persists** vectors back into MongoDB via a `ZMongo` repository.
- Wraps results in a **SafeResult** for predictable error handling.
- Leverages `llama-cpp-python` for local, offline embedding generation.

It integrates cleanly with `LocalVectorSearch` and higher-level retrievers.
All public APIs are **async**.

Environment
-----------
- `LLAMA_MODEL_PATH` (required): Full path to the GGUF-format embedding model file.

Return Conventions
------------------
All write operations return a `SafeResult`. On success:
`SafeResult.data` includes metadata such as `document_id`, `field`, `vectors_count`,
`dimensionality`, `chunk_style`, and flags `skipped_compute` / `from_cache`.

Design Notes
------------
- The Llama model is loaded into memory once on initialization.
- Synchronous embedding calls are performed in a background thread via
  `asyncio.to_thread` to keep the public API fully async.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from bson import ObjectId
from dotenv import load_dotenv

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: `llama-cpp-python` is not installed. This module requires it.")
    print("Please install it with: pip install llama-cpp-python")
    Llama = None

# Local (relative) imports
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.data_processing import SafeResult

# Load optional env files
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants / helpers
# ---------------------------------------------------------------------
# NOTE: Embedding and Chunking styles are specific to APIs like Gemini, LLama, OpenAi, etc.
# Chunking styles
CHUNK_STYLE_FIXED = "fixed"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_PARAGRAPH = "paragraph"

# Embedding “styles” (aka task types)
EMBEDDING_STYLE_SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBEDDING_STYLE_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
EMBEDDING_STYLE_CLASSIFICATION = "CLASSIFICATION"

# Model + dims (output_dimensionality is informational/config; the model defines true dims)
DEFAULT_OUTPUT_DIM = 768
EMBEDDING_MODEL = "embedding-001"  # Google Generative Language API model name



def field_name(base_field: str, model_name_suffix: str, chunk_style: str) -> str:
    """
    Compose a consistent MongoDB field name for persisted embeddings.
    """
    return f"{base_field}_{model_name_suffix}_{chunk_style}"


# ---------------------------------------------------------------------
# Chunking utilities (identical to Gemini version)
# ---------------------------------------------------------------------
def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text] if text else []
    overlap = max(0, min(overlap, size - 1 if size > 1 else 0))
    chunks: List[str] = []
    start, n, step = 0, len(text), size - overlap if size > overlap else 1
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks

def _sentence_split(text: str) -> List[str]:
    if not text: return []
    raw = [t.strip() for t in text.replace("\n", " ").split(".")]
    return [s + "." for s in raw if s]

def _paragraph_split(text: str) -> List[str]:
    if not text: return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]

def chunk_text(
    text: str,
    chunk_style: str = CHUNK_STYLE_SENTENCE,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    chunk_style = (chunk_style or CHUNK_STYLE_SENTENCE).lower()
    if chunk_style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if chunk_style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    return _sentence_split(text)

# ---------------------------------------------------------------------
# Config structures
# ---------------------------------------------------------------------

@dataclass
class EmbedConfig:
    """Configuration for local chunking + embedding."""
    chunk_style: str = CHUNK_STYLE_SENTENCE
    chunk_size: int = 400
    overlap: int = 50
    output_dimensionality: int = DEFAULT_OUTPUT_DIM

# ---------------------------------------------------------------------
# ZEmbedderLlama
# ---------------------------------------------------------------------

class ZEmbedder:
    """
    Embed text locally using a Llama model and persist vectors into MongoDB.
    """

    def __init__(
        self,
        repository: Optional[ZMongo] = None,
        model_path: Optional[str] = None,
    ):
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.model_path = os.getenv("LLAMA_MODEL_PATH") or model_path
        if not self.model_path.startswith("/") | self.model_path.startswith("C"):
            self.model_path = os.path.join(Path.home() / self.model_path)
        self.model = None

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "LLAMA_MODEL_PATH not found in constructor or environment, "
                "or the file does not exist."
            )

        if not Llama:
            raise ImportError("`llama-cpp-python` is required but not installed.")

        try:
            logger.info(f"Loading Llama embedding model from: {self.model_path}")
            # Adjust n_ctx based on your model's capabilities and expected text length
            self.model = Llama(model_path=self.model_path, embedding=True, verbose=False, n_ctx=2048)
            logger.info("Llama model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise

    def close(self) -> None:
        if self._owns_repo and hasattr(self.repo, "close"):
            self.repo.close()

    async def _get_llama_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal helper to call the local Llama embedding model."""
        if not self.model:
            logger.error("Llama model is not loaded. Cannot generate embeddings.")
            return [[] for _ in texts]
        try:
            # Use asyncio.to_thread to run the synchronous, CPU-bound embedding call
            embeddings = await asyncio.to_thread(self.model.embed, texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings from Llama model: {e}")
            return [[] for _ in texts]

    async def get_embedding(
        self,
        text: str,
        *,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> List[List[float]]:
        """Compute embeddings for a text, returning one vector per chunk."""
        chunks = chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []
        return await self._get_llama_embedding_batch(chunks)

    async def _load_existing_vectors(
        self,
        collection: str,
        document_id: Any,
        embedding_field: str,
    ) -> Tuple[bool, List[List[float]]]:
        """Load a document and read its existing vectors at `embedding_field`."""
        try:
            res = await self.repo.find_document(collection, {"_id": document_id})
            if res.success and res.data:
                doc = res.data
                existing = doc.get(embedding_field)
                if isinstance(existing, list) and existing and all(isinstance(x, list) for x in existing):
                    return True, existing
            return False, []
        except Exception:
            return False, []

    async def embed_and_store(
        self,
        *,
        collection: str,
        document_id,
        text: str,
        embedding_field: str,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        include_vectors_in_result: bool = True,
        skip_if_present: bool = True,
    ) -> SafeResult:
        """Compute and persist embeddings for a specific MongoDB document."""
        try:
            if skip_if_present:
                exists, existing_vectors = await self._load_existing_vectors(collection, document_id, embedding_field)
                if exists:
                    payload: Dict[str, Any] = {
                        "document_id": str(document_id),
                        "field": embedding_field,
                        "vectors_count": len(existing_vectors),
                        "dimensionality": len(existing_vectors[0]) if existing_vectors else 0,
                        "chunk_style": chunk_style,
                        "skipped_compute": True,
                        "from_cache": True,
                    }
                    if include_vectors_in_result:
                        payload["vectors"] = existing_vectors
                    return SafeResult.ok(payload)

            vectors = await self.get_embedding(
                text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap
            )

            if not vectors or not vectors[0]:
                return SafeResult.fail("Embedding computation returned no vectors.")

            update = {"$set": {embedding_field: vectors}}
            up_res = await self.repo.update_document(collection, {"_id": document_id}, update)

            if not up_res.success or up_res.data.get("matched_count", 0) == 0:
                err = up_res.error or f"Doc _id {document_id} not found."
                return SafeResult.fail(f"Failed to save embeddings: {err}")

            payload = {
                "document_id": str(document_id),
                "field": embedding_field,
                "vectors_count": len(vectors),
                "dimensionality": len(vectors[0]),
                "chunk_style": chunk_style,
                "skipped_compute": False,
                "from_cache": False,
            }
            if include_vectors_in_result:
                payload["vectors"] = vectors

            return SafeResult.ok(payload)
        except Exception as e:
            logger.exception("embed_and_store failed")
            return SafeResult.fail(str(e))

    async def embed_field_and_store(
        self,
        *,
        collection: str,
        document_id,
        base_field: str,
        text: str,
        model_name_suffix: str,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        include_vectors_in_result: bool = False,
        skip_if_present: bool = True,
    ) -> SafeResult:
        """Convenience wrapper to derive the target field name and persist."""
        target = field_name(base_field, model_name_suffix, chunk_style)
        return await self.embed_and_store(
            collection=collection,
            document_id=document_id,
            text=text,
            embedding_field=target,
            chunk_style=chunk_style,
            chunk_size=chunk_size,
            overlap=overlap,
            include_vectors_in_result=include_vectors_in_result,
            skip_if_present=skip_if_present,
        )

# ---------------------------------------------------------------------
# Simple demo (manual run)
# ---------------------------------------------------------------------

async def _demo() -> None:
    """Run a small end-to-end demonstration."""
    if not os.getenv("LLAMA_MODEL_PATH"):
        print("\nERROR: Please set the LLAMA_MODEL_PATH environment variable to a valid GGUF model file.")
        return

    embedder = ZEmbedder()
    DEMO_COLLECTION = "test_llama"
    try:
        text = (
            "Local embedding models offer privacy and control. They run on-premises, "
            "ensuring that sensitive data never leaves the local network. This is crucial for compliance."
        )

        print("\n--- Persisting Llama embeddings to Mongo ---")
        doc_id = ObjectId()
        await embedder.repo.delete_document(DEMO_COLLECTION, {"_id": doc_id})
        ins = await embedder.repo.insert_document(DEMO_COLLECTION, {"_id": doc_id, "text": text})
        assert ins.success, f"Insert failed: {ins.error}"

        # Get a short name for the model to use in the field name
        model_suffix = Path(embedder.model_path).stem.replace('.', '_')

        res1 = await embedder.embed_field_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            base_field="text",
            text=text,
            model_name_suffix=model_suffix,
            chunk_style=CHUNK_STYLE_SENTENCE,
            include_vectors_in_result=False,
            skip_if_present=True,
        )
        print("Call #1 — saved OK?:", res1.success, "skipped?:", res1.data.get("skipped_compute") if res1.success else "N/A")

        res2 = await embedder.embed_field_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            base_field="text",
            text=text,
            model_name_suffix=model_suffix,
            chunk_style=CHUNK_STYLE_SENTENCE,
            include_vectors_in_result=True,
            skip_if_present=True,
        )
        print("Call #2 — saved OK?:", res2.success, "skipped?:", res2.data.get("skipped_compute") if res2.success else "N/A")
        if res2.success:
            print("Returned vectors:", len(res2.data.get("vectors", [])))
            print("Dimensionality:", res2.data.get("dimensionality"))


    finally:
        embedder.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())
