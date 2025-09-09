"""
ZEmbedder_HF — Deterministic, Async Text Embedder using Hugging Face Sentence Transformers
=========================================================================================

This module provides a deterministic embedding utility that uses a local model from the
`sentence-transformers` library. It:

- Splits text into **chunks** (sentence / paragraph / fixed-size window).
- **Persists** vectors back into MongoDB via a `ZMongo` repository.
- Wraps results in a **SafeResult** for predictable error handling.
- Leverages `sentence-transformers` for high-quality, local, offline embeddings.

It integrates cleanly with vector search and higher-level retrievers. All public APIs
are **async**.

Environment
-----------
- `HF_MODEL_PATH` (required): Path to the sentence-transformer model directory
  (e.g., "C:/Users/iriye/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2").

Design Notes
------------
- The Sentence Transformer model is loaded into memory once on initialization.
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
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: `sentence-transformers` is not installed. This module requires it.")
    print("Please install it with: pip install sentence-transformers")
    SentenceTransformer = None

# Local (relative) imports
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.data_processing import SafeResult

# Load optional env files
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants / helpers (Chunking is identical to other versions)
# ---------------------------------------------------------------------

CHUNK_STYLE_FIXED = "fixed"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_PARAGRAPH = "paragraph"
DEFAULT_OUTPUT_DIM = 384  # For all-MiniLM-L6-v2

def field_name(base_field: str, model_name_suffix: str, chunk_style: str) -> str:
    """Compose a consistent MongoDB field name for persisted embeddings."""
    return f"{base_field}_{model_name_suffix}_{chunk_style}"

def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0: return [text] if text else []
    overlap = max(0, min(overlap, size - 1 if size > 1 else 0))
    chunks: List[str] = []
    start, n, step = 0, len(text), size - overlap if size > overlap else 1
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end == n: break
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
    style = (chunk_style or CHUNK_STYLE_SENTENCE).lower()
    if style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    return _sentence_split(text)

# ---------------------------------------------------------------------
# ZEmbedder_HF
# ---------------------------------------------------------------------

class ZEmbedder_HF:
    """Embed text locally using a Sentence Transformer model."""

    def __init__(
        self,
        repository: Optional[ZMongo] = None,
        model_path: Optional[str] = None,
    ):
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.model_path = model_path or os.getenv("HF_MODEL_PATH")
        self.model = None

        if not self.model_path or not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                "HF_MODEL_PATH not found in constructor or environment, "
                "or the path is not a valid directory."
            )

        if not SentenceTransformer:
            raise ImportError("`sentence-transformers` is required but not installed.")

        try:
            logger.info(f"Loading Sentence Transformer model from: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            logger.info("Sentence Transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise

    def close(self) -> None:
        if self._owns_repo and hasattr(self.repo, "close"):
            self.repo.close()

    async def _get_hf_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal helper to call the local Sentence Transformer model."""
        if not self.model:
            logger.error("Sentence Transformer model is not loaded.")
            return [[] for _ in texts]
        try:
            # Run the synchronous, CPU-bound encoding in a separate thread
            def encode_sync():
                embeddings_np = self.model.encode(texts, convert_to_tensor=False)
                return embeddings_np.tolist()

            embeddings = await asyncio.to_thread(encode_sync)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings from Sentence Transformer: {e}")
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
        return await self._get_hf_embedding_batch(chunks)

    async def _load_existing_vectors(
        self,
        collection: str,
        document_id: Any,
        embedding_field: str,
    ) -> Tuple[bool, List[List[float]]]:
        """Load and return existing vectors from a document field."""
        try:
            res = await self.repo.find_document(collection, {"_id": document_id})
            if res.success and res.data:
                existing = res.data.get(embedding_field)
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
                    payload = {
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
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        include_vectors_in_result: bool = False,
        skip_if_present: bool = True,
    ) -> SafeResult:
        """Convenience wrapper to derive the target field name and persist."""
        model_suffix = Path(self.model_path).name.replace('-', '_').replace('.', '_')
        target = field_name(base_field, model_suffix, chunk_style)
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
    if not os.getenv("HF_MODEL_PATH"):
        print("\nERROR: Please set the HF_MODEL_PATH environment variable.")
        print('Example: HF_MODEL_PATH="C:/Users/iriye/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"')
        return

    embedder = ZEmbedder_HF()
    DEMO_COLLECTION = "test_hf_embeddings"
    try:
        text = (
            "Sentence Transformers provide state-of-the-art embeddings. "
            "They are trained for semantic similarity tasks. This makes them ideal for search and clustering."
        )

        print("\n--- Persisting Sentence Transformer embeddings to Mongo ---")
        doc_id = ObjectId()
        await embedder.repo.delete_document(DEMO_COLLECTION, {"_id": doc_id})
        ins = await embedder.repo.insert_document(DEMO_COLLECTION, {"_id": doc_id, "text": text})
        assert ins.success, f"Insert failed: {ins.error}"

        res1 = await embedder.embed_field_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            base_field="text",
            text=text,
            chunk_style=CHUNK_STYLE_SENTENCE,
            skip_if_present=True,
        )
        print("Call #1 — saved OK?:", res1.success, "skipped?:", res1.data.get("skipped_compute") if res1.success else "N/A")

        res2 = await embedder.embed_field_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            base_field="text",
            text=text,
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
