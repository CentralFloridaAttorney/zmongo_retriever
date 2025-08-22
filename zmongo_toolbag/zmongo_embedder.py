# zmongo_retriever/zmongo_toolbag/zmongo_embedder.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from bson import ObjectId

# Local (relative) imports from the toolbag
try:
    from .zmongo import ZMongo
    from .data_processing import SafeResult
except Exception:  # pragma: no cover - allows running module directly for manual tests
    from zmongo import ZMongo
    from data_processing import SafeResult

logger = logging.getLogger(__name__)

# ----------------------------
# Public constants / helpers
# ----------------------------
CHUNK_STYLE_FIXED = "fixed"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_PARAGRAPH = "paragraph"

EMBEDDING_STYLE_SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBEDDING_STYLE_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
EMBEDDING_STYLE_CLASSIFICATION = "CLASSIFICATION"

DEFAULT_OUTPUT_DIM = 768


def field_name(base_field: str, embedding_style: str, chunk_style: str) -> str:
    """
    Compose the persisted embedding field name:
    [BASE_FIELD]_[EMBEDDING_STYLE]_[CHUNK_STYLE]
    e.g., "text_RETRIEVAL_DOCUMENT_sentence"
    """
    return f"{base_field}_{embedding_style}_{chunk_style}"


# ----------------------------
# Chunking utilities
# ----------------------------
def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = size - 1 if size > 1 else 0

    chunks: List[str] = []
    start = 0
    n = len(text)
    step = size - overlap if size > overlap else 1
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
    # Lightweight sentence split that is deterministic and dependency-free.
    # You can swap in nltk/spacy elsewhere.
    if not text:
        return []
    raw = [t.strip() for t in text.replace("\n", " ").split(".")]
    return [s + "." for s in raw if s]


def _paragraph_split(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_text(
    text: str,
    chunk_style: str = CHUNK_STYLE_SENTENCE,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Returns a list of chunk strings according to the specified strategy.
    """
    chunk_style = (chunk_style or CHUNK_STYLE_SENTENCE).lower()

    if chunk_style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if chunk_style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    # default: sentence
    return _sentence_split(text)


# ----------------------------
# Deterministic local embedder (fallback)
# ----------------------------
def _deterministic_vec(text: str, dim: int) -> np.ndarray:
    """
    Deterministically map text -> vector for tests/offline mode.
    Uses a seeded PRNG based on a stable hash of the text+dim.
    """
    h = hashlib.sha256(f"{text}::{dim}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.05, size=dim).astype(np.float64)
    # Apply a mild normalization for stability
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    # Nudge the 4th coordinate to carry some "style-ish" separation range
    v[3] -= 0.1
    return v


def _style_bias(style: str, dim: int) -> np.ndarray:
    """
    Simple style-dependent bias vector so styles aren't identical.
    """
    style_key = {
        EMBEDDING_STYLE_SEMANTIC_SIMILARITY: "SS",
        EMBEDDING_STYLE_RETRIEVAL_DOCUMENT: "RD",
        EMBEDDING_STYLE_RETRIEVAL_QUERY: "RQ",
        EMBEDDING_STYLE_CLASSIFICATION: "CLF",
    }.get(style, "GEN")

    h = hashlib.md5(f"{style_key}:{dim}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "little")
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, 0.01, size=dim).astype(np.float64)
    return bias


# ----------------------------
# Config structures
# ----------------------------
@dataclass
class EmbedConfig:
    embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
    chunk_style: str = CHUNK_STYLE_SENTENCE
    chunk_size: int = 400
    overlap: int = 50
    output_dimensionality: int = DEFAULT_OUTPUT_DIM


# ----------------------------
# ZMongoEmbedder
# ----------------------------
class ZMongoEmbedder:
    """
    Embeds text (by chunks) and persists embeddings into MongoDB documents.

    Field naming convention on the document:
        [BASE_FIELD]_[EMBEDDING_STYLE]_[CHUNK_STYLE]
    Example:
        "text_RETRIEVAL_DOCUMENT_sentence": [[v1...], [v2...], ...]

    You may pass a ZMongo `repository` or allow this class to create its own.
    """

    def __init__(
        self,
        collection: str,
        repository: Optional[ZMongo] = None,
        gemini_api_key: Optional[str] = None,  # placeholder for a real remote embedder
    ):
        self.collection = collection
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.gemini_api_key = gemini_api_key

    # --------- lifecycle ----------
    def close(self):
        if self._owns_repo and hasattr(self.repo, "close"):
            try:
                self.repo.close()
            except Exception:
                pass

    # --------- public API ----------
    async def get_embedding(
        self,
        text: str,
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
    ) -> List[List[float]]:
        """
        Returns chunked embeddings (NOT persisted). This is the building block for callers that
        want to compute vectors without saving them.

        Output is a list of vectors, one per chunk.
        """
        chunks = chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        bias = _style_bias(embedding_style, output_dimensionality)
        vectors: List[List[float]] = []
        for ch in chunks:
            base = _deterministic_vec(ch, output_dimensionality)
            v = (base + bias).tolist()
            vectors.append(v)
        return vectors

    async def embed_and_store(
        self,
        *,
        document_id,
        text: str,
        embedding_field: str,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
        include_vectors_in_result: bool = True,
    ) -> SafeResult:
        """
        Computes embeddings for `text` (chunked) and persists them on the document
        under `embedding_field`. Returns a SafeResult with summary + (optionally) vectors.

        It *does not* remove any other fields—only $sets this one field.
        """
        try:
            vectors = await self.get_embedding(
                text,
                embedding_style=embedding_style,
                chunk_style=chunk_style,
                chunk_size=chunk_size,
                overlap=overlap,
                output_dimensionality=output_dimensionality,
            )

            # Persist to Mongo
            update = {"$set": {embedding_field: vectors}}
            up_res = await self.repo.update_document(self.collection, {"_id": document_id}, update)
            if not getattr(up_res, "success", False):
                return SafeResult.fail(f"Failed to save embeddings: {getattr(up_res, 'error', 'unknown error')}")

            payload = {
                "document_id": str(document_id),
                "field": embedding_field,
                "vectors_count": len(vectors),
                "dimensionality": output_dimensionality,
                "embedding_style": embedding_style,
                "chunk_style": chunk_style,
            }
            if include_vectors_in_result:
                payload["vectors"] = vectors

            return SafeResult.ok(payload)
        except Exception as e:
            logger.exception("embed_and_store failed")
            return SafeResult.fail(str(e))

    # Convenience: one-shot for a base field where we compute the name for you.
    async def embed_field_and_store(
        self,
        *,
        document_id,
        base_field: str,
        text: str,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
        include_vectors_in_result: bool = False,
    ) -> SafeResult:
        target = field_name(base_field, embedding_style, chunk_style)
        return await self.embed_and_store(
            document_id=document_id,
            text=text,
            embedding_field=target,
            chunk_style=chunk_style,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_style=embedding_style,
            output_dimensionality=output_dimensionality,
            include_vectors_in_result=include_vectors_in_result,
        )

    # Batch text embeddings (no persistence) — handy for populating docs first, then writing.
    async def embed_texts_batched(
        self,
        texts: Iterable[str],
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
    ) -> dict[str, List[List[float]]]:
        """
        Returns a dict: original_text -> [vectors...]
        (pure compute, does not write to Mongo)
        """
        results: dict[str, List[List[float]]] = {}
        for t in texts:
            vecs = await self.get_embedding(
                t,
                embedding_style=embedding_style,
                chunk_style=chunk_style,
                chunk_size=chunk_size,
                overlap=overlap,
                output_dimensionality=output_dimensionality,
            )
            results[t] = vecs
        return results

# -------------------- Demo: run different methods on the same string --------------------



async def _demo():
    # Use the collection where demo docs + embeddings will live
    embedder = ZMongoEmbedder(collection="demo_embeddings")
    try:
        text = (
            "Artificial intelligence is transforming the legal industry. "
            "Lawyers now use AI for document review, case prediction, and drafting. "
            "These tools improve efficiency but also raise questions about ethics and accountability."
        )

        print("\n--- Chunking styles (embedding_style=RETRIEVAL_DOCUMENT, dim=768) ---")
        for cs in (CHUNK_STYLE_FIXED, CHUNK_STYLE_SENTENCE, CHUNK_STYLE_PARAGRAPH):
            vecs = await embedder.get_embedding(
                text,
                chunk_style=cs,
                chunk_size=160,   # small to provoke multiple chunks
                overlap=1,
                embedding_style="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            )
            first8 = vecs[0][:8] if vecs else []
            print(f"{cs:<10}: {len(vecs)} vector(s); first8={first8}")

        print("\n--- Embedding styles (chunk_style=sentence, dim=768) ---")
        for es in ("SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"):
            vecs = await embedder.get_embedding(
                text,
                chunk_style=CHUNK_STYLE_SENTENCE,
                chunk_size=220,
                overlap=0,
                embedding_style=es,     # task type
                output_dimensionality=768,
            )
            first8 = vecs[0][:8] if vecs else []
            print(f"{es:<20}: {len(vecs)} vector(s); first8={first8}")

        # ---- Persist an embedding on a real document ----
        print("\n--- Persisting embeddings to Mongo ---")
        doc_id = ObjectId()
        # Insert a base document with a 'text' field
        ins = await embedder.repo.insert_document(
            "demo_embeddings",
            {"_id": doc_id, "text": text}
        )
        assert ins.success, f"Insert failed: {ins.error}"

        # Name the embedding field using the convention [BASE]_[STYLE]_[CHUNK]
        target_field = field_name("text", "RETRIEVAL_DOCUMENT", "sentence")

        # Compute + save the embeddings onto the document
        res = await embedder.embed_and_store(
            document_id=doc_id,
            text=text,
            embedding_field=target_field,
            chunk_style=CHUNK_STYLE_SENTENCE,
            chunk_size=220,
            overlap=0,
            embedding_style="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
            include_vectors_in_result=True,  # SafeResult will include vectors
        )
        print("Saved OK?:", res.success)
        if res.success:
            print("Vectors persisted:", res.data.get("vectors_count"))
            # For demo purposes only, peek at the first 8 numbers of the first vector:
            vectors = res.data.get("vectors") or []
            if vectors:
                print("First vector first8:", vectors[0][:8])

        # Read back from Mongo and verify the field exists
        got = await embedder.repo.find_document("demo_embeddings", {"_id": doc_id})
        assert got.success and got.data, f"Find failed: {got.error}"
        present = target_field in got.data
        print(f"Field '{target_field}' present in doc?:", present)
        if present:
            print("Stored chunk count:", len(got.data[target_field]))
            print("Stored vector dim:", len(got.data[target_field][0]) if got.data[target_field] else 0)

    finally:
        embedder.close()


if __name__ == "__main__":
    asyncio.run(_demo())


