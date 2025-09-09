"""
ZEmbedder — deterministic, async text embedder for ZMongo
=========================================================

This module provides a compact, deterministic embedding utility that:

- Splits text into **chunks** (sentence / paragraph / fixed-size window).
- Produces embeddings in different **styles** (e.g., Retrieval-Document vs. Retrieval-Query).
- **Persists** vectors back into MongoDB via a `ZMongo` repository.
- Wraps results in a **SafeResult** for predictable error handling.

It integrates cleanly with `LocalVectorSearch` (for cosine search) and higher-level
retrievers like `ZRetriever`. All public APIs are **async**.

Environment
-----------
- `GEMINI_API_KEY` (required): used to call the Google Generative Language
  embedding endpoint.
- `EMBEDDING_MODEL` (optional): model name for the embedding endpoint.
  Defaults to `"embedding-001"`.

Return Conventions
------------------
All write operations return a `SafeResult`. On success:
`SafeResult.data` includes metadata such as `document_id`, `field`, `vectors_count`,
`dimensionality`, `embedding_style`, `chunk_style`, and flags
`skipped_compute` / `from_cache`. Optionally, the actual vectors (`vectors`)
can be included in the result.

Design Notes
------------
- Network calls are performed with `requests` in a background thread via
  `asyncio.to_thread`, keeping the public API fully async without introducing
  an async HTTP dependency.
- When `skip_if_present=True`, embeddings are *not recomputed* if the target
  field already contains a non-empty list of vectors.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any, Dict

from bson import ObjectId  # noqa: F401 (used in demo and type context)
from dotenv import load_dotenv

# Local (relative) imports from the toolbag. The try/except allows
# running this module directly for manual testing without package context.
try:
    from zmongo_toolbag.zmongo import ZMongo
    from zmongo_toolbag.data_processing import SafeResult
except Exception:  # pragma: no cover - allows running module directly for manual tests
    from zmongo import ZMongo
    from data_processing import SafeResult

# Load optional env files (no-op if they don't exist)
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants / helpers
# ---------------------------------------------------------------------

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


def field_name(base_field: str, embedding_style: str, chunk_style: str) -> str:
    """
    Compose a consistent MongoDB field name for persisted embeddings.

    Parameters
    ----------
    base_field : str
        The source text field name (e.g., "text").
    embedding_style : str
        One of the `EMBEDDING_STYLE_*` constants (e.g., "RETRIEVAL_DOCUMENT").
    chunk_style : str
        One of CHUNK_STYLE_FIXED / CHUNK_STYLE_SENTENCE / CHUNK_STYLE_PARAGRAPH.

    Returns
    -------
    str
        A deterministic compound field name, e.g. "text_RETRIEVAL_DOCUMENT_sentence".
    """
    return f"{base_field}_{embedding_style}_{chunk_style}"


# ---------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------

def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    """
    Produce fixed-size, overlapping chunks from `text`.

    Parameters
    ----------
    text : str
        Source text.
    size : int
        Target chunk size in characters. If <= 0, returns `[text]` (or `[]` for empty).
    overlap : int
        Overlap between adjacent chunks in characters. Negative values are treated as 0.

    Returns
    -------
    List[str]
        A list of non-empty chunks (whitespace-trimmed).
    """
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
    """
    Naïve sentence splitter using '.' as delimiter.

    Parameters
    ----------
    text : str
        Source text.

    Returns
    -------
    List[str]
        A list of sentence-like segments ending with '.'.
    """
    if not text:
        return []
    raw = [t.strip() for t in text.replace("\n", " ").split(".")]
    return [s + "." for s in raw if s]


def _paragraph_split(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines as delimiters.

    Parameters
    ----------
    text : str
        Source text.

    Returns
    -------
    List[str]
        A list of paragraph strings (non-empty).
    """
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
    Chunk text according to the requested `chunk_style`.

    Parameters
    ----------
    text : str
        Source text to chunk.
    chunk_style : str, default "sentence"
        One of CHUNK_STYLE_FIXED / CHUNK_STYLE_SENTENCE / CHUNK_STYLE_PARAGRAPH.
    chunk_size : int, default 500
        Character window size (only used when `chunk_style == "fixed"`).
    overlap : int, default 50
        Character overlap for fixed-size windows.

    Returns
    -------
    List[str]
        A list of chunk strings (may be length 1 if sentence/paragraph logic yields a single block).
    """
    chunk_style = (chunk_style or CHUNK_STYLE_SENTENCE).lower()

    if chunk_style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if chunk_style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    # default: sentence
    return _sentence_split(text)


# ---------------------------------------------------------------------
# Config structures
# ---------------------------------------------------------------------

@dataclass
class EmbedConfig:
    """
    Configuration for chunking + embedding.

    Attributes
    ----------
    embedding_style : str
        e.g., EMBEDDING_STYLE_RETRIEVAL_DOCUMENT / EMBEDDING_STYLE_RETRIEVAL_QUERY.
    chunk_style : str
        CHUNK_STYLE_FIXED / CHUNK_STYLE_SENTENCE / CHUNK_STYLE_PARAGRAPH.
    chunk_size : int
        Character window when using fixed-size chunking.
    overlap : int
        Overlap between adjacent fixed-size chunks.
    output_dimensionality : int
        Target dimensionality for the embedding model (informational).
    """
    embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
    chunk_style: str = CHUNK_STYLE_SENTENCE
    chunk_size: int = 400
    overlap: int = 50
    output_dimensionality: int = DEFAULT_OUTPUT_DIM


# ---------------------------------------------------------------------
# ZEmbedder
# ---------------------------------------------------------------------

class ZEmbedder:
    """
    Embed text (by chunks) and optionally persist vectors into MongoDB.

    The embedder uses Google Generative Language’s embedding endpoint with
    the model specified by `EMBEDDING_MODEL`. All network calls are performed
    via `requests` in a thread (through `asyncio.to_thread`) to keep async APIs.

    Parameters
    ----------
    repository : ZMongo, optional
        Repository used for reading/updating MongoDB documents. If omitted, a new
        `ZMongo()` is created and owned by this instance.
    gemini_api_key : str, optional
        API key for Google’s Generative Language API. If not provided, the
        value from environment variable `GEMINI_API_KEY` is used.
    embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
        Default embedding style used by the instance when a per-call style
        is not explicitly provided.

    Notes
    -----
    - If no API key is available, embedding calls will log a warning and return
      empty vectors for each requested chunk (so callers can handle gracefully).
    """

    def __init__(
        self,
        repository: Optional[ZMongo] = None,
        gemini_api_key: Optional[str] = None,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    ):
        self.embedding_style = embedding_style or EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not found in constructor or environment. "
                "Embedding calls will fail."
            )

    def close(self) -> None:
        """
        Close the underlying repository if this embedder created it.

        Notes
        -----
        Safe to call multiple times.
        """
        if self._owns_repo and hasattr(self.repo, "close"):
            try:
                self.repo.close()
            except Exception:
                pass

    async def _get_gemini_embedding_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Internal helper to call the Gemini embedding API for a batch of texts.

        Parameters
        ----------
        texts : List[str]
            List of input strings to embed.
        task_type : str
            One of: "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY", or "CLASSIFICATION".

        Returns
        -------
        List[List[float]]
            For each input string, a vector of floats representing its embedding.
            If the call fails (no API key or network error), returns a list of
            empty lists aligned to the inputs.

        Notes
        -----
        - Performs up to 5 attempts with exponential backoff (2^i seconds).
        - Uses `requests.post` in a background thread to avoid blocking the event loop.
        """
        if not self.gemini_api_key:
            logger.error("Cannot call Gemini API: GEMINI_API_KEY is not set.")
            return [[] for _ in texts]

        apiUrl = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EMBEDDING_MODEL}:batchEmbedContents?key={self.gemini_api_key}"
        )

        # Prepare JSON payload for batch embed call
        requests_payload = []
        for text in texts:
            requests_payload.append({
                "model": f"models/{EMBEDDING_MODEL}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            })
        payload = {"requests": requests_payload}

        # Exponential backoff on transient failures
        for i in range(5):  # Retry up to 5 times
            try:
                response = await asyncio.to_thread(
                    lambda: __import__("requests").post(
                        apiUrl,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=60,
                    )
                )
                response.raise_for_status()
                result = response.json()
                return [embedding["values"] for embedding in result["embeddings"]]
            except Exception as e:
                logger.warning("Gemini API call failed (attempt %d): %s", i + 1, e)
                if i < 4:
                    await asyncio.sleep(2**i)  # 1, 2, 4, 8 seconds

        logger.error("Failed to get embeddings from Gemini API after multiple retries.")
        return [[] for _ in texts]  # Return empty lists on failure

    async def get_embedding(
        self,
        text: str,
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_SENTENCE,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,  # informational only
    ) -> List[List[float]]:
        """
        Compute embeddings for a text, returning one vector per chunk.

        Parameters
        ----------
        text : str
            Source text to embed.
        embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
            Embedding task type to use for the API call.
        chunk_style : str, default CHUNK_STYLE_SENTENCE
            Chunking strategy: one of CHUNK_STYLE_FIXED / CHUNK_STYLE_SENTENCE / CHUNK_STYLE_PARAGRAPH.
        chunk_size : int, default 400
            Character window (used only when `chunk_style == "fixed"`).
        overlap : int, default 50
            Character overlap for fixed-size chunking.
        output_dimensionality : int, default 768
            Informational value for callers; the model controls the actual size.

        Returns
        -------
        List[List[float]]
            A list of vectors, one per chunk (empty if no chunks or on failure).

        """
        chunks = chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        task_type_map = {
            EMBEDDING_STYLE_RETRIEVAL_QUERY: "RETRIEVAL_QUERY",
            EMBEDDING_STYLE_RETRIEVAL_DOCUMENT: "RETRIEVAL_DOCUMENT",
            EMBEDDING_STYLE_SEMANTIC_SIMILARITY: "SEMANTIC_SIMILARITY",
            EMBEDDING_STYLE_CLASSIFICATION: "CLASSIFICATION",
        }
        task_type = task_type_map.get(embedding_style, "RETRIEVAL_DOCUMENT")

        return await self._get_gemini_embedding_batch(chunks, task_type)

    async def _load_existing_vectors(
        self,
        collection: str,
        document_id: Any,
        embedding_field: str,
    ) -> Tuple[bool, List[List[float]]]:
        """
        Load a document and read its existing vectors at `embedding_field`.

        Parameters
        ----------
        collection : str
            MongoDB collection name.
        document_id : Any
            The document `_id`. May be `ObjectId` or string (repository handles coercion).
        embedding_field : str
            Field to read (e.g., "text_RETRIEVAL_DOCUMENT_sentence").

        Returns
        -------
        Tuple[bool, List[List[float]]]
            `(exists, vectors)` where:
            - `exists` is True if the field is present and contains a non-empty list.
            - `vectors` is the existing list of vectors if present; otherwise empty.

        Notes
        -----
        Any repository errors are swallowed and treated as "not found", so the
        caller can proceed to recompute vectors.
        """
        try:
            res = await self.repo.find_document(collection, {"_id": document_id})
            if not res or not res.success or not res.data:
                return False, []
            doc = res.data
            existing = doc.get(embedding_field)
            if isinstance(existing, list) and existing and all(isinstance(x, (list, tuple)) for x in existing):
                return True, [list(x) for x in existing]
            return False, []
        except Exception:
            logger.debug("Failed to load existing vectors (fallback to recompute).", exc_info=True)
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
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,  # informational only
        include_vectors_in_result: bool = True,
        skip_if_present: bool = True,
    ) -> SafeResult:
        """
        Compute and persist embeddings for a specific MongoDB document.

        Parameters
        ----------
        collection : str
            MongoDB collection name.
        document_id : Any
            Target document `_id`.
        text : str
            Source text to embed (and potentially chunk).
        embedding_field : str
            Field name to write vectors into (e.g., from `field_name()`).
        chunk_style : str, default CHUNK_STYLE_PARAGRAPH
            Chunking strategy.
        chunk_size : int, default 400
            Character window (used only when `chunk_style == "fixed"`).
        overlap : int, default 50
            Character overlap for fixed-size chunking.
        embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
            Embedding task type to use.
        output_dimensionality : int, default 768
            Informational value; the model controls actual size.
        include_vectors_in_result : bool, default True
            If True, include the vectors in the SafeResult payload.
        skip_if_present : bool, default True
            If True and the field already contains non-empty vectors, skip compute
            and return metadata indicating a cache hit.

        Returns
        -------
        SafeResult
            - `SafeResult.ok({...})` on success; payload includes counts, dims, styles,
              and optionally the vectors.
            - `SafeResult.fail("...")` on error (e.g., repo update failed).

        Notes
        -----
        - Uses `update_document` with `$set` to write vectors.
        - Ensures `matched_count > 0` so you get a clear error if the `_id` doesn't exist.
        """
        try:
            if skip_if_present:
                exists, existing_vectors = await self._load_existing_vectors(collection, document_id, embedding_field)
                if exists:
                    payload: Dict[str, Any] = {
                        "document_id": str(document_id),
                        "field": embedding_field,
                        "vectors_count": len(existing_vectors),
                        "dimensionality": (len(existing_vectors[0]) if existing_vectors else output_dimensionality),
                        "embedding_style": embedding_style,
                        "chunk_style": chunk_style,
                        "skipped_compute": True,
                        "from_cache": True,
                    }
                    if include_vectors_in_result:
                        payload["vectors"] = existing_vectors
                    return SafeResult.ok(payload)

            vectors = await self.get_embedding(
                text,
                embedding_style=embedding_style,
                chunk_style=chunk_style,
                chunk_size=chunk_size,
                overlap=overlap,
                output_dimensionality=output_dimensionality,
            )

            if not vectors or not vectors[0]:
                return SafeResult.fail("Embedding computation returned no vectors.")

            update = {"$set": {embedding_field: vectors}}
            up_res = await self.repo.update_document(collection, {"_id": document_id}, update)

            if not up_res.success or up_res.data.get("matched_count", 0) == 0:
                error_msg = up_res.error or f"Document with _id {document_id} not found in collection {collection}."
                return SafeResult.fail(f"Failed to save embeddings: {error_msg}")

            payload = {
                "document_id": str(document_id),
                "field": embedding_field,
                "vectors_count": len(vectors),
                "dimensionality": len(vectors[0]),
                "embedding_style": embedding_style,
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
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,  # informational only
        include_vectors_in_result: bool = False,
        skip_if_present: bool = True,
    ) -> SafeResult:
        """
        Convenience wrapper to derive the target field name and persist.

        Parameters
        ----------
        collection : str
            MongoDB collection name.
        document_id : Any
            Target document `_id`.
        base_field : str
            Logical source field (e.g., "text") for naming the embedding field.
        text : str
            Source text to embed.
        embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
            Embedding task type to use.
        chunk_style : str, default CHUNK_STYLE_PARAGRAPH
            Chunking strategy.
        chunk_size : int, default 400
            Character window for fixed-size chunking.
        overlap : int, default 50
            Overlap for fixed-size chunking.
        output_dimensionality : int, default 768
            Informational value; the model controls the actual size.
        include_vectors_in_result : bool, default False
            Include vectors in SafeResult payload if True.
        skip_if_present : bool, default True
            Skip compute if embedding field already exists with non-empty vectors.

        Returns
        -------
        SafeResult
            See :meth:`embed_and_store`.
        """
        target = field_name(base_field, embedding_style, chunk_style)
        return await self.embed_and_store(
            collection=collection,
            document_id=document_id,
            text=text,
            embedding_field=target,
            chunk_style=chunk_style,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_style=embedding_style,
            output_dimensionality=output_dimensionality,
            include_vectors_in_result=include_vectors_in_result,
            skip_if_present=skip_if_present,
        )

    async def embed_texts_batched(
        self,
        texts: Iterable[str],
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,  # informational only
    ) -> dict[str, List[List[float]]]:
        """
        Embed multiple texts and return a mapping of text → vectors.

        Parameters
        ----------
        texts : Iterable[str]
            A list (or any iterable) of strings to embed.
        embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
            Embedding task type to use.
        chunk_style : str, default CHUNK_STYLE_PARAGRAPH
            Chunking strategy (currently not applied in batching; see Notes).
        chunk_size : int, default 400
            Character window for fixed-size chunking (not used in batching).
        overlap : int, default 50
            Overlap for fixed-size chunking (not used in batching).
        output_dimensionality : int, default 768
            Informational value; the model controls the actual size.

        Returns
        -------
        dict[str, List[List[float]]]
            A mapping from the original text to a list of vectors.
            This implementation returns **one** vector per text (i.e., a list
            with a single vector), so the value looks like `[vector]` or `[]`.

        Notes
        -----
        - This batching helper currently assumes one vector per text. If you need
          true chunked embeddings for multiple texts, consider calling
          :meth:`get_embedding` per text, or extend this method to pre-chunk and
          reconcile N:1 mappings.
        """
        results: dict[str, List[List[float]]] = {}
        text_list = list(texts)
        if not text_list:
            return {}

        task_type_map = {
            EMBEDDING_STYLE_RETRIEVAL_QUERY: "RETRIEVAL_QUERY",
            EMBEDDING_STYLE_RETRIEVAL_DOCUMENT: "RETRIEVAL_DOCUMENT",
            EMBEDDING_STYLE_SEMANTIC_SIMILARITY: "SEMANTIC_SIMILARITY",
            EMBEDDING_STYLE_CLASSIFICATION: "CLASSIFICATION",
        }
        task_type = task_type_map.get(embedding_style, "RETRIEVAL_DOCUMENT")

        all_vectors = await self._get_gemini_embedding_batch(text_list, task_type)

        # One vector per input text for this simplified batching flow.
        for text, vectors in zip(text_list, all_vectors):
            results[text] = [vectors] if vectors else []
        return results


# ---------------------------------------------------------------------
# Simple demo (manual run)
# ---------------------------------------------------------------------

async def _demo() -> None:
    """
    Run a small end-to-end demonstration when executed as a script.

    The demo:
      1) Inserts a document with sample text.
      2) Embeds it using sentence chunking.
      3) Writes vectors into a deterministic field on the document.
      4) Demonstrates cache behavior by calling twice with `skip_if_present=True`.

    This function is intended for manual/local testing and is excluded from coverage.
    """
    embedder = ZEmbedder()
    DEMO_COLLECTION = "test"
    try:
        text = (
            "Artificial intelligence is transforming the legal industry. "
            "Lawyers now use AI for document review, case prediction, and drafting. "
            "These tools improve efficiency but also raise questions about ethics and accountability."
        )

        print("\n--- Persisting embeddings to Mongo ---")
        doc_id = ObjectId()
        ins = await embedder.repo.insert_document(
            DEMO_COLLECTION,
            {"_id": doc_id, "text": text}
        )
        assert ins.success, f"Insert failed: {ins.error}"

        target_field = field_name("text", "RETRIEVAL_DOCUMENT", "sentence")

        res1 = await embedder.embed_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            text=text,
            embedding_field=target_field,
            chunk_style=CHUNK_STYLE_SENTENCE,
            embedding_style="RETRIEVAL_DOCUMENT",
            include_vectors_in_result=False,
            skip_if_present=True,
        )
        print("Call #1 — saved OK?:", res1.success, "skipped?:", res1.data.get("skipped_compute") if res1.success else "N/A")

        res2 = await embedder.embed_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            text=text,
            embedding_field=target_field,
            chunk_style=CHUNK_STYLE_SENTENCE,
            embedding_style="RETRIEVAL_DOCUMENT",
            include_vectors_in_result=True,
            skip_if_present=True,
        )
        print("Call #2 — saved OK?:", res2.success, "skipped?:", res2.data.get("skipped_compute") if res2.success else "N/A")
        if res2.success:
            print("Returned vectors:", len(res2.data.get("vectors", [])))

    finally:
        embedder.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())
