"""
ZEmbedder — deterministic, async text embedder for ZMongo
=========================================================

This module provides a small, deterministic embedding utility that:

- Splits text into chunks (sentence / paragraph / fixed window)
- Produces embeddings in different *styles* (e.g., Retrieval‑Document vs. Retrieval‑Query)
- Persists vectors back into MongoDB via the `ZMongo` repository
- Returns results using a `SafeResult` wrapper for predictable error handling

It is designed to integrate cleanly with `LocalVectorSearch` for local cosine
search and with higher‑level retrievers. All public APIs are **async**.

Quick Start
-----------

```python
import asyncio
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    field_name,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    CHUNK_STYLE_SENTENCE,
)

COLL = "kb_docs"

async def main():
    repo = ZMongo()
    emb = ZEmbedder(repository=repo)

    # 1) Insert a document to embed
    _id = ObjectId()
    await repo.insert_document(COLL, {"_id": _id, "text": "Mitochondria are the powerhouse of the cell."})

    # 2) Persist sentence‑level Retrieval‑Document vectors under a consistent field name
    target = field_name("text", EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_SENTENCE)
    res = await emb.embed_and_store(
        collection=COLL,
        document_id=_id,
        text="Mitochondria are the powerhouse of the cell.",
        embedding_field=target,
        chunk_style=CHUNK_STYLE_SENTENCE,
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    )
    assert res.success, res.error

    # 3) Retrieve the updated document
    doc = await repo.find_document(COLL, {"_id": _id})
    print("Stored chunks:", len(doc.data[target]))

    emb.close()

asyncio.run(main())
```

Notes
-----
- Use **`EMBEDDING_STYLE_RETRIEVAL_DOCUMENT`** when storing document vectors and
  **`EMBEDDING_STYLE_RETRIEVAL_QUERY`** when building query vectors at retrieval time.
- The embedder is deterministic: the same text produces the same vector(s).
- Use `field_name(base, style, chunk_style)` to keep field names consistent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
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
    """Compose a consistent field name for persisted embeddings.

    The pattern is:
    `[BASE_FIELD]_[EMBEDDING_STYLE]_[CHUNK_STYLE]`

    Examples
    --------
    >>> field_name("text", EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_SENTENCE)
    'text_RETRIEVAL_DOCUMENT_sentence'
    >>> field_name("body", EMBEDDING_STYLE_SEMANTIC_SIMILARITY, CHUNK_STYLE_PARAGRAPH)
    'body_SEMANTIC_SIMILARITY_paragraph'

    Parameters
    ----------
    base_field : str
        Base field (e.g., "text").
    embedding_style : str
        Embedding style key, e.g. `EMBEDDING_STYLE_RETRIEVAL_DOCUMENT`.
    chunk_style : str
        Chunk granularity, e.g. `CHUNK_STYLE_SENTENCE`.

    Returns
    -------
    str
        The composed field name.
    """
    return f"{base_field}_{embedding_style}_{chunk_style}"


# ----------------------------
# Chunking utilities
# ----------------------------

def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping fixed-size windows.

    Parameters
    ----------
    text : str
        Input text.
    size : int
        Window size in characters. If <= 0, the function returns the full text.
    overlap : int
        Overlap between consecutive windows in characters. Negative values are treated as 0.

    Returns
    -------
    List[str]
        A list of window strings.

    Examples
    --------
    >>> _sliding_window("abcdef", size=4, overlap=2)
    ['abcd', 'cdef']
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
    """A minimal, deterministic sentence splitter.

    Splits on periods, retains a trailing period, and collapses newlines.
    Empty segments are removed.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        Sentence strings.

    Examples
    --------
    >>> _sentence_split("Hello world. New line.\nAnother.")
    ['Hello world.', 'New line.', 'Another.']
    """
    if not text:
        return []
    raw = [t.strip() for t in text.replace("\n", " ").split(".")]
    return [s + "." for s in raw if s]


def _paragraph_split(text: str) -> List[str]:
    """Split text on blank lines to form paragraphs.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        Paragraph strings (no empties).

    Examples
    --------
    >>> _paragraph_split("A\n\nB\n\n\nC")
    ['A', 'B', 'C']
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
    """Split text according to the specified chunking strategy.

    Parameters
    ----------
    text : str
        Input text to split.
    chunk_style : str, optional
        One of `CHUNK_STYLE_SENTENCE` (default), `CHUNK_STYLE_PARAGRAPH`, or
        `CHUNK_STYLE_FIXED`.
    chunk_size : int, optional
        Used only for `CHUNK_STYLE_FIXED`: window size in characters. Default 500.
    overlap : int, optional
        Used only for `CHUNK_STYLE_FIXED`: overlap size. Default 50.

    Returns
    -------
    List[str]
        The list of chunk strings.

    Examples
    --------
    >>> chunk_text("A. B. C.")
    ['A.', 'B.', 'C.']
    >>> chunk_text("A\n\nB", chunk_style=CHUNK_STYLE_PARAGRAPH)
    ['A', 'B']
    >>> chunk_text("abcdef", chunk_style=CHUNK_STYLE_FIXED, chunk_size=4, overlap=2)
    ['abcd', 'cdef']
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
    """Create a deterministic pseudo‑random unit vector from text.

    This helper hashes the `(text, dim)` pair into a seed, then uses a small
    normal distribution and renormalizes the resulting vector to roughly unit
    norm. It is suitable for tests and demos but **not** a substitute for real
    embedding models.

    Parameters
    ----------
    text : str
        The input text.
    dim : int
        Desired vector dimensionality.

    Returns
    -------
    numpy.ndarray
        A deterministic vector of shape `(dim,)`.
    """
    h = hashlib.sha256(f"{text}::{dim}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.05, size=dim).astype(np.float64)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    # small, fixed skew to keep tests from collapsing to identical values
    v[3] -= 0.1
    return v


def _style_bias(style: str, dim: int) -> np.ndarray:
    """Return a small, style‑specific bias vector.

    For retrieval, both `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` map to the
    same internal bias key so query and document vectors live in a compatible
    space.

    Parameters
    ----------
    style : str
        Embedding style key.
    dim : int
        Vector dimensionality.

    Returns
    -------
    numpy.ndarray
        A small bias vector of shape `(dim,)`.
    """
    # Map both retrieval styles to a shared key
    if style in {EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, EMBEDDING_STYLE_RETRIEVAL_QUERY}:
        style_key = "RET"
    else:
        style_key = {
            EMBEDDING_STYLE_SEMANTIC_SIMILARITY: "SS",
            EMBEDDING_STYLE_CLASSIFICATION: "CLF",
        }.get(style, "GEN")

    h = hashlib.md5(f"{style_key}:{dim}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "little")
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, 0.004, size=dim).astype(np.float64)  # smaller magnitude
    return bias


def _token_overlap_bias(text: str, dim: int, strength: float = 0.18) -> np.ndarray:
    """Compute a tiny lexical signal from token overlap.

    Deterministically hashes lowercase word tokens to indices and adds a
    normalized bump. This is a toy lexical prior to help tests and demos—feel
    free to disable or tune for your own use.

    Parameters
    ----------
    text : str
        Input text.
    dim : int
        Vector dimensionality.
    strength : float, optional
        Scale of the contribution. Default 0.18.

    Returns
    -------
    numpy.ndarray
        A vector of shape `(dim,)` representing lexical signal.
    """
    v = np.zeros(dim, dtype=np.float64)
    for tok in re.findall(r"\w+", text.lower()):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        v[idx] += 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v * strength


# ----------------------------
# Config structures
# ----------------------------
@dataclass
class EmbedConfig:
    """Convenience configuration for embedding calls.

    Attributes
    ----------
    embedding_style : str
        Style key, e.g., `EMBEDDING_STYLE_RETRIEVAL_DOCUMENT`.
    chunk_style : str
        Chunking mode: sentence, paragraph, or fixed.
    chunk_size : int
        Window size (for fixed) or soft size hint.
    overlap : int
        Overlap between windows (fixed mode only).
    output_dimensionality : int
        Target vector dimensionality (e.g., 768).
    """

    embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
    chunk_style: str = CHUNK_STYLE_SENTENCE
    chunk_size: int = 400
    overlap: int = 50
    output_dimensionality: int = DEFAULT_OUTPUT_DIM


# ----------------------------
# ZEmbedder
# ----------------------------
class ZEmbedder:
    """Embed text (by chunks) and optionally persist vectors into MongoDB.

    The class is intentionally *stateless*: callers pass collection / field
    parameters on each call. A `ZMongo` repository is used to perform updates
    when persisting results.

    Examples
    --------
    Basic usage to compute embeddings only:

    >>> import asyncio
    >>> from zmongo_toolbag.zembedder import ZEmbedder, CHUNK_STYLE_SENTENCE
    >>> async def demo():
    ...     emb = ZEmbedder()
    ...     vecs = await emb.get_embedding(
    ...         "Hello world. Testing.",
    ...         chunk_style=CHUNK_STYLE_SENTENCE,
    ...     )
    ...     print(len(vecs))
    >>> asyncio.run(demo())

    Persist vectors into MongoDB:

    >>> import asyncio
    >>> from bson import ObjectId
    >>> from zmongo_toolbag.zmongo import ZMongo
    >>> from zmongo_toolbag.zembedder import ZEmbedder, field_name, \
    ...     EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_PARAGRAPH
    >>> async def save_demo():
    ...     repo = ZMongo()
    ...     emb = ZEmbedder(repository=repo)
    ...     _id = ObjectId()
    ...     await repo.insert_document("kb", {"_id": _id, "text": "Legal AI improves review."})
    ...     target = field_name("text", EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_PARAGRAPH)
    ...     res = await emb.embed_and_store(
    ...         collection="kb",
    ...         document_id=_id,
    ...         text="Legal AI improves review.",
    ...         embedding_field=target,
    ...         chunk_style=CHUNK_STYLE_PARAGRAPH,
    ...     )
    ...     assert res.success, res.error
    >>> asyncio.run(save_demo())
    """

    def __init__(
        self,
        repository: Optional[ZMongo] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """Initialize the embedder.

        Parameters
        ----------
        repository : ZMongo, optional
            Existing repository instance. If omitted, an internal `ZMongo`
            is created and owned by this embedder (and closed by `close()`).
        gemini_api_key : str, optional
            Reserved for external providers; not used in the deterministic
            fallback implementation.
        """
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.gemini_api_key = gemini_api_key

    def close(self):
        """Close the internally‑owned `ZMongo` repository, if any."""
        if self._owns_repo and hasattr(self.repo, "close"):
            try:
                self.repo.close()
            except Exception:
                pass

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
        """Compute embeddings for a text, returning one vector per chunk.

        Parameters
        ----------
        text : str
            Input text to embed.
        embedding_style : str, optional
            Embedding style (e.g., `EMBEDDING_STYLE_RETRIEVAL_DOCUMENT`).
            When pairing with a retriever, store docs with RD and query with RQ.
        chunk_style : str, optional
            Chunk strategy: sentence, paragraph, or fixed.
        chunk_size : int, optional
            Window size for fixed chunking. Ignored otherwise.
        overlap : int, optional
            Overlap for fixed chunking.
        output_dimensionality : int, optional
            Size of each output vector (default 768).

        Returns
        -------
        List[List[float]]
            A list of vectors (one per chunk). Empty list if text yields no
            chunks.

        Examples
        --------
        >>> import asyncio
        >>> from zmongo_toolbag.zembedder import ZEmbedder, CHUNK_STYLE_SENTENCE
        >>> async def run():
        ...     e = ZEmbedder()
        ...     vecs = await e.get_embedding("A. B.", chunk_style=CHUNK_STYLE_SENTENCE)
        ...     print(len(vecs))  # 2
        >>> asyncio.run(run())
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
        collection: str,
        document_id,
        text: str,
        embedding_field: str,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
        include_vectors_in_result: bool = True,
    ) -> SafeResult:
        """Compute and persist embeddings for a specific MongoDB document.

        This performs an in‑place `$set` of `embedding_field` on the identified
        document. The document must already exist in the target collection.

        Parameters
        ----------
        collection : str
            MongoDB collection name.
        document_id : Any
            The document `_id` value.
        text : str
            Raw text to embed.
        embedding_field : str
            Field path to store vectors into (use `field_name(...)` to derive).
        chunk_style, chunk_size, overlap : see `get_embedding`.
        embedding_style, output_dimensionality : see `get_embedding`.
        include_vectors_in_result : bool, optional
            If True, include vectors in the `SafeResult.data` payload.

        Returns
        -------
        SafeResult
            On success, `.data` contains a summary payload with `document_id`,
            `field`, `vectors_count`, `dimensionality`, `embedding_style`, and
            `chunk_style` (and optionally `vectors`). On failure, `.error`
            contains a human‑readable message.

        Examples
        --------
        >>> import asyncio
        >>> from bson import ObjectId
        >>> from zmongo_toolbag.zmongo import ZMongo
        >>> from zmongo_toolbag.zembedder import ZEmbedder, field_name, \
        ...     EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_SENTENCE
        >>> async def store_demo():
        ...     repo = ZMongo()
        ...     emb = ZEmbedder(repository=repo)
        ...     _id = ObjectId()
        ...     await repo.insert_document("kb", {"_id": _id, "text": "AI and law."})
        ...     target = field_name("text", EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, CHUNK_STYLE_SENTENCE)
        ...     res = await emb.embed_and_store(
        ...         collection="kb",
        ...         document_id=_id,
        ...         text="AI and law.",
        ...         embedding_field=target,
        ...         chunk_style=CHUNK_STYLE_SENTENCE,
        ...     )
        ...     assert res.success, res.error
        >>> asyncio.run(store_demo())
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

            update = {"$set": {embedding_field: vectors}}
            up_res = await self.repo.update_document(collection, {"_id": document_id}, update)

            if not up_res.success or up_res.data.get("matched_count", 0) == 0:
                error_msg = up_res.error or f"Document with _id {document_id} not found in collection {collection}."
                return SafeResult.fail(f"Failed to save embeddings: {error_msg}")

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
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
        include_vectors_in_result: bool = False,
    ) -> SafeResult:
        """Convenience wrapper to derive the target field name and persist.

        This constructs the destination field name using `field_name(base_field,
        embedding_style, chunk_style)` and delegates to `embed_and_store(...)`.

        Parameters
        ----------
        base_field : str
            The logical base (e.g., "text"). The final field is derived via
            `field_name(base_field, embedding_style, chunk_style)`.
        Other parameters : see `embed_and_store`.

        Returns
        -------
        SafeResult
            See `embed_and_store` for details.

        Examples
        --------
        >>> import asyncio
        >>> from bson import ObjectId
        >>> from zmongo_toolbag.zmongo import ZMongo
        >>> from zmongo_toolbag.zembedder import ZEmbedder, EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
        >>> async def convenience_demo():
        ...     repo = ZMongo()
        ...     emb = ZEmbedder(repository=repo)
        ...     _id = ObjectId()
        ...     await repo.insert_document("kb", {"_id": _id, "text": "Paragraph one.\n\nParagraph two."})
        ...     res = await emb.embed_field_and_store(
        ...         collection="kb",
        ...         document_id=_id,
        ...         base_field="text",
        ...         text="Paragraph one.\n\nParagraph two.",
        ...     )
        ...     assert res.success, res.error
        >>> asyncio.run(convenience_demo())
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
        )

    async def embed_texts_batched(
        self,
        texts: Iterable[str],
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        output_dimensionality: int = DEFAULT_OUTPUT_DIM,
    ) -> dict[str, List[List[float]]]:
        """Embed multiple texts and return a mapping of text → vectors.

        Parameters
        ----------
        texts : Iterable[str]
            A collection of raw texts to embed.
        embedding_style, chunk_style, chunk_size, overlap, output_dimensionality
            See `get_embedding` for semantics.

        Returns
        -------
        dict[str, List[List[float]]]
            A dictionary mapping each input text to its list of vectors.

        Examples
        --------
        >>> import asyncio
        >>> from zmongo_toolbag.zembedder import ZEmbedder
        >>> async def batch_demo():
        ...     e = ZEmbedder()
        ...     out = await e.embed_texts_batched(["A.", "B."])
        ...     assert "A." in out and "B." in out
        ...     print(len(out["A."]))
        >>> asyncio.run(batch_demo())
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


async def _demo():
    """Run a small end‑to‑end demonstration when executed as a script.

    Steps
    -----
    1. Insert a document
    2. Persist sentence‑level Retrieval‑Document vectors to a derived field
    3. Fetch and inspect the updated document
    """
    embedder = ZEmbedder()
    DEMO_COLLECTION = "demo_embeddings"
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

        res = await embedder.embed_and_store(
            collection=DEMO_COLLECTION,
            document_id=doc_id,
            text=text,
            embedding_field=target_field,
            chunk_style=CHUNK_STYLE_SENTENCE,
            embedding_style="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        )
        print("Saved OK?:", res.success)
        if not res.success:
            print("Error:", res.error)

        got = await embedder.repo.find_document(DEMO_COLLECTION, {"_id": doc_id})
        assert got.success and got.data, f"Find failed: {got.error}"
        present = target_field in got.data
        print(f"Field '{target_field}' present in doc?:", present)
        if present:
            print("Stored chunk count:", len(got.data[target_field]))

    finally:
        embedder.close()


if __name__ == "__main__":
    asyncio.run(_demo())
