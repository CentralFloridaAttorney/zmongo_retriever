"""
ZEmbedder — single‑method API (get_embedding) with transparent caching/persistence
=================================================================================

Design goals
------------
- **Only one public method**: `get_embedding(...)`.
- **ZRetriever‑compatible**: `get_embedding(text, embedding_style=...)` works exactly
  as before for queries (default return is `[single_vector]`).
- **ZMongo‑aware**: when `collection + document_id + embedding_field` are provided
  and `embedding_style` is for documents, the method **checks for existing vectors**
  and **stores new ones** if missing — all behind the scenes.
- **SafeResult-aware output**: you can opt into a `SafeResult` return shape via
  `as_safe_result=True`. By default, QUERY returns vectors; DOCUMENT returns
  `SafeResult` unless you override with `as_safe_result=False`.

Usage
-----
**Query (no persistence, vectors list):**

```python
emb = ZEmbedder()
qv = await emb.get_embedding(
    "powerhouse of the cell?",
    embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY,
)
# qv -> [[float, float, ...]]; use qv[0]
```

**Document (auto‑persist if missing, SafeResult by default):**
```python
res = await emb.get_embedding(
    text="Mitochondria are the powerhouse of the cell.",
    embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    collection="cases",
    document_id=some_id,
    embedding_field="embeddings",
    chunk_style=CHUNK_STYLE_PARAGRAPH,
)
# res is SafeResult; res.data contains {"text": ..., "vectors": [...]}.
```

"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

# llama-cpp (optional import error gets raised later on init if missing)
try:  # pragma: no cover
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None  # type: ignore

# Flexible local/packaged imports for repo + SafeResult
try:
    from zmongo_toolbag.zmongo import ZMongo  # type: ignore
except Exception:  # pragma: no cover
    from zmongo import ZMongo  # type: ignore

try:
    from zmongo_toolbag.data_processing import SafeResult  # type: ignore
except Exception:  # pragma: no cover
    from data_processing import SafeResult  # type: ignore

# Optional local env files
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants expected by ZRetriever
# ---------------------------------------------------------------------
EMBEDDING_STYLE_RETRIEVAL_QUERY = "retrieval_query"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "retrieval_document"

# Chunking styles
CHUNK_STYLE_FIXED = "fixed"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_PARAGRAPH = "paragraph"

__all__ = [
    "ZEmbedder",
    "EMBEDDING_STYLE_RETRIEVAL_QUERY",
    "EMBEDDING_STYLE_RETRIEVAL_DOCUMENT",
    "CHUNK_STYLE_FIXED",
    "CHUNK_STYLE_SENTENCE",
    "CHUNK_STYLE_PARAGRAPH",
]

# ---------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------

def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    size = max(1, int(size or 1))
    overlap = max(0, min(int(overlap or 0), size - 1))
    out: List[str] = []
    i, n, step = 0, len(text), size - overlap if size > overlap else 1
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j == n:
            break
        i += step
    return out


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    parts = [t.strip() for t in text.replace("", " ").split(".")]
    return [p + "." for p in parts if p]


def _paragraph_split(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("")]
    return [p for p in parts if p]


def _chunk_text(
    text: str,
    *,
    chunk_style: str = CHUNK_STYLE_SENTENCE,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[str]:
    style = (chunk_style or CHUNK_STYLE_SENTENCE).lower()
    if style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    return _sentence_split(text)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class EmbedConfig:
    chunk_style: str = CHUNK_STYLE_SENTENCE
    chunk_size: int = 400
    overlap: int = 50

# ---------------------------------------------------------------------
# ZEmbedder (single public method)
# ---------------------------------------------------------------------

class ZEmbedder:
    """Local llama-cpp embedder compatible with ZRetriever & ZMongo.

    Only the `get_embedding(...)` method is public. For **query** style it
    computes a single vector (returned as `[vec]` by default). For **document**
    style it (optionally) checks/stores vectors on an existing MongoDB document
    when `collection`, `document_id`, and `embedding_field` are provided.

    Parameters
    ----------
    repository : ZMongo | None
        Repository used for persistence. Created if omitted.
    model_path : str | None
        Path to a GGUF model with embedding support. Falls back to
        `LLAMA_MODEL_PATH` environment variable.
    """

    def __init__(self, *, repository: Optional[ZMongo] = None, model_path: Optional[str] = None) -> None:
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None

        env_path = os.getenv("LLAMA_MODEL_PATH")
        raw_path = model_path or env_path
        if not raw_path:
            raise FileNotFoundError(
                "LLAMA_MODEL_PATH not provided and environment variable not set."
            )
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = Path.home() / p
        self.model_path = str(p)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Llama model not found at: {self.model_path}")
        if Llama is None:
            raise ImportError("llama-cpp-python is required: pip install llama-cpp-python")

        logger.info("Loading Llama embedding model from: %s", self.model_path)
        self.model = Llama(model_path=self.model_path, embedding=True, verbose=False, n_ctx=2048)
        logger.info("Llama model loaded.")

    # lifecycle
    def close(self) -> None:
        if self._owns_repo and hasattr(self.repo, "close"):
            try:
                self.repo.close()
            except Exception:
                pass

    # ----------------------------- internals --------------------------------
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self.model.embed, texts)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logger.error("Embedding call failed: %s", e)
            return [[] for _ in texts]

    async def _load_existing(self, collection: str, document_id: Any, embedding_field: str) -> Tuple[bool, List[List[float]], Optional[dict]]:
        try:
            res = await self.repo.find_document(collection, {"_id": document_id})
            if res.success and res.data:
                doc = res.data
                val = doc.get(embedding_field)
                if isinstance(val, list) and val and all(isinstance(x, list) for x in val):
                    return True, val, doc
                return False, [], doc
            return False, [], None
        except Exception:
            return False, [], None

    async def _save_embeddings(self, collection: str, document_id: Any, field: str, vectors: List[List[float]]) -> SafeResult:
        try:
            update = {"$set": {field: vectors}}
            return await self.repo.update_document(collection, {"_id": document_id}, update)
        except Exception as e:  # pragma: no cover
            return SafeResult.fail(str(e), exc=e)

    # ----------------------------- single public API -------------------------
    async def get_embedding(
        self,
        text: Optional[str] = None,
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_QUERY,
        # Optional persistence context (only used for DOCUMENT style)
        collection: Optional[str] = None,
        document_id: Any = None,
        embedding_field: Optional[str] = None,
        text_field: str = "text",
        # Chunking (used for DOCUMENT style)
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        skip_if_present: bool = True,
        # Return-shape control: None => QUERY->vectors, DOCUMENT->SafeResult
        as_safe_result: Optional[bool] = None,
    ) -> Any:
        """Return embeddings for queries or documents.

        **Return type policy**
        - If `as_safe_result is True` => returns `SafeResult` with keys: `text`, `vectors`,
          and metadata (`embedding_style`, `chunk_style`, etc.).
        - If `as_safe_result is False` => returns `List[List[float]]`.
        - If `as_safe_result is None` => QUERY returns vectors; DOCUMENT returns `SafeResult`.

        Document style will auto-persist when `collection`, `document_id`, and
        `embedding_field` are provided.
        """
        style = (embedding_style or EMBEDDING_STYLE_RETRIEVAL_QUERY).lower()

        # decide default return shape
        if as_safe_result is None:
            as_safe_result = (style == EMBEDDING_STYLE_RETRIEVAL_DOCUMENT)

        # -------------------- QUERY (no persistence) ------------------------
        if style == EMBEDDING_STYLE_RETRIEVAL_QUERY:
            if not text:
                msg = "Query get_embedding called without text."
                return SafeResult.fail(msg) if as_safe_result else []
            vectors = await self._embed_batch([text])
            if as_safe_result:
                return SafeResult.ok({
                    "embedding_style": EMBEDDING_STYLE_RETRIEVAL_QUERY,
                    "text": text,
                    "vectors": vectors,
                    "vectors_count": len(vectors),
                    "dimensionality": (len(vectors[0]) if vectors else 0),
                })
            return vectors

        # -------------------- DOCUMENT (chunk + optional persistence) ------
        doc_context_ok = bool(collection and document_id is not None and embedding_field)
        cached_doc: Optional[dict] = None

        if doc_context_ok and skip_if_present:
            exists, existing, cached_doc = await self._load_existing(collection, document_id, embedding_field)  # type: ignore[arg-type]
            if exists:
                payload = {
                    "embedding_style": EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                    "text": (text if text is not None else (cached_doc or {}).get(text_field)),
                    "vectors": existing,
                    "vectors_count": len(existing),
                    "dimensionality": (len(existing[0]) if existing else 0),
                    "from_cache": True,
                    "skipped_compute": True,
                    "collection": collection,
                    "document_id": str(document_id),
                    "embedding_field": embedding_field,
                    "chunk_style": chunk_style,
                }
                return SafeResult.ok(payload) if as_safe_result else existing

            if text is None and cached_doc is not None:
                text = cached_doc.get(text_field)

        if not text:
            msg = "Document get_embedding called without text (and not found on doc)."
            return SafeResult.fail(msg) if as_safe_result else []

        chunks = _chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return SafeResult.ok({
                "embedding_style": EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                "text": text,
                "vectors": [],
                "vectors_count": 0,
                "dimensionality": 0,
                "from_cache": False,
                "skipped_compute": False,
                "collection": collection,
                "document_id": (str(document_id) if document_id is not None else None),
                "embedding_field": embedding_field,
                "chunk_style": chunk_style,
            }) if as_safe_result else []

        vectors = await self._embed_batch(chunks)
        if not vectors or not (isinstance(vectors[0], list) and vectors[0]):
            return SafeResult.fail("Embedding computation returned no vectors.") if as_safe_result else []

        if doc_context_ok:
            save_res = await self._save_embeddings(collection, document_id, embedding_field, vectors)  # type: ignore[arg-type]
            if not save_res.success:
                logger.error("Failed to save embeddings to %s/%s: %s", collection, document_id, save_res.error)

        payload = {
            "embedding_style": EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            "text": text,
            "vectors": vectors,
            "vectors_count": len(vectors),
            "dimensionality": len(vectors[0]),
            "from_cache": False,
            "skipped_compute": False,
            "collection": collection,
            "document_id": (str(document_id) if document_id is not None else None),
            "embedding_field": embedding_field,
            "chunk_style": chunk_style,
        }
        return SafeResult.ok(payload) if as_safe_result else vectors


# --------------------------- demo (manual) ------------------------------------
async def _demo() -> None:  # pragma: no cover
    from bson import ObjectId  # type: ignore

    if not os.getenv("LLAMA_MODEL_PATH"):
        print("Set LLAMA_MODEL_PATH to a valid GGUF file before running the demo.")
        return

    emb = ZEmbedder()
    coll = "zembedder_demo"

    # Fresh doc
    doc_id = ObjectId()
    await emb.repo.delete_document(coll, {"_id": doc_id})
    await emb.repo.insert_document(coll, {"_id": doc_id, "text": "Mitochondria are the powerhouse of the cell."})

    # DOCUMENT style: persisted if missing (SafeResult default)
    res = await emb.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=coll,
        document_id=doc_id,
        embedding_field="embeddings",
        text_field="text",
        chunk_style=CHUNK_STYLE_SENTENCE,
    )
    print("Doc vectors:", res.success, res.data.get("vectors_count") if res.success else res.error)  # type: ignore

    # QUERY style: transient (vectors list default)
    qv = await emb.get_embedding("powerhouse of the cell?", embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY)
    print("Query vector dims:", len(qv[0]) if qv else 0)

    # Document embedding — returns SafeResult with both text and vectors (default)
    res = await emb.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection="cases",
        document_id=doc_id,
        embedding_field="embeddings",
        text_field="text",
        text="Mitochondria are the powerhouse of the cell.",
    )

    print(res.data["text"][:80], "...")  # the text used
    print(len(res.data["vectors"]), "chunks")  # the vectors

    emb.close()




if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(_demo())
