"""
ZEmbedder — modular, single‑method API (get_embedding) with pluggable backends
===============================================================================

Goals
-----
- **One public method**: `get_embedding(...)` (query + document styles).
- **Modular backends**: switch between **Gemini**, **Llama (llama‑cpp)**, or
  **HuggingFace SentenceTransformers** via a constructor arg or env var.
- **ZRetriever compatibility**: constants `EMBEDDING_STYLE_RETRIEVAL_QUERY` and
  `EMBEDDING_STYLE_RETRIEVAL_DOCUMENT` exposed; query returns `[single_vector]`.
- **ZMongo‑aware**: when `collection + document_id + embedding_field` are given,
  the class checks for existing vectors and stores new ones if missing.
- **SafeResult inside**: ZMongo calls use your SafeResult wrapper; the public
  method still returns **raw vectors** (`List[List[float]]`).

Quick usage
-----------
```python
# Choose a backend: "gemini" | "llama" | "hf" (or pass a backend instance)
emb = ZEmbedder(backend_name="llama", backend_kwargs={"model_path": "C:/.../model.gguf"})
# Query style → no chunking, one vector
qv = await emb.get_embedding("what is consideration?", embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY)
# Document style → chunk + optional persistence
vecs = await emb.get_embedding(
    text=None,  # read from doc["text"] automatically
    embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    collection="cases", document_id=some_id,
    embedding_field="embeddings", text_field="text",
    chunk_style=CHUNK_STYLE_PARAGRAPH,
)
```
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Flexible local/packaged imports for repo + SafeResult
try:
    from zmongo_toolbag.zmongo import ZMongo  # type: ignore
except Exception:  # pragma: no cover
    from zmongo import ZMongo  # type: ignore

try:
    from zmongo_toolbag.data_processing import SafeResult  # type: ignore
except Exception:  # pragma: no cover
    from data_processing import SafeResult  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants (kept for ZRetriever compatibility)
# ---------------------------------------------------------------------
EMBEDDING_STYLE_RETRIEVAL_QUERY = "retrieval_query"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "retrieval_document"
# Optional/aux styles a backend may support (not required by retriever)
EMBEDDING_STYLE_SEMANTIC_SIMILARITY = "semantic_similarity"
EMBEDDING_STYLE_CLASSIFICATION = "classification"

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
# Chunking utilities (shared across backends)
# ---------------------------------------------------------------------

def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text] if text else []
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = size - 1 if size > 1 else 0

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
    if not text:
        return []
    raw = [t.strip() for t in text.replace("\n", " ").split(".")]
    return [s + "." for s in raw if s]


def _paragraph_split(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")]
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
    # default sentence
    return _sentence_split(text)

# ---------------------------------------------------------------------
# Backend protocol + factory
# ---------------------------------------------------------------------

@dataclass
class BackendInfo:
    name: str
    model_id: str | None = None
    dims: Optional[int] = None


class BaseEmbedBackend(ABC):
    """Abstract backend interface used by ZEmbedder.

    Backends should implement **batch** embedding. `embedding_style` is passed
    through (some backends may ignore it). Should return one vector per input
    string. Never raise on recoverable errors — return `[[] ...]` aligned to
    inputs so callers can handle gracefully.
    """

    def __init__(self) -> None:
        self.info = BackendInfo(name=self.__class__.__name__)

    @abstractmethod
    async def embed_batch(self, texts: List[str], *, embedding_style: str) -> List[List[float]]:
        ...


class GeminiBackend(BaseEmbedBackend):
    def __init__(self, api_key: Optional[str] = None, model: str = "embedding-001") -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL", "embedding-001")
        self.info = BackendInfo(name="gemini", model_id=self.model, dims=768)
        if not self.api_key:
            logger.warning("GeminiBackend initialized without GEMINI_API_KEY; calls will return empty vectors.")

    async def embed_batch(self, texts: List[str], *, embedding_style: str) -> List[List[float]]:
        if not texts:
            return []
        if not self.api_key:
            return [[] for _ in texts]

        # Map our style names to Gemini taskType
        style = (embedding_style or EMBEDDING_STYLE_RETRIEVAL_DOCUMENT).lower()
        task_map = {
            EMBEDDING_STYLE_RETRIEVAL_QUERY: "RETRIEVAL_QUERY",
            EMBEDDING_STYLE_RETRIEVAL_DOCUMENT: "RETRIEVAL_DOCUMENT",
            EMBEDDING_STYLE_SEMANTIC_SIMILARITY: "SEMANTIC_SIMILARITY",
            EMBEDDING_STYLE_CLASSIFICATION: "CLASSIFICATION",
        }
        task_type = task_map.get(style, "RETRIEVAL_DOCUMENT")

        api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents?key={self.api_key}"
        )
        payload = {
            "requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}, "taskType": task_type}
                for t in texts
            ]
        }

        # Use requests in a worker thread with simple retries
        for i in range(5):
            try:
                import requests  # local import to avoid hard dependency when unused

                resp = await asyncio.to_thread(
                    lambda: requests.post(api_url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
                )
                resp.raise_for_status()
                data = resp.json()
                return [emb["values"] for emb in data.get("embeddings", [])]
            except Exception as e:
                logger.warning("Gemini API call failed (attempt %d): %s", i + 1, e)
                if i < 4:
                    await asyncio.sleep(2**i)
        return [[] for _ in texts]


class LlamaBackend(BaseEmbedBackend):
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048) -> None:
        super().__init__()
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.n_ctx = n_ctx
        self.info = BackendInfo(name="llama", model_id=(Path(self.model_path).name if self.model_path else None))
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                "LlamaBackend requires a valid GGUF model file; set LLAMA_MODEL_PATH or pass model_path"
            )
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("llama-cpp-python is required for LlamaBackend: pip install llama-cpp-python") from e

        logger.info("Loading Llama embedding model from: %s", self.model_path)
        self.model = Llama(model_path=self.model_path, embedding=True, verbose=False, n_ctx=self.n_ctx)
        logger.info("Llama model loaded.")

    async def embed_batch(self, texts: List[str], *, embedding_style: str) -> List[List[float]]:  # noqa: ARG002
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self.model.embed, texts)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logger.error("Llama embed failed: %s", e)
            return [[] for _ in texts]


class HFBackend(BaseEmbedBackend):
    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        self.model_path = model_path or os.getenv("HF_MODEL_PATH")
        if not self.model_path or not Path(self.model_path).is_dir():
            raise FileNotFoundError(
                "HFBackend requires a SentenceTransformer model directory; set HF_MODEL_PATH or pass model_path"
            )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("sentence-transformers is required for HFBackend: pip install sentence-transformers") from e

        self.model = SentenceTransformer(self.model_path)
        self.info = BackendInfo(name="hf", model_id=Path(self.model_path).name, dims=None)
        logger.info("SentenceTransformer model loaded from: %s", self.model_path)

    async def embed_batch(self, texts: List[str], *, embedding_style: str) -> List[List[float]]:  # noqa: ARG002
        if not texts:
            return []
        try:
            def _encode_sync():
                vecs = self.model.encode(texts, convert_to_tensor=False)
                return vecs.tolist()

            return await asyncio.to_thread(_encode_sync)
        except Exception as e:
            logger.error("HF encode failed: %s", e)
            return [[] for _ in texts]


BACKEND_ALIASES = {
    "gemini": GeminiBackend,
    "llama": LlamaBackend,
    "hf": HFBackend,
    "huggingface": HFBackend,
}


def make_backend(name: Optional[str], **kwargs) -> BaseEmbedBackend:
    name = (name or os.getenv("ZEMBEDDER_BACKEND", "gemini")).strip().lower()
    cls = BACKEND_ALIASES.get(name)
    if not cls:
        raise ValueError(f"Unknown embedder backend '{name}'. Options: {sorted(BACKEND_ALIASES)}")
    return cls(**kwargs)

# ---------------------------------------------------------------------
# ZEmbedder (single public method)
# ---------------------------------------------------------------------

class ZEmbedder:
    """Pluggable embedder with a single public method: `get_embedding(...)`.

    Parameters
    ----------
    repository : ZMongo | None
        ZMongo repository for persistence. Created if omitted.
    backend : BaseEmbedBackend | None
        A pre‑constructed backend instance (GeminiBackend, LlamaBackend, HFBackend, ...).
    backend_name : str | None
        If `backend` is not provided, build one using this name (e.g. "gemini", "llama", "hf").
    backend_kwargs : dict
        Extra kwargs passed to the backend constructor (e.g., model_path, api_key).
    """

    def __init__(
        self,
        *,
        repository: Optional[ZMongo] = None,
        backend: Optional[BaseEmbedBackend] = None,
        backend_name: Optional[str] = None,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None
        self.backend = backend or make_backend(backend_name, **(backend_kwargs or {}))

    # lifecycle
    def close(self) -> None:
        if self._owns_repo and hasattr(self.repo, "close"):
            try:
                self.repo.close()
            except Exception:
                pass

    # ----------------------------- internals --------------------------------
    async def _embed_batch(self, texts: List[str], *, embedding_style: str) -> List[List[float]]:
        return await self.backend.embed_batch(texts, embedding_style=embedding_style)

    async def _load_existing(
        self, collection: str, document_id: Any, embedding_field: str
    ) -> Tuple[bool, List[List[float]], Optional[dict]]:
        try:
            res = await self.repo.find_document(collection, {"_id": document_id})
            if res.success and res.data:
                doc = res.data
                val = doc.get(embedding_field)
                if isinstance(val, list) and val and all(isinstance(x, (list, tuple)) for x in val):
                    return True, [list(x) for x in val], doc
                return False, [], doc
            return False, [], None
        except Exception:
            return False, [], None

    async def _save_embeddings(
        self, collection: str, document_id: Any, field: str, vectors: List[List[float]]
    ) -> SafeResult:
        try:
            update = {"$set": {field: vectors}}
            return await self.repo.update_document(collection, {"_id": document_id}, update)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)  # type: ignore[arg-type]

    @staticmethod
    def _derive_field_name(
        *, base_field: str, embedding_style: str, chunk_style: str, backend_info: BackendInfo
    ) -> str:
        # Stable, collision‑resistant naming across backends
        bname = backend_info.name
        model = backend_info.model_id or "model"
        return f"{base_field}_{embedding_style}_{chunk_style}_{bname}_{model}".replace(".", "_").replace("-", "_")

    # ----------------------------- single public API -------------------------
    async def get_embedding(
        self,
        text: Optional[str] = None,
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_QUERY,
        # Optional persistence context (used for DOCUMENT style)
        collection: Optional[str] = None,
        document_id: Any = None,
        embedding_field: Optional[str] = None,
        text_field: str = "text",
        # Chunking (used for DOCUMENT style)
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 400,
        overlap: int = 50,
        skip_if_present: bool = True,
        auto_name_field_when_missing: bool = True,
    ) -> List[List[float]]:
        """Return embeddings for queries or documents.

        Behavior
        ---------
        - **Query style** (`retrieval_query`):
          No chunking; returns a single vector in a list (`[vector]`).
        - **Document style** (`retrieval_document`):
          Chunks `text` (or `doc[text_field]` if `text is None`), embeds all chunks,
          and (when `collection+document_id+embedding_field` are provided) persists
          vectors into the document if missing.
        """
        style = (embedding_style or EMBEDDING_STYLE_RETRIEVAL_QUERY).lower()

        # Fast path: QUERY style -> single vector, no persistence
        if style == EMBEDDING_STYLE_RETRIEVAL_QUERY:
            if not text:
                logger.warning("Query get_embedding called without text; returning empty list.")
                return []
            return await self._embed_batch([text], embedding_style=style)

        # DOCUMENT style
        doc_context_ok = bool(collection and document_id is not None)

        # If we have persistence context, ensure we have a field (or compute one)
        if doc_context_ok and not embedding_field and auto_name_field_when_missing:
            base_field = text_field or "text"
            embedding_field = self._derive_field_name(
                base_field=base_field,
                embedding_style=style,
                chunk_style=(chunk_style or CHUNK_STYLE_PARAGRAPH),
                backend_info=self.backend.info,
            )

        # If persistence context and skip_if_present, check storage first
        if doc_context_ok and embedding_field and skip_if_present:
            exists, existing, doc = await self._load_existing(collection, document_id, embedding_field)
            if exists:
                return existing
            # If no text provided, read from document now
            if text is None and doc is not None:
                text = doc.get(text_field)

        # Prepare text
        if not text:
            logger.warning("Document get_embedding called without text (and no text_field value available); returning empty list.")
            return []

        # Chunk and embed
        chunks = _chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []
        vectors = await self._embed_batch(chunks, embedding_style=style)
        if not vectors or not (isinstance(vectors[0], list) and vectors[0]):
            return []

        # Maybe persist
        if doc_context_ok and embedding_field:
            save_res = await self._save_embeddings(collection, document_id, embedding_field, vectors)
            if not save_res.success:
                logger.error(
                    "Failed to save embeddings to %s/%s field '%s': %s",
                    collection,
                    document_id,
                    embedding_field,
                    save_res.error,
                )
        return vectors


# --------------------------- demo (manual) ------------------------------------
async def _demo() -> None:  # pragma: no cover
    from bson import ObjectId  # type: ignore

    # Example: choose backend from env ZEMBEDDER_BACKEND or override below
    emb = ZEmbedder(backend_name=os.getenv("ZEMBEDDER_BACKEND", "gemini"))
    coll = "zembedder_modular_demo"

    # Fresh doc
    doc_id = ObjectId()
    await emb.repo.delete_document(coll, {"_id": doc_id})
    await emb.repo.insert_document(coll, {"_id": doc_id, "text": "Mitochondria are the powerhouse of the cell."})

    # DOCUMENT style: persisted if missing
    _ = await emb.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=coll,
        document_id=doc_id,
        embedding_field=None,  # auto‑named using backend + styles
        text_field="text",
        chunk_style=CHUNK_STYLE_SENTENCE,
    )

    # QUERY style: transient
    qv = await emb.get_embedding("powerhouse of the cell?", embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY)
    print("Query vector dims:", len(qv[0]) if qv else 0)

    emb.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(_demo())
