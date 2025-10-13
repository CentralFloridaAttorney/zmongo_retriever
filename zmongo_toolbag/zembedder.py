from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

try:
    from llama_cpp import Llama  # type: ignore
except ImportError:
    Llama = None  # type: ignore

try:
    from bson import ObjectId  # type: ignore
except ImportError:
    ObjectId = None

from zmongo_toolbag.data_processing import SafeResult
from zmongo_toolbag.zmongo import ZMongo

load_dotenv(Path.home() / ".resources" / ".env")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

EMBEDDING_STYLE_RETRIEVAL_QUERY = "retrieval_query"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "retrieval_document"
CHUNK_STYLE_PARAGRAPH = "paragraph"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_FIXED = "fixed"

__all__ = [
    "ZEmbedder", "EMBEDDING_STYLE_RETRIEVAL_QUERY", "EMBEDDING_STYLE_RETRIEVAL_DOCUMENT",
    "CHUNK_STYLE_PARAGRAPH", "CHUNK_STYLE_SENTENCE", "CHUNK_STYLE_FIXED",
]


def _paragraph_split(text: str) -> List[str]:
    if not text:
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    if size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += (size - overlap)
    return chunks


def _chunk_text(text: str, *, chunk_style: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if chunk_style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    return _paragraph_split(text)


class ZEmbedder:
    def __init__(
        self,
        *,
        repository: Optional[ZMongo] = None,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
    ) -> None:
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None

        if Llama is None:
            raise ImportError("`pip install llama-cpp-python` is required for embeddings.")

        env_var_path = os.getenv("EMBEDDING_MODEL_PATH")
        model_base = Path.home() / env_var_path if env_var_path else None
        raw_path = model_path or model_base
        if not raw_path:
            raise FileNotFoundError("model_path or EMBEDDING_MODEL_PATH environment variable required.")

        self.model_path = str(Path(raw_path).expanduser().resolve())
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Embedding model not found at {self.model_path}")

        logger.info("Loading Llama model from: %s", self.model_path)
        self.model = Llama(
            model_path=self.model_path, embedding=True,
            verbose=False, n_ctx=n_ctx, n_gpu_layers=-1
        )
        logger.info("Llama model loaded successfully.")

    def close(self) -> None:
        if self._owns_repo:
            self.repo.close()

    async def _embed_batch(self, texts: List[str]) -> SafeResult:
        if not texts:
            return SafeResult.ok([])
        try:
            loop = asyncio.get_running_loop()
            vectors = await loop.run_in_executor(None, self.model.embed, texts)
            if not vectors or not all(isinstance(v, list) for v in vectors):
                return SafeResult.fail("Malformed embedding output.")
            return SafeResult.ok(vectors)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return SafeResult.fail("Embedding call failed", exc=e)

    def _build_payload(self, *, text: str, vectors: List[List[float]], style: str, from_cache: bool, **kwargs: Any) -> dict:
        return {
            "embedding_style": style,
            "text": text,
            "vectors": vectors,
            "vectors_count": len(vectors),
            "dimensionality": len(vectors[0]) if vectors and vectors[0] else 0,
            "from_cache": from_cache,
            "skipped_compute": from_cache,
            **kwargs,
        }

    async def _process_document_embedding(
        self,
        text: Optional[str],
        collection: str,
        doc_id: Any,
        field: str,
        text_field: str,
        chunk_style: str,
        chunk_size: int,
        overlap: int,
        skip: bool,
    ) -> SafeResult:
        meta = {"collection": collection, "document_id": str(doc_id), "embedding_field": field,
                "chunk_style": chunk_style}

        if skip:
            find_res = self.repo.find_document(collection, {"_id": doc_id})
            if find_res.success and find_res.data:
                cached_doc = find_res.data
                existing_vectors = cached_doc.get(field)
                if isinstance(existing_vectors, list) and existing_vectors:
                    final_text = text or cached_doc.get(text_field, "")
                    payload = self._build_payload(
                        text=final_text, vectors=existing_vectors,
                        style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                        from_cache=True, **meta
                    )
                    return SafeResult.ok(payload)
                if text is None:
                    text = cached_doc.get(text_field)
            elif not find_res.success:
                return SafeResult.fail(f"Lookup failed: {find_res.error}", exc=find_res.original())

        if not text:
            return SafeResult.fail("Document text not provided or missing from source document.")

        chunks = _chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return SafeResult.ok(self._build_payload(text=text, vectors=[], style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                                                     from_cache=False, **meta))

        embed_res = await self._embed_batch(chunks)
        if not embed_res.success:
            return embed_res
        vectors = embed_res.data

        save_res = self.repo.update_document(collection, {"_id": doc_id}, {"$set": {field: vectors}})
        if not save_res.success:
            logger.error("Failed to save embeddings to %s/%s: %s", collection, doc_id, save_res.error)
            meta["save_error"] = save_res.error

        return SafeResult.ok(self._build_payload(text=text, vectors=vectors,
                                                 style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                                                 from_cache=False, **meta))

    async def get_embedding(
        self,
        text: Optional[str] = None,
        *,
        embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_QUERY,
        collection: Optional[str] = None,
        document_id: Any = None,
        embedding_field: Optional[str] = None,
        text_field: str = "text",
        chunk_style: str = CHUNK_STYLE_PARAGRAPH,
        chunk_size: int = 512,
        overlap: int = 50,
        skip_if_present: bool = True,
        as_safe_result: Optional[bool] = None,
    ) -> Any:
        style = (embedding_style or EMBEDDING_STYLE_RETRIEVAL_QUERY).lower()
        if as_safe_result is None:
            as_safe_result = (style == EMBEDDING_STYLE_RETRIEVAL_DOCUMENT)

        if style == EMBEDDING_STYLE_RETRIEVAL_QUERY:
            if not text:
                return SafeResult.fail("Query text cannot be empty.") if as_safe_result else []
            embed_res = await self._embed_batch([text])
            if not embed_res.success:
                return embed_res if as_safe_result else []
            vectors = embed_res.data
            if as_safe_result:
                return SafeResult.ok(self._build_payload(text=text, vectors=vectors, style=style, from_cache=False))
            return vectors

        if not (collection and document_id is not None and embedding_field):
            return SafeResult.fail("Document embedding requires collection, document_id, and embedding_field.")

        result = await self._process_document_embedding(
            text, collection, document_id, embedding_field,
            text_field, chunk_style, chunk_size, overlap, skip_if_present
        )

        if as_safe_result:
            return result
        return result.data.get("vectors", []) if result.success else []
