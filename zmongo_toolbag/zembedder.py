"""
ZEmbedder – Unified Embedding Manager with SafeResult and ZMongo compatibility.
Fully supports synchronous ZMongo repositories and optional async Motor-style repos.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from zmongo_toolbag.safe_result import SafeResult

logger = logging.getLogger(__name__)


# --- Constants for embedding styles ---
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "retrieval.document"
EMBEDDING_STYLE_RETRIEVAL_QUERY = "retrieval.query"


class ZEmbedder:
    """
    Central class for managing text embeddings and persistence in MongoDB.
    Works with both sync (SafeResult-returning) and async repositories.
    """

    def __init__(self, repository, model=None):
        self.repository = repository
        self.model = model or self._load_default_model()
        logger.info("✅ ZEmbedder initialized with model: %s", getattr(self.model, "name", "unnamed"))

    # ----------------------------------------------------------------------
    # Internal helper to normalize SafeResult or coroutine
    # ----------------------------------------------------------------------
    async def _await_repo_result(self, maybe_result):
        """Normalize a repository result (SafeResult, coroutine, or dict)."""
        if asyncio.iscoroutine(maybe_result):
            maybe_result = await maybe_result
        if not isinstance(maybe_result, SafeResult):
            return SafeResult.ok(maybe_result)
        return maybe_result

    # ----------------------------------------------------------------------
    # Stub / load model
    # ----------------------------------------------------------------------
    def _load_default_model(self):
        """Placeholder for actual model loading (e.g., llama.cpp or OpenAI)."""
        class DummyModel:
            name = "dummy-embedder"
            async def embed(self, texts: List[str], **kwargs):
                # Return deterministic embeddings for testing
                return [[float(i % 3) for i, _ in enumerate(text)] for text in texts]
        return DummyModel()

    # ----------------------------------------------------------------------
    # Core embedding logic
    # ----------------------------------------------------------------------
    async def get_embedding(
        self,
        text: Optional[str] = None,
        *,
        collection: Optional[str] = None,
        document_id: Optional[Any] = None,
        embedding_style: Optional[str] = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        embedding_field: str = "embedding",
        text_field: str = "text",
        skip_if_present: bool = False,
        as_safe_result: bool = True,
        **kwargs,
    ):
        """
        Generate (and optionally store) embeddings for a given text or document.
        Fully compatible with SafeResult-based ZMongo.
        """

        doc = None

        # ------------------------------------------------------------
        # 1. Load text from document if not explicitly provided
        # ------------------------------------------------------------
        if not text and collection and document_id:
            fetch_result = await self._await_repo_result(
                self.repository.find_one(collection, {"_id": document_id})
            )
            if not fetch_result.success or not fetch_result.data:
                return SafeResult.fail(f"Could not load document: {fetch_result.error or 'No data'}")
            doc = fetch_result.data
            if text_field not in doc:
                return SafeResult.fail("Document text not provided or missing from source document.")
            text = doc[text_field]

        if not text:
            return SafeResult.fail("Document text not provided or missing from source document.")

        # ------------------------------------------------------------
        # 2. Skip if embeddings already exist and skip_if_present=True
        # ------------------------------------------------------------
        if skip_if_present and collection and document_id:
            existing_res = await self._await_repo_result(
                self.repository.find_one(collection, {"_id": document_id})
            )
            if existing_res.success:
                data = existing_res.data or {}
                if embedding_field in data and data[embedding_field]:
                    return SafeResult.ok({
                        "vectors": data[embedding_field],
                        "from_cache": True,
                        "dimensionality": len(data[embedding_field][0])
                    })

        # ------------------------------------------------------------
        # 3. Generate embeddings
        # ------------------------------------------------------------
        try:
            embed_data = await self._embed_texts([text], style=embedding_style)
        except Exception as e:
            logger.exception("Embedding model error: %s", e)
            return SafeResult.fail(f"Embedding model error: {e}")

        if not embed_data or "vectors" not in embed_data or not embed_data["vectors"]:
            return SafeResult.fail("No embedding vectors generated.")

        # ------------------------------------------------------------
        # 4. Save embeddings to MongoDB
        # ------------------------------------------------------------
        if collection and document_id:
            try:
                update_data = {embedding_field: embed_data["vectors"]}
                update_res = await self._await_repo_result(
                    self.repository.update_document(collection, {"_id": document_id}, update_data)
                )
                if not update_res.success:
                    logger.warning("Failed to save embeddings: %s", update_res.error)
            except Exception as e:
                logger.warning("Error saving embeddings: %s", e)

        # ------------------------------------------------------------
        # 5. Return SafeResult
        # ------------------------------------------------------------
        result_data = {
            "vectors": embed_data["vectors"],
            "dimensionality": len(embed_data["vectors"][0]),
            "from_cache": False,
        }
        return SafeResult.ok(result_data) if as_safe_result else result_data

    # ------------------------------------------------------------------
    # Sync helper wrapper for SafeResult-based integrations
    # ------------------------------------------------------------------
    def get_embedding_sync(self, *args, **kwargs):
        """
        Synchronous version of get_embedding().
        Runs the coroutine safely inside the ZEmbedder's event loop
        (or a temporary loop if none exists).

        Returns
        -------
        SafeResult
            Always returns a SafeResult containing either data or an error.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # If we're already inside the right loop, just run directly
        if loop and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.get_embedding(*args, **kwargs), loop)
            try:
                result = fut.result(timeout=30)
                return result if isinstance(result, SafeResult) else SafeResult.ok(result)
            except Exception as e:
                return SafeResult.fail(str(e))
        else:
            try:
                result = asyncio.run(self.get_embedding(*args, **kwargs))
                return result if isinstance(result, SafeResult) else SafeResult.ok(result)
            except Exception as e:
                return SafeResult.fail(str(e))


    # ----------------------------------------------------------------------
    # 6. Internal embedding engine
    # ----------------------------------------------------------------------
    async def _embed_texts(self, texts: List[str], style: Optional[str] = None) -> Dict[str, Any]:
        """Wrap model's embedding call into unified result structure."""
        vectors = await self.model.embed(texts, style=style)
        if not vectors:
            raise RuntimeError("Model returned no vectors.")
        if not isinstance(vectors[0], list):
            vectors = [vectors]
        return {"vectors": vectors, "style": style or EMBEDDING_STYLE_RETRIEVAL_DOCUMENT}

    # ----------------------------------------------------------------------
    # 7. Utility: cosine similarity
    # ----------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    # ----------------------------------------------------------------------
    # 8. Utility: batch embedding (optional)
    # ----------------------------------------------------------------------
    async def embed_many(self, texts: List[str], **kwargs) -> SafeResult:
        """Embed a list of texts in sequence, returning all results."""
        try:
            data = await self._embed_texts(texts, **kwargs)
            return SafeResult.ok(data)
        except Exception as e:
            return SafeResult.fail(f"embed_many failed: {e}")

    def close(self):
        """Graceful shutdown placeholder for compatibility."""
        try:
            if hasattr(self, "model") and hasattr(self.model, "close"):
                self.model.close()
        except Exception:
            pass
