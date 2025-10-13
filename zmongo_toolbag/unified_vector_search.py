"""
LocalVectorSearch – SafeResult-compatible vector similarity search engine
for ZMongo repositories (sync or async).
"""

from __future__ import annotations
import asyncio
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from zmongo_toolbag.safe_result import SafeResult

logger = logging.getLogger(__name__)


class LocalVectorSearch:
    """
    Performs local in-memory vector search over embeddings stored in MongoDB.
    Compatible with ZMongo SafeResult-based synchronous repositories.
    """

    def __init__(
        self,
        repository,
        collection: str,
        embedding_field: str = "embeddings",
        chunked_embeddings: bool = False,
        score_mode: str = "cosine_0_1",
    ):
        self.repo = repository
        self.collection = collection
        self.embedding_field = embedding_field
        self.chunked_embeddings = chunked_embeddings
        self.score_mode = score_mode
        self._index_matrix = None
        self._meta_docs = []

    # ------------------------------------------------------------------
    # Internal SafeResult normalization
    # ------------------------------------------------------------------
    async def _await_repo_result(self, maybe_result):
        if asyncio.iscoroutine(maybe_result):
            maybe_result = await maybe_result
        if not isinstance(maybe_result, SafeResult):
            return SafeResult.ok(maybe_result)
        return maybe_result

    # ------------------------------------------------------------------
    # Main search entrypoint
    # ------------------------------------------------------------------
    async def search(self, query_vector: List[float], top_k: int = 5) -> SafeResult:
        """
        Perform cosine similarity search in the given collection.
        Returns SafeResult with a ranked list of hits.
        """
        try:
            # Load embedding matrix
            load_res = await self._load_embeddings_matrix()
            if not load_res.success:
                return load_res

            M, meta = load_res.data["matrix"], load_res.data["meta"]

            if M.size == 0:
                return SafeResult.ok([])

            query = np.array(query_vector, dtype=float)
            norms = np.linalg.norm(M, axis=1) * np.linalg.norm(query)
            valid = norms > 0
            sims = np.zeros(len(M))
            sims[valid] = (M[valid] @ query) / norms[valid]

            scores = [self._to_output_score(cos=s, dist=1 - s) for s in sims]
            ranked = sorted(
                zip(meta, scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            results = [
                {
                    "document": m,
                    "retrieval_score": float(score),
                }
                for m, score in ranked
            ]
            return SafeResult.ok(results)

        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            return SafeResult.fail(f"Vector search failed: {e}")

    # ------------------------------------------------------------------
    # Load all embeddings from MongoDB
    # ------------------------------------------------------------------
    async def _load_embeddings_matrix(self) -> SafeResult:
        """Loads all vectors and builds an in-memory matrix."""
        try:
            docs_res = await self._find_all_with_embeddings()
            if not docs_res.success:
                return docs_res

            docs = docs_res.data or []
            meta: List[Dict[str, Any]] = []
            vecs: List[List[float]] = []

            for d in docs:
                emb = d.get(self.embedding_field)
                if not emb:
                    continue
                # Handle chunked embeddings
                if self.chunked_embeddings and isinstance(emb[0], list):
                    for sub in emb:
                        vecs.append(sub)
                        meta.append(d)
                else:
                    vecs.append(emb)
                    meta.append(d)

            if not vecs:
                return SafeResult.ok({"matrix": np.zeros((0, 0)), "meta": []})

            M = np.array(vecs, dtype=float)
            self._index_matrix, self._meta_docs = M, meta
            return SafeResult.ok({"matrix": M, "meta": meta})

        except Exception as e:
            logger.exception("Failed to load embeddings: %s", e)
            return SafeResult.fail(f"load_embeddings_matrix failed: {e}")

    # ------------------------------------------------------------------
    # Retrieve documents with embeddings
    # ------------------------------------------------------------------
    async def _find_all_with_embeddings(self) -> SafeResult:
        """Fetch all documents that have the embedding field."""
        try:
            res = await self._await_repo_result(
                self.repo.find(
                    self.collection,
                    {self.embedding_field: {"$exists": True, "$ne": []}},
                    limit=1_000_000,
                )
            )
            if not res.success:
                return res
            return SafeResult.ok(res.data or [])
        except Exception as e:
            return SafeResult.fail(f"_find_all_with_embeddings failed: {e}")

    # ------------------------------------------------------------------
    # Score transformation
    # ------------------------------------------------------------------
    def _to_output_score(self, *, cos: float, dist: float) -> float:
        """Map raw cosine or distance values to unified [0, 1] score space."""
        mode = (self.score_mode or "cosine_0_1").lower()
        if mode in {"cosine", "cosine_1"}:
            return float(cos)
        if mode == "cosine_0_1":
            return 0.5 * (cos + 1.0)
        if mode in {"distance", "l2"}:
            return float(1.0 - dist)
        return float(cos)

    # ------------------------------------------------------------------
    # Index management helpers (optional)
    # ------------------------------------------------------------------
    async def rebuild_index(self) -> SafeResult:
        """Rebuild the in-memory vector index."""
        return await self._load_embeddings_matrix()

    def clear_index(self):
        """Clear the current index."""
        self._index_matrix = None
        self._meta_docs = []

    # ------------------------------------------------------------------
    # Utility: manual cosine search (optional)
    # ------------------------------------------------------------------
    def cosine_search_local(self, query_vector: List[float], top_k: int = 5) -> SafeResult:
        """Perform cosine similarity search on the current in-memory index."""
        if self._index_matrix is None or len(self._index_matrix) == 0:
            return SafeResult.fail("No in-memory index loaded.")

        try:
            query = np.array(query_vector, dtype=float)
            norms = np.linalg.norm(self._index_matrix, axis=1) * np.linalg.norm(query)
            valid = norms > 0
            sims = np.zeros(len(self._index_matrix))
            sims[valid] = (self._index_matrix[valid] @ query) / norms[valid]
            scores = [self._to_output_score(cos=s, dist=1 - s) for s in sims]
            ranked = sorted(
                zip(self._meta_docs, scores), key=lambda x: x[1], reverse=True
            )[:top_k]

            results = [
                {"document": m, "retrieval_score": float(score)} for m, score in ranked
            ]
            return SafeResult.ok(results)
        except Exception as e:
            return SafeResult.fail(f"cosine_search_local failed: {e}")
