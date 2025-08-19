# =============================================
# unified_vector_search.py  (drop-in replacement)
# =============================================
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_processing import SafeResult
from zmongo_toolbag.zmongo import ZMongo

# Optional HNSW acceleration
try:  # pragma: no cover
    import hnswlib
    _HNSW_AVAILABLE = True
except Exception:  # pragma: no cover
    _HNSW_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalVectorSearch:
    """
    Local cosine vector search over MongoDB docs that already store embeddings.

    All vector operations (coercion, normalization, rescoring) are centralized
    in this module to keep demos/tests simple and avoid drift.
    """

    def __init__(
        self,
        repository: ZMongo,
        collection: str,
        embedding_field: str,
        *,
        ttl_seconds: int = 300,
        id_field: str = "_id",
        chunked_embeddings: bool = True,
        # HNSW options
        use_hnsw: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
        # Re-ranking options
        exact_rescore: bool = False,
        re_rank_candidates: Optional[int] = None,
        re_rank_multiplier: float = 3.0,
    ):
        self.repo = repository
        self.collection = collection
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.ttl = ttl_seconds
        self.chunked = chunked_embeddings

        # HNSW controls
        self.use_hnsw = bool(use_hnsw and _HNSW_AVAILABLE)
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # Re-ranking controls
        self.exact_rescore = exact_rescore
        self.re_rank_candidates = re_rank_candidates
        self.re_rank_multiplier = re_rank_multiplier

        # Cache/index state
        self._lock = asyncio.Lock()
        self._built_at: float = 0.0
        self.emb_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[Any] = []   # keep original _id types (ObjectId or str)
        self._dim: int = 0

        self._hnsw_index = None

    # ---------------- Public static utilities (centralized here) ----------------

    @staticmethod
    def summarize_embedding_shape(emb_val: Any) -> str:
        """Human-friendly shape string for logging/debugging."""
        try:
            if hasattr(emb_val, "shape"):
                return f"array{tuple(getattr(emb_val, 'shape', ())) }"
            if isinstance(emb_val, dict):
                return f"dict(keys={list(emb_val.keys())})"
            if isinstance(emb_val, (list, tuple)):
                if not emb_val:
                    return "[]"
                first = emb_val[0]
                if isinstance(first, (list, tuple)):
                    inner = len(first) if isinstance(first, (list, tuple)) else "?"
                    return f"list[{len(emb_val)} x {inner}]"
                if isinstance(first, dict):
                    return f"list[{len(emb_val)} dicts]"
                return f"list[{len(emb_val)}]"
            return type(emb_val).__name__
        except Exception:
            return "<unknown>"

    @staticmethod
    def _to_1d_vector(v: Any) -> Optional[np.ndarray]:
        """Convert arbitrary v -> 1D float32 vector if possible."""
        try:
            arr = np.asarray(v, dtype=np.float32)
        except Exception:
            try:
                if isinstance(v, (list, tuple)):
                    arr = np.asarray([float(x) for x in v], dtype=np.float32)
                else:
                    return None
            except Exception:
                return None
        if arr.ndim == 1 and arr.size > 0:
            return arr
        if arr.ndim == 2 and 1 in arr.shape:
            arr = np.squeeze(arr)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        arr = arr.ravel()
        return arr if arr.size > 0 else None

    @staticmethod
    def coerce_to_chunk_matrix(emb_val: Any) -> Optional[np.ndarray]:
        """
        Coerce arbitrary stored embedding formats into a [C, D] float32 matrix:
          - Single vector -> shape (1, D)
          - List of vectors -> shape (C, D)
          - Extra nesting (e.g., [[vec]], [[[vec]]]) -> squeezed/flattened
          - Mixed lengths -> keep the most common dimension D, drop outliers
          - Stringified numbers -> coerced to float
        Returns None if nothing usable is found.
        """
        to_1d = LocalVectorSearch._to_1d_vector

        # numpy-like input
        if hasattr(emb_val, "tolist"):
            emb_val = emb_val.tolist()

        # Case A: single vector
        if isinstance(emb_val, (list, tuple, np.ndarray)):
            try:
                if all(not isinstance(x, (list, tuple, np.ndarray)) for x in emb_val):
                    vec = to_1d(emb_val)
                    if vec is not None:
                        return vec.astype(np.float32)[np.newaxis, :]
            except Exception:
                pass

            # Case B: list of vectors (possibly nested)
            vectors: List[np.ndarray] = []
            for item in emb_val:
                vec = to_1d(item)
                if vec is None and isinstance(item, (list, tuple, np.ndarray)) and len(item) == 1:
                    vec = to_1d(item[0])
                if vec is not None and vec.size > 0:
                    vectors.append(vec.astype(np.float32))
            if not vectors:
                return None

            lengths = [v.size for v in vectors]
            if not lengths:
                return None
            counts: Dict[int, int] = {}
            for L in lengths:
                counts[L] = counts.get(L, 0) + 1
            D = max(counts.items(), key=lambda kv: kv[1])[0]
            filtered = [v for v in vectors if v.size == D]
            if not filtered:
                return None
            try:
                M = np.vstack(filtered).astype(np.float32)
            except Exception:
                return None
            return M

        return None

    @staticmethod
    def coerce_query_vector(emb_val: Any) -> Optional[List[float]]:
        """Return a single L2-normalized vector (as list[float]) for searching from many stored formats."""
        M = LocalVectorSearch.coerce_to_chunk_matrix(emb_val)
        if M is None or M.size == 0:
            return None
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        norm_chunks = M / norms
        rep = norm_chunks.mean(axis=0)
        n = np.linalg.norm(rep)
        if n == 0.0:
            return None
        v = (rep / n).astype(np.float32).tolist()
        return v

    # ---------------- Internal helpers ----------------

    async def _find_all_with_embeddings(self) -> List[Dict[str, Any]]:
        res = await self.repo.find_documents(
            self.collection,
            {self.embedding_field: {"$exists": True}},
            limit=1_000_000,
        )
        if not res.success:
            raise RuntimeError(res.error)
        return res.data or []

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    async def _load_embeddings_matrix(self) -> Tuple[np.ndarray, List[Any]]:
        docs = await self._find_all_with_embeddings()
        rows: List[np.ndarray] = []
        ids: List[Any] = []
        for doc in docs:
            vec = LocalVectorSearch.coerce_query_vector(doc.get(self.embedding_field))
            if vec is None:
                continue
            rows.append(np.asarray(vec, dtype=np.float32))
            ids.append(doc.get(self.id_field))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.vstack(rows).astype(np.float32), ids

    def _build_hnsw(self, M: np.ndarray):
        if not self.use_hnsw:
            return
        if M.size == 0:
            self._hnsw_index = None
            return
        num_elements, dim = M.shape
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=self.hnsw_ef_construction, M=self.hnsw_m)
        index.add_items(M, np.arange(num_elements))
        index.set_ef(self.hnsw_ef_search)
        self._hnsw_index = index

    async def _ensure_index(self):
        async with self._lock:
            if (time.time() - self._built_at) < self.ttl and self.emb_matrix is not None:
                return
            M, ids = await self._load_embeddings_matrix()
            self.emb_matrix = M
            self.doc_ids = ids
            self._dim = 0 if M.size == 0 else M.shape[1]
            self._built_at = time.time()
            self._build_hnsw(M)

    async def _fetch_doc_by_index(self, i: int) -> Optional[Dict[str, Any]]:
        # Pass raw id back to ZMongo; it will normalize string/ObjectId automatically.
        doc_id = self.doc_ids[i]
        res = await self.repo.find_document(self.collection, {self.id_field: doc_id})
        return res.data if res and res.data else None

    def _numpy_topk(self, qn: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.emb_matrix @ qn
        k = min(top_k, scores.shape[0]) if top_k > 0 else 1
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def _hnsw_topk(self, qn: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._hnsw_index is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(top_k, self.emb_matrix.shape[0]) if top_k > 0 else 1
        labels, distances = self._hnsw_index.knn_query(qn[np.newaxis, :], k=k)
        scores = 1.0 - distances[0]
        return labels[0].astype(np.int64), scores.astype(np.float32)

    async def _rescore_candidates(self, qn: np.ndarray, cand_indices: np.ndarray) -> List[Tuple[int, float]]:
        """Sequential rescoring to avoid nested/concurrent async complexities."""
        scored: List[Tuple[int, float]] = []
        for i_raw in cand_indices.tolist():
            i = int(i_raw)
            doc = await self._fetch_doc_by_index(i)
            if not doc:
                continue
            emb_val = doc.get(self.embedding_field)
            if emb_val is None:
                continue
            try:
                M = LocalVectorSearch.coerce_to_chunk_matrix(emb_val)
                if M is None or M.size == 0:
                    continue
                norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
                s = float(np.max((M / norms) @ qn))
                scored.append((i, s))
            except Exception:
                continue
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # ---------------- Public API ----------------

    async def search(self, query_embedding: List[float], top_k: int) -> SafeResult:
        """
        Cosine search with Atlas-style candidate expansion and a simple inline fallback.
        - No nested async defs; no asyncio.gather.
        """
        await self._ensure_index()
        if self.emb_matrix is None or self.emb_matrix.size == 0:
            return SafeResult.ok([])

        # Coerce & sanitize query
        try:
            q = np.asarray([float(x) for x in query_embedding], dtype=np.float32)
        except Exception:
            return SafeResult.fail("Invalid query embedding: cannot convert to float array.")
        if q.ndim != 1 or q.size == 0:
            return SafeResult.fail("Invalid query embedding: expected 1-D non-empty vector.")
        if not np.isfinite(q).all():
            q = np.where(np.isfinite(q), q, 0.0).astype(np.float32)
        qn = self._normalize(q)
        if qn.size == 0 or not np.isfinite(qn).all():
            return SafeResult.fail("Invalid query embedding after normalization.")

        # Atlas-style candidate sizing (≈ numCandidates)
        dataset_size = self.emb_matrix.shape[0]
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 1
        top_k = min(top_k, dataset_size)

        if isinstance(self.re_rank_candidates, int) and self.re_rank_candidates > 0:
            first_k = self.re_rank_candidates
        else:
            default_candidates = top_k * 15
            first_k = max(top_k, default_candidates)
        first_k = min(max(1, first_k), dataset_size)

        try:
            if self.use_hnsw and self._hnsw_index is not None:
                idx, scores = self._hnsw_topk(qn, first_k)
            else:
                idx, scores = self._numpy_topk(qn, first_k)

            if idx.size == 0:
                return SafeResult.ok([])

            if self.exact_rescore:
                rescored = await self._rescore_candidates(qn, idx)
                final_pairs = rescored[:top_k]
                final_idx = [int(i) for (i, _) in final_pairs]
                final_scores = [float(s) for (_, s) in final_pairs]
            else:
                k = min(top_k, idx.size)
                final_idx = [int(i) for i in idx[:k].tolist()]
                final_scores = [float(s) for s in scores[:k].tolist()]

            results = []
            for i, s in zip(final_idx, final_scores):
                doc = await self._fetch_doc_by_index(i)
                if doc:
                    results.append({"retrieval_score": s, "document": doc})
            return SafeResult.ok(results)

        except Exception:
            # Inline manual fallback: scan all docs with embeddings and score max-over-chunks
            find_res = await self.repo.find_documents(self.collection, {self.embedding_field: {"$exists": True}})
            if not find_res.success:
                return find_res

            scored_docs: List[Dict[str, Any]] = []
            for doc in find_res.data or []:
                try:
                    M = LocalVectorSearch.coerce_to_chunk_matrix(doc.get(self.embedding_field))
                    if M is None or M.size == 0:
                        continue
                    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
                    s = float(np.max((M / norms) @ qn))
                    scored_docs.append({"retrieval_score": s, "document": doc})
                except Exception:
                    continue

            scored_docs.sort(key=lambda x: x["retrieval_score"], reverse=True)
            return SafeResult.ok(scored_docs[:top_k])


