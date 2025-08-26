# =============================================
# unified_vector_search.py  (drop-in replacement)
# =============================================
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bson import ObjectId

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.data_processing import SafeResult

# Optional HNSW acceleration
try:  # pragma: no cover
    import hnswlib

    _HNSW_AVAILABLE = True
except Exception:  # pragma: no cover
    _HNSW_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalVectorSearch:
    """
    Local cosine vector search over MongoDB docs that store chunked embeddings.
    This implementation builds an in-memory index of individual chunks for precise search.
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
            use_hnsw: bool = False,
            hnsw_m: int = 16,
            hnsw_ef_construction: int = 200,
            hnsw_ef_search: int = 200,
            exact_rescore: bool = True,
    ):
        self.repo = repository
        self.collection = collection
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.ttl = ttl_seconds
        self.chunked = chunked_embeddings

        self.use_hnsw = bool(use_hnsw and _HNSW_AVAILABLE)
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.exact_rescore = exact_rescore

        self._lock = asyncio.Lock()
        self._built_at: float = 0.0
        self.emb_matrix: Optional[np.ndarray] = None
        self.chunk_metadata: List[Tuple[Any, int]] = []
        self._dim: int = 0
        self._hnsw_index = None

    @staticmethod
    def coerce_to_chunk_matrix(emb_val: Any) -> Optional[np.ndarray]:
        if emb_val is None: return None
        try:
            arr = np.array(emb_val, dtype=np.float32)
            if arr.ndim == 1: return arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0: return arr
            return None
        except (ValueError, TypeError):
            logger.debug("Failed to coerce embedding value to a matrix.")
            return None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12) if norm > 0 else vec

    async def _find_all_with_embeddings(self) -> List[Dict[str, Any]]:
        res = await self.repo.find_documents(
            self.collection,
            {self.embedding_field: {"$exists": True, "$ne": []}},
            limit=1_000_000,
        )
        if not res.success: raise RuntimeError(res.error)
        return res.data or []

    async def _load_embeddings_matrix(self) -> Tuple[np.ndarray, List[Tuple[Any, int]]]:
        docs = await self._find_all_with_embeddings()
        all_chunks, chunk_meta = [], []
        for doc in docs:
            doc_id = doc.get(self.id_field)
            if doc_id is None: continue
            chunk_matrix = self.coerce_to_chunk_matrix(doc.get(self.embedding_field))
            if chunk_matrix is None: continue
            for i, chunk_vector in enumerate(chunk_matrix):
                all_chunks.append(chunk_vector)
                chunk_meta.append((doc_id, i))
        if not all_chunks: return np.zeros((0, 0), dtype=np.float32), []
        matrix = np.vstack(all_chunks).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12), chunk_meta

    def _build_hnsw(self, M: np.ndarray):
        if not self.use_hnsw or M.size == 0:
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
            M, meta = await self._load_embeddings_matrix()
            self.emb_matrix = M
            self.chunk_metadata = meta
            self._dim = M.shape[1] if M.size > 0 else 0
            self._built_at = time.time()
            self._build_hnsw(M)

    async def _fetch_doc_by_id(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        res = await self.repo.find_document(self.collection, {self.id_field: doc_id})
        return res.data if res and res.success else None

    def _search_index(self, qn: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        M = self.emb_matrix
        if M is None or M.size == 0 or k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # FIX: Consolidated and corrected scoring logic.
        # This is the dot product for cosine similarity, as both M and qn are normalized.
        scores = (M @ qn).astype(np.float32)

        # Efficiently find the top-k scores and their original indices
        k = min(k, scores.shape[0])
        topk_unsorted_indices = np.argpartition(scores, -k)[-k:]

        # Sort only the top-k candidates to get the final order
        order_of_topk = np.argsort(-scores[topk_unsorted_indices])

        sorted_indices = topk_unsorted_indices[order_of_topk]

        return sorted_indices, scores[sorted_indices]

    async def search(self, query_embedding: List[float], top_k: int) -> SafeResult:
        await self._ensure_index()
        if self.emb_matrix is None or self.emb_matrix.size == 0:
            return SafeResult.ok([])

        try:
            q = np.asarray(query_embedding, dtype=np.float32)
            qn = self._normalize(q)
            if qn.size != self._dim:
                return SafeResult.fail(f"Query vector dimension mismatch")
        except Exception as e:
            return SafeResult.fail(f"Invalid query embedding: {e}")

        k = max(0, min(int(top_k), self.emb_matrix.shape[0]))
        if k == 0: return SafeResult.ok([])

        candidate_indices, candidate_scores = self._search_index(qn, k)

        best_for_doc: Dict[Any, Tuple[float, int]] = {}
        for idx, score in zip(candidate_indices, candidate_scores):
            doc_id, chunk_idx = self.chunk_metadata[int(idx)]
            key = doc_id.binary if isinstance(doc_id, ObjectId) else doc_id
            if key not in best_for_doc or score > best_for_doc[key][0]:
                best_for_doc[key] = (float(score), int(chunk_idx))

        ordered = sorted(best_for_doc.items(), key=lambda kv: kv[1][0], reverse=True)

        results = []
        for key, (score, _) in ordered[:k]:
            doc_id = ObjectId(key) if isinstance(key, bytes) else key
            doc = await self._fetch_doc_by_id(doc_id)
            if doc:
                results.append({"retrieval_score": score, "document": doc})

        return SafeResult.ok(results)
