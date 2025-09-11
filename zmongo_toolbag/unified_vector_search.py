# =============================================
# unified_vector_search.py  (drop-in replacement)
# =============================================
"""
LocalVectorSearch — in-memory cosine search over chunked MongoDB embeddings
===========================================================================

This module provides :class:`LocalVectorSearch`, a lightweight vector search
component that:

- Pulls **chunked embeddings** from a MongoDB collection via a `ZMongo` repository.
- Builds an **in-memory row-normalized matrix** of all chunks (each row = one chunk).
- Supports **brute-force cosine** or optional **HNSW** (if hnswlib is available).
- Returns **per-document** results (deduped across chunks), keeping the **best chunk**
  per document and sorting by cosine **descending**.
- Emits flexible `retrieval_score` semantics using `score_mode`:
  - `"cosine"` (default): raw cosine in **[-1, 1]**, higher is better.
  - `"cosine_0_1"`: cosine mapped to **[0, 1]** via `(cos + 1) / 2`.
  - `"distance"`: **1 − cosine** in **[0, 2]**, lower is better.
  - `"distance_0_1"`: `(1 − cosine) / 2` in **[0, 1]`, lower is better.

Typical wiring pairs this searcher with an embedder (e.g., `ZEmbedder`) and a
LangChain-compatible retriever (e.g., `ZRetriever`). In that setup, **documents**
are usually embedded using a *document* style (e.g., `RETRIEVAL_DOCUMENT`), while
**queries** use a *query* style (e.g., `RETRIEVAL_QUERY`), producing consistent,
comparable vectors.

Key Design Points
-----------------
- **Normalization first**: all chunk vectors are L2-normalized row-wise so that
  cosine similarity reduces to a simple dot product with a normalized query.
- **Chunk-level search, doc-level output**: we search individual chunks to
  maximize recall/precision, then deduplicate by document and keep the best chunk.
- **Exact re-score (HNSW)**: even when using HNSW, you can request an **exact
  cosine** recomputation on the top candidates (`exact_rescore=True`) for accuracy.
- **Stateless construction with cached matrix**: the matrix is rebuilt on each
  `_ensure_index()` call in this implementation (TTL check removed to avoid state
  leakage across tests). You can add TTL caching back if desired.

Quick Example
-------------
.. code-block:: python

    repo = ZMongo()
    lvs = LocalVectorSearch(
        repository=repo,
        collection="kb",
        embedding_field="embeddings",
        chunked_embeddings=True,
        use_hnsw=False,
        exact_rescore=True,
        score_mode="cosine",  # raw cosine in [-1, 1]
    )

    q_emb = await embedder.get_embedding("What is the powerhouse of the cell?")[0]
    hits_sr = await lvs.search(q_emb, top_k=5)
    assert hits_sr.success
    hits = hits_sr.data
    for h in hits:
        print(h["retrieval_score"], h["text"], h["metadata"].get("topic"))

Notes
-----
This module returns results as a **list of dicts** (wrapped in `SafeResult`) using
a *new* shape:

    {
      "doc_id": "<string>",
      "text": "<best chunk text or doc text>",
      "metadata": { ...original doc fields except the embedding field... },
      "retrieval_score": <float according to score_mode>,
      # debugging/back-compat:
      "raw_cosine": <float>, "chunk_index": <int>, "document": {<original doc>}
    }

The legacy shape (with `"document"` only) is kept for back-compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

    This implementation:
      1) Loads documents with embeddings from a MongoDB collection.
      2) Converts each document's embedding field into a **chunk matrix**
         (N_chunks x dim), supporting both 1D and 2D representations.
      3) **Row-normalizes** the concatenated matrix so cosine = dot product.
      4) Answers search queries via brute-force or optional HNSW.
      5) Deduplicates hits by document, keeping the **highest-cosine** chunk.

    Parameters
    ----------
    repository : ZMongo
        Repository that exposes async CRUD methods for MongoDB.
    collection : str
        The MongoDB collection name containing documents with embeddings.
    embedding_field : str
        The field on each document that stores its (chunked) embeddings.
    ttl_seconds : int, default 300
        Nominal TTL for an index build. (This implementation always rebuilds
        to avoid test interference; reintroduce TTL logic if you need caching.)
    id_field : str, default "_id"
        The document's primary key field (commonly `"_id"`).
    chunked_embeddings : bool, default True
        If True, embeddings are expected as chunks (2D). 1D is coerced to (1, D).
    use_hnsw : bool, default False
        Whether to use HNSW for candidate retrieval (requires `hnswlib`).
    hnsw_m : int, default 16
        HNSW `M` parameter (graph connectivity).
    hnsw_ef_construction : int, default 200
        HNSW `ef_construction` parameter (build-time accuracy/speed trade-off).
    hnsw_ef_search : int, default 200
        HNSW `ef` parameter used during search (higher -> better recall).
    exact_rescore : bool, default True
        If True and HNSW is used, recompute **exact cosine** for candidates.
    score_mode : str, default "cosine"
        Output score mode: "cosine", "cosine_0_1", "distance", "distance_0_1".

    Attributes
    ----------
    emb_matrix : Optional[np.ndarray]
        The row-normalized chunk matrix of shape (N_chunks, dim). `None` if empty.
    chunk_metadata : List[Tuple[Any, int]]
        Per-row metadata `(doc_id, chunk_index)`, parallel to `emb_matrix`.
    _hnsw_index : Optional[hnswlib.Index]
        The in-memory HNSW index if enabled and built.
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
        score_mode: str = "cosine_0_1",
    ):
        # Public config
        self.score_mode = score_mode  # "cosine", "cosine_0_1", "distance", "distance_0_1"
        self.repo = repository
        self.collection = collection
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.ttl = ttl_seconds
        self.chunked = chunked_embeddings

        # HNSW knobs
        self.use_hnsw = bool(use_hnsw and _HNSW_AVAILABLE)
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.exact_rescore = exact_rescore

        # Internal state
        self._lock = asyncio.Lock()
        self._built_at: float = 0.0
        self.emb_matrix: Optional[np.ndarray] = None
        self.chunk_metadata: List[Tuple[Any, int]] = []
        self._dim: int = 0
        self._hnsw_index = None

    # ----------------------------
    # Coercion & normalization
    # ----------------------------
    @staticmethod
    def coerce_to_chunk_matrix(emb_val: Any) -> Optional[np.ndarray]:
        """
        Convert an arbitrary embedding value into a 2D chunk matrix.

        Accepts:
          - 1D list/array -> coerced to shape (1, D)
          - 2D list/array -> validated and returned

        Returns
        -------
        numpy.ndarray or None
            A `float32` array of shape (N, D), or `None` if coercion fails.
        """
        if emb_val is None:
            return None
        try:
            arr = np.array(emb_val, dtype=np.float32)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                return arr
            return None
        except (ValueError, TypeError):
            logger.debug("Failed to coerce embedding value to a matrix.")
            return None

    @staticmethod
    def _normalize_vec(v: np.ndarray) -> np.ndarray:
        """
        Return a normalized copy of a vector (or the original if near-zero norm)."""
        n = np.linalg.norm(v)
        return v if n < 1e-12 else (v / n)

    def _normalize_matrix_rows(self, M: np.ndarray) -> np.ndarray:
        """
        Normalize each row of a matrix (filtering out near-zero rows).

        Notes
        -----
        Not used in the main pipeline (left as a utility). The main normalization
        occurs in `_load_embeddings_matrix()`.
        """
        norms = np.linalg.norm(M, axis=1)
        keep = norms > 1e-8
        M = M[keep]
        M = M / norms[keep, None]
        # If you filter here, remember to filter `chunk_metadata` in lockstep.
        return M

    # ----------------------------
    # Index building & search core
    # ----------------------------
    def _search_index(self, qn: np.ndarray, k: int):
        """
        Search either the brute-force matrix or the HNSW index.

        Parameters
        ----------
        qn : np.ndarray
            **Normalized** query vector of shape (D,).
        k : int
            Desired number of chunk candidates.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (chunk_row_indices, cosines, dists) where:
              - cosines = dot(emb_matrix[row], qn), i.e., cosine similarity
              - dists   = 1 − cosines  (to align with HNSW's 'cosine' distance)
        """
        M = self.emb_matrix
        if self.use_hnsw and self._hnsw_index is not None:
            # Over-fetch candidates to preserve quality, then exact re-score if requested.
            ef_k = min(max(k * (3 if self.exact_rescore else 1), k), M.shape[0])
            labels, dists = self._hnsw_index.knn_query(qn.reshape(1, -1), k=ef_k)
            idxs = labels[0]
            dists = dists[0].astype(np.float32)  # hnswlib 'cosine' distance = 1 − cos

            if self.exact_rescore:
                # Matrix is row-normalized -> dot == cosine
                cos = (M[idxs] @ qn).astype(np.float32)
            else:
                cos = (1.0 - dists).astype(np.float32)

            # Rank by cosine desc and take top-k
            k_eff = min(k, cos.shape[0])
            sorted_candidate_indices = np.argsort(-cos)[:k_eff]
            final_idxs = idxs[sorted_candidate_indices]
            final_cos = cos[sorted_candidate_indices]
            final_dists = dists[sorted_candidate_indices]
            return final_idxs, final_cos, final_dists

        # Brute-force: cosine = M @ qn (rows already normalized)
        scores = (M @ qn).astype(np.float32)
        k_eff = min(k, scores.shape[0])
        sorted_indices = np.argsort(-scores)[:k_eff]
        top_scores = scores[sorted_indices]
        top_dists = (1.0 - top_scores).astype(np.float32)  # for symmetry with HNSW
        return sorted_indices, top_scores, top_dists

    async def search(self, query_embedding: List[float], top_k: int):
        """
        Execute a vector search for the given query embedding.

        Steps
        -----
        1) Ensure the in-memory index is built (loads/normalizes embeddings).
        2) Normalize the **query** embedding.
        3) Retrieve top-k chunk candidates (HNSW or brute-force).
        4) Deduplicate by document, keeping the **best chunk** per doc.
        5) Return at most `top_k` **documents** (sorted by cosine desc).

        Parameters
        ----------
        query_embedding : List[float]
            The embedding vector for the query (any float-like sequence).
        top_k : int
            Maximum number of documents to return.

        Returns
        -------
        SafeResult
            `SafeResult.ok(list_of_hits)` where each hit is a dict using the
            *new* shape documented at the top of the module, or `SafeResult.fail(...)`
            if input is degenerate (e.g., near-zero query norm).

        Notes
        -----
        - If no embeddings are present, returns an empty OK result.
        - If `top_k <= 0`, returns an empty OK result.
        """
        await self._ensure_index()
        if self.emb_matrix is None or self.emb_matrix.size == 0:
            return SafeResult.ok([])

        # Normalize the query embedding
        q = np.asarray(query_embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return SafeResult.fail("Query embedding has (near) zero norm; cannot compute cosine similarity.")
        qn = q / n

        # Guard top_k
        k = max(0, min(int(top_k), self.emb_matrix.shape[0]))
        if k == 0:
            return SafeResult.ok([])

        idxs, cosines, dists = self._search_index(qn, k)

        # Deduplicate per document; keep best cosine per doc
        best_for_doc: Dict[Any, Tuple[float, int, float]] = {}
        for idx, cos in zip(idxs, cosines):
            doc_id, chunk_idx = self.chunk_metadata[int(idx)]
            if (doc_id not in best_for_doc) or (cos > best_for_doc[doc_id][0]):
                best_for_doc[doc_id] = (float(cos), int(chunk_idx), float(1.0 - cos))

        # Sort by cosine desc (document-level)
        ordered = sorted(best_for_doc.items(), key=lambda kv: kv[1][0], reverse=True)

        out: List[Dict[str, Any]] = []
        for doc_id, (cos, chunk_idx, dist) in ordered[:k]:
            doc = await self._fetch_doc_by_id(doc_id)
            if not doc:
                continue
            out.append({
                # NEW shape (preferred by downstream retrievers)
                "doc_id": str(doc.get(self.id_field, doc_id)),
                "text": doc.get("text") or doc.get("content") or "",
                "metadata": {k: v for k, v in doc.items() if k != self.embedding_field},
                "retrieval_score": self._to_output_score(cos=cos, dist=dist),
                # Debug/back-compat
                "raw_cosine": float(cos),
                "chunk_index": int(chunk_idx),
                "document": doc,  # OLD shape kept for compatibility
            })

        return SafeResult.ok(out)

    # ----------------------------
    # Index building helpers
    # ----------------------------
    async def _find_all_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Fetch all documents that have a non-empty embedding field.

        Returns
        -------
        List[dict]
            List of documents (raw Mongo dicts). Raises `RuntimeError` if the
            repository call fails.
        """
        res = await self.repo.find_documents(
            self.collection,
            {self.embedding_field: {"$exists": True, "$ne": []}},
            limit=1_000_000,
        )
        if not res.success:
            raise RuntimeError(res.error)
        return res.data or []

    def _build_hnsw(self, M: np.ndarray) -> None:
        """
        Build (or clear) the HNSW index based on the current matrix.

        Parameters
        ----------
        M : np.ndarray
            Row-normalized chunk matrix. If empty, clears the HNSW index.
        """
        if not self.use_hnsw or M.size == 0:
            self._hnsw_index = None
            return
        num_elements, dim = M.shape
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=self.hnsw_ef_construction, M=self.hnsw_m)
        index.add_items(M, np.arange(num_elements))
        index.set_ef(self.hnsw_ef_search)
        self._hnsw_index = index

    async def _ensure_index(self) -> None:
        """
        (Re)build the in-memory matrix (and HNSW index if enabled).

        Notes
        -----
        - TTL-based short-circuiting is intentionally **omitted** here to avoid
          state leakage in tests. To reintroduce caching, check `self._built_at`
          against `self.ttl` and skip rebuilds until expired.
        """
        async with self._lock:
            # Always rebuild to guarantee test isolation.
            M, meta = await self._load_embeddings_matrix()
            self.emb_matrix = M
            self.chunk_metadata = meta
            self._dim = M.shape[1] if M.size > 0 else 0
            self._built_at = time.time()
            self._build_hnsw(M)

    async def _fetch_doc_by_id(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        """
        Fetch a single document by ID, coercing string ObjectIds when appropriate.

        Parameters
        ----------
        doc_id : Any
            The raw ID from `chunk_metadata` (could be `ObjectId` or `str`).

        Returns
        -------
        dict or None
            The document dict if found; otherwise `None`.
        """
        key = doc_id
        # Coerce if it *looks* like a valid ObjectId hex string.
        if isinstance(doc_id, str):
            try:
                from bson import ObjectId as _OID
                if _OID.is_valid(doc_id):
                    key = _OID(doc_id)
            except Exception:
                key = doc_id
        res = await self.repo.find_document(self.collection, {self.id_field: key})
        return res.data if res and res.success else None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """
        Normalize a vector; returns the input if near-zero norm (caller handles it)."""
        n = np.linalg.norm(vec)
        if n < 1e-12:
            return vec  # caller will handle zero-vector
        return vec / n

    async def _load_embeddings_matrix(self) -> Tuple[np.ndarray, List[Tuple[Any, int]]]:
        """
        Load all embedding chunks from MongoDB and row-normalize them.

        Returns
        -------
        Tuple[np.ndarray, List[Tuple[Any, int]]]
            (M, meta) where:
              - `M` is a `float32` array of shape (N_chunks, dim), with each
                row L2-normalized.
              - `meta` is a list of `(doc_id, chunk_index)` aligned to `M` rows.

        Notes
        -----
        - Any documents with missing/invalid embeddings are skipped.
        - Any chunk rows with near-zero norm are filtered out to avoid
          degenerate cosine behavior.
        """
        docs = await self._find_all_with_embeddings()
        all_chunks: List[np.ndarray] = []
        chunk_meta: List[Tuple[Any, int]] = []

        # Collect all chunks across docs
        for doc in docs:
            doc_id = doc.get(self.id_field)
            if doc_id is None:
                continue
            chunk_matrix = self.coerce_to_chunk_matrix(doc.get(self.embedding_field))
            if chunk_matrix is None:
                continue
            for i, chunk_vector in enumerate(chunk_matrix):
                all_chunks.append(chunk_vector)
                chunk_meta.append((doc_id, i))

        if not all_chunks:
            return np.zeros((0, 0), dtype=np.float32), []

        M = np.vstack(all_chunks).astype(np.float32)

        # Filter zero/near-zero norm rows to avoid degenerate normalization
        norms = np.linalg.norm(M, axis=1)
        keep = norms > 1e-8
        if not np.any(keep):
            return np.zeros((0, 0), dtype=np.float32), []
        M = M[keep]
        chunk_meta = [cm for cm, k in zip(chunk_meta, keep) if k]

        # Row-normalize for cosine
        M = M / norms[keep, None]
        return M, chunk_meta

    # ----------------------------
    # Output score shaping
    # ----------------------------
    def _to_output_score(self, *, cos: float, dist: float) -> float:
        """
        Map internal cosine/distance to the configured output scale.

        Parameters
        ----------
        cos : float
            Cosine similarity in [-1, 1].
        dist : float
            1 − cosine (i.e., HNSW's cosine distance), in [0, 2].

        Returns
        -------
        float
            The output `retrieval_score` according to `self.score_mode`:
              - "cosine"       -> `cos`               ([-1, 1], higher is better)
              - "cosine_0_1"   -> `(cos + 1) / 2`     ([0, 1], higher is better)
              - "distance"     -> `dist`              ([0, 2], lower is better)
              - "distance_0_1" -> `dist / 2`          ([0, 1], lower is better)
            Falls back to `"cosine"` if mode is unknown.
        """
        if self.score_mode == "cosine":
            return float(cos)  # [-1, 1]
        if self.score_mode == "cosine_0_1":
            return float(0.5 * (cos + 1.0))  # [0, 1]
        if self.score_mode == "distance":
            return float(dist)  # (1 - cos) in [0, 2]
        if self.score_mode == "distance_0_1":
            return float(0.5 * dist)  # [0, 1]
        return float(cos)
