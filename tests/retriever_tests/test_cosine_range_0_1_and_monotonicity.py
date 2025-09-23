import numpy as np
import pytest

from zmongo_toolbag.unified_vector_search import LocalVectorSearch


class _DummyRepo:
    """Minimal stub so we can new-up LocalVectorSearch without touching Mongo."""
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("score_mode", ["cosine_0_1"])
async def test_cosine_0_1_range_and_monotonicity(score_mode, monkeypatch):
    """
    Contract test: with score_mode=cosine_0_1, retrieval_score must be in [0,1]
    and must be monotonic with the true cosine.

    We create two unit vectors:
      - v_hi is perfectly aligned with q  -> cosine = 1.0 -> score = 1.0
      - v_lo has cosine = 0.6            -> score = 0.8
    The higher-cosine doc must appear first, and scores must lie within [0,1].
    """
    lvs = LocalVectorSearch(
        repository=_DummyRepo(),
        collection="unused",
        embedding_field="embeddings",
        use_hnsw=False,
        score_mode=score_mode,
    )

    # Query and two candidate chunk vectors
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_hi = np.array([1.0, 0.0, 0.0], dtype=np.float32)           # cos(q, v_hi) = 1.0
    v_lo = np.array([0.6, 0.8, 0.0], dtype=np.float32)           # cos(q, v_lo) = 0.6

    # Row-normalized emb matrix (order: lo, hi), with matching chunk metadata
    def _unit(x):
        n = np.linalg.norm(x)
        return x if n == 0 else (x / n)

    lvs.emb_matrix = np.vstack([_unit(v_lo), _unit(v_hi)]).astype(np.float32)
    lvs.chunk_metadata = [("doc_lo", 0), ("doc_hi", 0)]

    # Avoid touching Mongo inside search()
    async def _noop():
        return None
    monkeypatch.setattr(lvs, "_ensure_index", _noop)

    async def _fake_fetch(doc_id):
        return {"_id": doc_id, "text": f"text for {doc_id}"}
    monkeypatch.setattr(lvs, "_fetch_doc_by_id", _fake_fetch)

    # Act
    sr = await lvs.search(query_embedding=q.tolist(), top_k=2)
    assert sr.success, sr.error
    hits = sr.data
    assert len(hits) == 2

    # Assert score range and ordering
    for h in hits:
        assert 0.0 <= h["retrieval_score"] <= 1.0

    # Highest cosine first (doc_hi), and monotonicity
    assert hits[0]["doc_id"] == "doc_hi"
    assert hits[1]["doc_id"] == "doc_lo"
    assert hits[0]["retrieval_score"] >= hits[1]["retrieval_score"]


@pytest.mark.parametrize(
    "cos, expected",
    [
        (-1.0, 0.0),  # farthest possible
        (0.0, 0.5),   # orthogonal
        (1.0, 1.0),   # identical
        (0.6, 0.8),   # typical mid value
    ],
)
def test_to_output_score_cosine_0_1_exact_mapping(cos, expected):
    """
    _to_output_score must map cosine to [0,1] via (cos+1)/2 exactly.
    """
    lvs = LocalVectorSearch(
        repository=_DummyRepo(),
        collection="unused",
        embedding_field="embeddings",
        score_mode="cosine_0_1",
    )
    out = lvs._to_output_score(cos=cos, dist=(1.0 - cos))
    assert out == pytest.approx(expected, rel=1e-9, abs=1e-9)
