# zgemini_tests/test_local_vector_search_integration.py
# Python 3.10 compatible
#
# REAL integration test (no mocks):
# - Requires a running MongoDB (defaults picked up by ZMongo:
#   MONGO_URI, MONGO_DATABASE_NAME) or falls back to 127.0.0.1:27017 / "test".
# - Inserts single-vector and chunked-vector docs.
# - Exercises LocalVectorSearch with/without exact rescoring and optional HNSW.
#
# Run:
#   pip install pytest pytest-asyncio
#   pytest -q zgemini_tests/test_local_vector_search_integration.py

import asyncio
from typing import List

import pytest

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.unified_vector_search import LocalVectorSearch

EMBED_FIELD = "content_embedding"
DIM = 4  # simple/clear demo dimension


def _vec(values: List[float]) -> List[float]:
    return [float(x) for x in values]


@pytest.mark.asyncio
async def test_local_vector_search_end_to_end():
    # 1) Real Mongo: ZMongo opens the client in __init__, no connect() needed.
    repo = ZMongo()

    # Clean slate (ZMongo uses delete_documents)
    await repo.delete_documents("vs_test_docs", {})

    # 2) Insert documents:
    # d1..d3: single vectors; d4..d5: chunked vectors.
    d1 = {"_id": "d1", "content": "one-hot-x", EMBED_FIELD: _vec([0.98, 0.02, 0.0, 0.0])}
    d2 = {"_id": "d2", "content": "one-hot-y", EMBED_FIELD: _vec([0.01, 0.99, 0.0, 0.0])}
    d3 = {"_id": "d3", "content": "one-hot-z", EMBED_FIELD: _vec([0.01, 0.0, 0.99, 0.0])}
    d4 = {
        "_id": "d4",
        "content": "chunked-x-and-w",
        EMBED_FIELD: [
            _vec([0.97, 0.03, 0.0, 0.0]),
            _vec([0.0, 0.0, 0.0, 1.0]),
        ],
    }
    d5 = {
        "_id": "d5",
        "content": "chunked-y-and-z",
        EMBED_FIELD: [
            _vec([0.0, 1.0, 0.0, 0.0]),
            _vec([0.0, 0.0, 1.0, 0.0]),
        ],
    }

    ins = await repo.insert_documents("vs_test_docs", [d1, d2, d3, d4, d5])
    assert ins.success, f"Insert failed: {ins.error}"

    # 3) LocalVectorSearch: NumPy backend first (no rescoring)
    lvs = LocalVectorSearch(
        repository=repo,
        collection="vs_test_docs",
        embedding_field=EMBED_FIELD,
        ttl_seconds=1,            # short TTL to exercise refresh path
        id_field="_id",
        chunked_embeddings=True,  # collection mixes single & chunked
        use_hnsw=False,
        exact_rescore=False,
    )

    # 4) Query near [1,0,0,0]
    query = _vec([1.0, 0.0, 0.0, 0.0])

    res = await lvs.search(query, top_k=3)
    assert res.success, f"Search failed: {res.error}"
    hits = res.data
    assert isinstance(hits, list) and len(hits) > 0

    ids_order = [h["document"]["_id"] for h in hits]
    # With index-time mean-pooling, d1 is very likely #1; d4 should be in top 3.
    assert "d1" in ids_order[:2], f"Expected d1 near top, got {ids_order}"
    assert "d4" in ids_order[:3], f"Expected d4 in top-3, got {ids_order}"

    # 5) Enable exact rescoring (max over chunks) — d4 should rise
    lvs.exact_rescore = True
    lvs.re_rank_candidates = None
    lvs.re_rank_multiplier = 3.0

    res2 = await lvs.search(query, top_k=3)
    assert res2.success
    ids_order2 = [h["document"]["_id"] for h in res2.data]
    assert "d4" in ids_order2[:2], f"With rescoring, d4 should rise: {ids_order2}"

    # 6) Optional HNSW path (skip if not installed)
    try:
        import hnswlib  # noqa: F401
        have_hnsw = True
    except Exception:
        have_hnsw = False

    if have_hnsw:
        lvs_hnsw = LocalVectorSearch(
            repository=repo,
            collection="vs_test_docs",
            embedding_field=EMBED_FIELD,
            ttl_seconds=1,
            id_field="_id",
            chunked_embeddings=True,
            use_hnsw=True,
            hnsw_m=8,
            hnsw_ef_construction=100,
            hnsw_ef_search=100,
            exact_rescore=True,
            re_rank_candidates=10,
        )
        res3 = await lvs_hnsw.search(query, top_k=3)
        assert res3.success
        ids_order3 = [h["document"]["_id"] for h in res3.data]
        assert "d4" in ids_order3[:2], f"HNSW+rescore should keep d4 near top: {ids_order3}"

    # 7) TTL refresh check — update d2 to be more x-like and confirm climb
    await asyncio.sleep(1.2)  # exceed ttl_seconds
    new_d2_vec = _vec([0.9, 0.1, 0.0, 0.0])
    upd = await repo.update_document("vs_test_docs", {"_id": "d2"}, {"$set": {EMBED_FIELD: new_d2_vec}})
    assert upd.success

    res4 = await lvs.search(query, top_k=3)
    assert res4.success
    ids_order4 = [h["document"]["_id"] for h in res4.data]
    assert "d2" in ids_order4[:3], f"d2 should climb after update: {ids_order4}"

    # 8) Cleanup
    await repo.delete_documents("vs_test_docs", {})
    repo.close()  # ZMongo defines async close()
