# zgemini_tests/test_local_vector_search_integration.py
import asyncio
from typing import List

import pytest
import numpy as np

from zmongo_retriever.zmongo_toolbag import ZMongo, LocalVectorSearch

EMBED_FIELD = "content_embedding"
DIM = 4  # simple/clear demo dimension


def _vec(values: List[float]) -> List[float]:
    """Helper to normalize vectors for consistent test data."""
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()


def _hit_doc_id(hit: dict) -> str:
    """
    Support both result formats:
      - NEW: {"doc_id": "...", "text": "...", "metadata": {...}}
      - OLD: {"document": {"_id": "...", ...}, "retrieval_score": ...}
    """
    # New schema first
    if "doc_id" in hit and hit["doc_id"]:
        return hit["doc_id"]
    if "metadata" in hit and isinstance(hit["metadata"], dict) and "_id" in hit["metadata"]:
        return str(hit["metadata"]["_id"])
    # Fallback to old schema
    if "document" in hit and isinstance(hit["document"], dict) and "_id" in hit["document"]:
        return str(hit["document"]["_id"])
    raise KeyError("Could not determine document id from search hit: keys present = %r" % list(hit.keys()))


@pytest.mark.asyncio
async def test_local_vector_search_end_to_end():
    # 1) Setup a real ZMongo instance
    repo = ZMongo()
    collection_name = "vs_test_docs"

    # Clean slate for the test run
    await repo.delete_documents(collection_name, {})

    # 2) Insert documents with a mix of single and chunked embeddings
    docs = [
        {"_id": "d1", "content": "strong-x", EMBED_FIELD: _vec([0.98, 0.02, 0.0, 0.0])},
        {"_id": "d2", "content": "strong-y", EMBED_FIELD: _vec([0.01, 0.99, 0.0, 0.0])},
        {"_id": "d3", "content": "strong-z", EMBED_FIELD: _vec([0.01, 0.0, 0.99, 0.0])},
        {
            "_id": "d4",
            "content": "chunked-x-and-w",
            EMBED_FIELD: [_vec([0.97, 0.03, 0.0, 0.0]), _vec([0.0, 0.0, 0.0, 1.0])],
        },
        {
            "_id": "d5",
            "content": "chunked-y-and-z",
            EMBED_FIELD: [_vec([0.0, 1.0, 0.0, 0.0]), _vec([0.0, 0.0, 1.0, 0.0])],
        },
    ]
    ins = await repo.insert_documents(collection_name, docs)
    assert ins.success, f"Insert failed: {ins.error}"

    # 3) Initialize LocalVectorSearch
    lvs = LocalVectorSearch(
        repository=repo,
        collection=collection_name,
        embedding_field=EMBED_FIELD,
        ttl_seconds=2,  # Short TTL to exercise the refresh path
    )

    # 4) Query for a vector near [1,0,0,0]
    query = _vec([1.0, 0.0, 0.0, 0.0])
    res = await lvs.search(query, top_k=3)
    assert res.success, f"Search failed: {res.error}"

    hits = res.data
    assert isinstance(hits, list) and len(hits) == 3

    # Because the new searcher correctly finds the best *chunk*, d1 and d4 should be top results.
    ids_order = [_hit_doc_id(h) for h in hits]
    assert ids_order[0] == "d1", f"Expected d1 to be the top hit, got {ids_order}"
    assert ids_order[1] == "d4", f"Expected d4 to be the second hit, got {ids_order}"

    # 5) Test HNSW path (if available)
    if lvs.use_hnsw:
        lvs_hnsw = LocalVectorSearch(
            repository=repo,
            collection=collection_name,
            embedding_field=EMBED_FIELD,
            use_hnsw=True,
        )
        res_hnsw = await lvs_hnsw.search(query, top_k=3)
        assert res_hnsw.success, f"HNSW search failed: {res_hnsw.error}"
        ids_order_hnsw = [_hit_doc_id(h) for h in res_hnsw.data]
        assert ids_order_hnsw[0] == "d1", f"HNSW expected d1 at top, got {ids_order_hnsw}"
        assert ids_order_hnsw[1] == "d4", f"HNSW expected d4 second, got {ids_order_hnsw}"

    # 6) Test TTL refresh by updating a document
    await asyncio.sleep(2.2)  # Exceed ttl_seconds

    # Update d2 to be the best match for the query
    new_d2_vec = _vec([1.0, 0.0, 0.0, 0.0])
    upd = await repo.update_document(collection_name, {"_id": "d2"}, {"$set": {EMBED_FIELD: new_d2_vec}})
    assert upd.success

    # Search again; the index should refresh and d2 should now be the top hit
    res_refresh = await lvs.search(query, top_k=3)
    assert res_refresh.success, f"Search after refresh failed: {res_refresh.error}"
    ids_order_refresh = [_hit_doc_id(h) for h in res_refresh.data]
    assert ids_order_refresh[0] == "d2", f"d2 should be the top hit after update and refresh, got {ids_order_refresh}"

    # 7) Cleanup
    await repo.delete_documents(collection_name, {})
    repo.close()
