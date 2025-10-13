import logging
import numpy as np
import pytest
from bson import ObjectId

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import ZEmbedder, EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.safe_result import SafeResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEST_COLLECTION = "unified_vs_integration"
EMBED_FIELD = "embeddings"
TEXT_FIELD = "text"


@pytest.fixture(scope="module")
def repo():
    """Provides a real ZMongo connection for integration tests."""
    zm = ZMongo()
    zm.delete_all_documents(TEST_COLLECTION)
    yield zm
    zm.delete_all_documents(TEST_COLLECTION)
    zm.close()


@pytest.fixture(scope="module")
def embedder(repo):
    """Provides an initialized ZEmbedder using the same repository."""
    return ZEmbedder(repository=repo)


@pytest.fixture(scope="module")
def vector_search(repo):
    """Provides a vector searcher tied to the same repo and collection."""
    return LocalVectorSearch(repository=repo, collection=TEST_COLLECTION, embedding_field=EMBED_FIELD)


def insert_docs_and_embed(repo: ZMongo, embedder: ZEmbedder):
    """Helper: insert real docs and embed them."""
    docs = [
        {"_id": ObjectId(), TEXT_FIELD: "The quick brown fox jumps over the lazy dog."},
        {"_id": ObjectId(), TEXT_FIELD: "Mitochondria are the powerhouse of the cell."},
        {"_id": ObjectId(), TEXT_FIELD: "The Sun is the center of the solar system."},
    ]
    for doc in docs:
        ins = repo.insert_one(TEST_COLLECTION, doc)
        assert ins.success, f"Insert failed: {ins.error}"

        emb_res = embedder.get_embedding(
            text=doc[TEXT_FIELD],
            collection=TEST_COLLECTION,
            document_id=doc["_id"],
            embedding_field=EMBED_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        )
        if isinstance(emb_res, SafeResult):
            assert emb_res.success, f"Embedding failed: {emb_res.error}"
        else:
            logger.warning("Non-SafeResult returned from embedder: %r", emb_res)

    count_res = repo.count_documents(TEST_COLLECTION)
    assert count_res.success and count_res.data >= 3


def test_end_to_end_real_vector_search(repo: ZMongo, embedder: ZEmbedder, vector_search: LocalVectorSearch):
    """
    Full real-data test:
    1. Insert and embed documents.
    2. Run vector search.
    3. Verify ranked results and SafeResult integrity.
    """
    repo.delete_all_documents(TEST_COLLECTION)
    insert_docs_and_embed(repo, embedder)

    rebuild_res = repo.run_sync(vector_search.rebuild_index())
    assert rebuild_res.success, f"Index rebuild failed: {rebuild_res.error}"

    query_text = "What provides energy in a biological cell?"
    emb_res = embedder.get_embedding(query_text)
    assert emb_res.success
    qvec = emb_res.data["vectors"][0]

    search_res = repo.run_sync(vector_search.search(qvec, top_k=3))
    assert search_res.success, f"Search failed: {search_res.error}"
    results = search_res.data
    assert isinstance(results, list)
    assert len(results) > 0

    top_doc = results[0]["document"]
    score = results[0]["retrieval_score"]
    assert isinstance(score, float)
    assert EMBED_FIELD in top_doc
    logger.info("Top document: %s (score=%.3f)", top_doc.get(TEXT_FIELD), score)


def test_zero_vector_query_handling(repo: ZMongo, vector_search: LocalVectorSearch):
    """Zero-vector queries should produce SafeResult.ok([]) gracefully."""
    zero_vec = [0.0, 0.0, 0.0]
    res = repo.run_sync(vector_search.search(zero_vec, top_k=3))
    assert res.success
    assert isinstance(res.data, list)


def test_score_mapping_modes(repo: ZMongo, vector_search: LocalVectorSearch):
    """Verify consistent _to_output_score mapping."""
    # cosine_0_1 mode
    vector_search.score_mode = "cosine_0_1"
    assert np.isclose(vector_search._to_output_score(cos=1.0, dist=0.0), 1.0)
    assert np.isclose(vector_search._to_output_score(cos=-1.0, dist=0.0), 0.0)

    # cosine
    vector_search.score_mode = "cosine"
    assert np.isclose(vector_search._to_output_score(cos=0.5, dist=0.0), 0.5)

    # distance
    vector_search.score_mode = "distance"
    assert np.isclose(vector_search._to_output_score(cos=0.0, dist=0.2), 0.8)


def test_chunked_embeddings_flattening(repo: ZMongo, vector_search: LocalVectorSearch):
    """Ensure multi-vector (chunked) docs flatten properly."""
    repo.delete_all_documents(TEST_COLLECTION)
    doc = {
        "_id": ObjectId(),
        TEXT_FIELD: "multi-vector doc",
        EMBED_FIELD: [[1.0, 0.0], [0.0, 1.0]],
    }
    ins = repo.insert_one(TEST_COLLECTION, doc)
    assert ins.success

    vector_search.chunked_embeddings = True
    res = repo.run_sync(vector_search.rebuild_index())
    assert res.success
    M = res.data["matrix"]
    assert M.shape[0] == 2
