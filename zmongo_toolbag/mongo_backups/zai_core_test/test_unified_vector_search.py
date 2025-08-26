# zmongo_retriever\zmongo_toolbag\mongo_backups\zai_core_test\test_unified_vector_search.py
import numpy as np
import pytest
from zmongo_retriever.zmongo_toolbag.unified_vector_search import LocalVectorSearch


class MockHNSWIndex:
    def knn_query(self, query, k):
        labels = np.array([[3, 2, 0]])  # Mock chunk indices
        distances = np.array([[0.1, 0.2, 0.3]])  # Mock distances
        return labels, distances


@pytest.fixture
def local_vector_search():
    # Create a mock instance of LocalVectorSearch with necessary attributes
    lvs = LocalVectorSearch(
        repository=None,
        collection="test_collection",
        embedding_field="embedding",
        ttl_seconds=300,
    )
    # Mock attributes used in `_search_index`
    lvs.emb_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    lvs.use_hnsw = False  # Disable HNSW for some tests
    lvs._hnsw_index = MockHNSWIndex()  # Mock HNSW index for testing
    lvs.exact_rescore = False
    return lvs


def test_search_index_empty_embeddings(local_vector_search):
    local_vector_search.emb_matrix = np.array([])
    query_vector = np.array([0.1, 0.2], dtype=np.float32)
    indices, scores = local_vector_search._search_index(query_vector, k=3)
    assert indices.shape == (0,)
    assert scores.shape == (0,)


def test_search_index_no_k_provided(local_vector_search):
    query_vector = np.array([0.1, 0.2], dtype=np.float32)
    indices, scores = local_vector_search._search_index(query_vector, k=0)
    assert indices.shape == (0,)
    assert scores.shape == (0,)


def test_search_index_exact_search(local_vector_search):
    query_vector = np.array([0.5, 0.6], dtype=np.float32)
    indices, scores = local_vector_search._search_index(query_vector, k=2)
    assert indices.shape == (2,)
    assert scores.shape == (2,)
    assert np.allclose(scores, [0.61, 0.53], atol=1e-2)


def test_search_index_hnsw_no_rescore(local_vector_search):
    local_vector_search.use_hnsw = True
    local_vector_search.exact_rescore = False
    query_vector = np.array([0.5, 0.6], dtype=np.float32)
    indices, scores = local_vector_search._search_index(query_vector, k=3)
    assert indices.shape == (3,)
    assert scores.shape == (3,)
    assert np.allclose(scores, [0.9, 0.8, 0.7], atol=1e-2)


def test_search_index_hnsw_with_rescore(local_vector_search):
    local_vector_search.use_hnsw = True
    local_vector_search.exact_rescore = True
    query_vector = np.array([0.5, 0.6], dtype=np.float32)
    indices, scores = local_vector_search._search_index(query_vector, k=3)
    assert indices.shape == (3,)
    assert scores.shape == (3,)
    assert np.allclose(scores, [0.93, 0.91, 0.89], atol=1e-2)
