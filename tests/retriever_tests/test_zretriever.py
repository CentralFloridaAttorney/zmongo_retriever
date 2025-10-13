import os
import uuid
import asyncio
import platform
from pathlib import Path
from bson import ObjectId
import pytest

# -------------------- .env loading (as requested) --------------------
def _load_env_from_home():
    """
    Load env files from:
      - ~/.resources/.env
      - ~/.resources/.secrets
    Then resolve EMBEDDING_MODEL_PATH to an absolute path under HOME
    if it's provided as a home-relative value (e.g. ".resources/models/...").
    """
    try:
        from dotenv import load_dotenv  # requires python-dotenv
    except Exception:
        load_dotenv = None

    home = Path.home()
    if load_dotenv:
        load_dotenv(home / ".resources" / ".env")
        load_dotenv(home / ".resources" / ".secrets")

    # Resolve EMBEDDING_MODEL_PATH relative to HOME if needed
    var_name = "EMBEDDING_MODEL_PATH"
    env_var_path = os.getenv(var_name)
    if env_var_path:
        env_path = Path(env_var_path)
        if not env_path.is_absolute():
            env_path = home / env_var_path
        os.environ[var_name] = str(env_path)

_load_env_from_home()

# -------------------- Async loop + gating --------------------
pytestmark = [pytest.mark.asyncio(loop_scope="session")]

MODEL_PATH_VAR = "EMBEDDING_MODEL_PATH"
MONGO_URI_VAR = "MONGO_URI"
MONGO_DB_NAME_VAR = "MONGO_DATABASE_NAME"
IS_CONFIGURED = all(os.getenv(v) for v in [MODEL_PATH_VAR, MONGO_URI_VAR, MONGO_DB_NAME_VAR])

pytestmark.append(
    pytest.mark.skipif(
        not IS_CONFIGURED,
        reason=(
            "Set EMBEDDING_MODEL_PATH, MONGO_URI, MONGO_DATABASE_NAME. "
            "This test loads ~/.resources/.env and ~/.resources/.secrets and "
            "resolves EMBEDDING_MODEL_PATH relative to your HOME."
        ),
    )
)

# Detect unsafe concurrency combo (Windows + GGUF/llama.cpp)
IS_WINDOWS = platform.system() == "Windows"
IS_GGUF = str(os.getenv(MODEL_PATH_VAR, "")).lower().endswith(".gguf")
UNSAFE_CONCURRENT_EMBED = IS_WINDOWS and IS_GGUF

# -------------------- Project imports --------------------
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import ZEmbedder, EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.zretriever import ZRetriever

EMBEDDING_FIELD = "embeddings"

def _require_ok(sr, msg: str = ""):
    assert hasattr(sr, "success"), "Expected SafeResult from ZMongo methods"
    assert sr.success, msg or f"SafeResult failed: {sr.error}"

# -------------------- Session-scoped fixtures on active loop --------------------
@pytest.fixture(scope="session")
async def repo() -> ZMongo:
    # ZMongo reads MONGO_URI / MONGO_DATABASE_NAME from env
    return ZMongo()

@pytest.fixture(scope="session")
async def embedder(repo: ZMongo) -> ZEmbedder:
    # ZEmbedder reads EMBEDDING_MODEL_PATH from env; n_ctx mirrors your demo
    return ZEmbedder(repository=repo)

# -------------------- KB fixture: unique collection; no drop --------------------
@pytest.fixture(scope="module")
async def kb(repo: ZMongo, embedder: ZEmbedder):
    collection = f"zretriever_pytest_{uuid.uuid4().hex[:8]}"

    docs = [
        {"_id": ObjectId(), "topic": "Biology",
         "text": "Mitochondria are organelles often called the powerhouse of the cell."},
        {"_id": ObjectId(), "topic": "Astronomy",
         "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System."},
        {"_id": ObjectId(), "topic": "Geology",
         "text": "The Earth's crust is divided into several large tectonic plates."},
    ]

    ins = await repo.insert_documents(collection, docs, cache=True)
    _require_ok(ins, "Insert failed")
    inserted_ids = ins.data.get("inserted_ids") or []
    assert len(inserted_ids) == len(docs), "Not all docs inserted"

    # Create embeddings so LocalVectorSearch has vectors
    for d in docs:
        sr = await embedder.get_embedding(
            collection=collection,
            document_id=d["_id"],
            text=d["text"],
            embedding_field=EMBEDDING_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        )
        _require_ok(sr, f"Embedding failed for {d['_id']}")

    # Sanity: all docs now have non-empty embeddings
    sr_have_emb = await repo.find_many(
        collection,
        {EMBEDDING_FIELD: {"$exists": True, "$ne": []}},
        limit=10
    )
    _require_ok(sr_have_emb, "Failed to verify embeddings presence")
    assert isinstance(sr_have_emb.data, list) and len(sr_have_emb.data) == 3

    yield {"collection": collection, "docs": docs, "ids": inserted_ids}

    # Targeted cleanup: delete only inserted docs (no drop)
    dele = await repo.delete_documents(collection, {"_id": {"$in": inserted_ids}})
    _require_ok(dele, "Cleanup delete failed")

# -------------------- Helper to build retriever --------------------
def _make_retriever(repo: ZMongo, embedder: ZEmbedder, collection: str, threshold: float) -> ZRetriever:
    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=collection,
        embedding_field=EMBEDDING_FIELD,
        chunked_embeddings=True,
    )
    return ZRetriever(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=collection,
        similarity_threshold=threshold,
    )

# -------------------- Tests: core behavior --------------------
async def test_retrieval_success_and_metadata_cleanup(repo: ZMongo, embedder: ZEmbedder, kb):
    retriever = _make_retriever(repo, embedder, kb["collection"], threshold=0.8)
    results = await retriever.ainvoke("What is the powerhouse of the cell?")

    assert isinstance(results, list)
    assert len(results) == 1

    doc = results[0]
    assert hasattr(doc, "page_content")
    assert hasattr(doc, "metadata")
    assert doc.page_content == kb["docs"][0]["text"]
    assert "retrieval_score" in doc.metadata
    assert EMBEDDING_FIELD not in doc.metadata

async def test_similarity_threshold_filters(repo: ZMongo, embedder: ZEmbedder, kb):
    retriever = _make_retriever(repo, embedder, kb["collection"], threshold=0.99)
    results = await retriever.ainvoke("Tell me about tectonic plates.")
    assert isinstance(results, list)
    assert len(results) == 0

async def test_irrelevant_query_returns_nothing(repo: ZMongo, embedder: ZEmbedder, kb):
    retriever = _make_retriever(repo, embedder, kb["collection"], threshold=0.8)
    results = await retriever.ainvoke("Who was the first president of France?")
    assert isinstance(results, list)
    assert len(results) == 0

async def test_zmongo_methods_return_saferesult(repo: ZMongo, kb):
    sr_find = await repo.find_many(kb["collection"], {"topic": "Biology"}, limit=5)
    _require_ok(sr_find, "find_many should return SafeResult.ok")
    assert isinstance(sr_find.data, list)
    assert any(d.get("text", "").startswith("Mitochondria") for d in sr_find.data)

    sr_count = await repo.count_documents(kb["collection"], {})
    _require_ok(sr_count, "count_documents should return SafeResult.ok")
    assert sr_count.data.get("count") == 3

# -------------------- Tests: expanded coverage --------------------
async def test_top_k_limits_results(repo: ZMongo, embedder: ZEmbedder, kb):
    """
    Ensure top_k limits results; don't depend on ranking specifics.
    """
    retriever = _make_retriever(repo, embedder, kb["collection"], threshold=0.0)
    # Prefer to set the attribute to avoid assuming constructor signature
    retriever.top_k = 1  # type: ignore[attr-defined]
    results = await retriever.ainvoke("Tell me something about Earth or planets or biology.")
    assert isinstance(results, list)
    assert len(results) <= 1
    for d in results:
        assert "retrieval_score" in d.metadata

async def test_concurrent_invocations_independent(repo: ZMongo, embedder: ZEmbedder, kb):
    """
    Smoke test for independence. On Windows+GGUF (llama.cpp), concurrent embeddings can segfault.
    We fallback to sequential invocations in that environment to keep the suite stable.
    """
    retriever = _make_retriever(repo, embedder, kb["collection"], threshold=0.8)
    q1 = "What is the powerhouse of the cell?"
    q2 = "Tell me about tectonic plates."

    if UNSAFE_CONCURRENT_EMBED:
        # Sequential fallback: still verifies no cross-talk between invocations
        r1 = await retriever.ainvoke(q1)
        r2 = await retriever.ainvoke(q2)
    else:
        r1, r2 = await asyncio.gather(
            retriever.ainvoke(q1),
            retriever.ainvoke(q2),
        )

    assert isinstance(r1, list) and len(r1) == 1
    assert r1[0].page_content == kb["docs"][0]["text"]
    assert isinstance(r2, list)
    # r2 may be 0 or 1 depending on score; require it's not >1 and has score if present
    assert len(r2) in (0, 1)
    for d in r2:
        assert "retrieval_score" in d.metadata

async def test_embeddings_present_in_db(repo: ZMongo, kb):
    """
    Verify the documents in our temp collection have persisted embeddings in Mongo.
    """
    sr = await repo.find_many(
        kb["collection"],
        {EMBEDDING_FIELD: {"$exists": True, "$ne": []}},
        limit=10,
    )
    _require_ok(sr, "find_many (embeddings) should return SafeResult.ok")
    assert isinstance(sr.data, list)
    assert len(sr.data) == 3

async def test_threshold_boundary_behavior(repo: ZMongo, embedder: ZEmbedder, kb):
    """
    With threshold=0, a relevant query should yield at least 1 result.
    With threshold=1.0, no results should pass (scores are < 1.0 in typical cosine sims).
    """
    retriever_lo = _make_retriever(repo, embedder, kb["collection"], threshold=0.0)
    got = await retriever_lo.ainvoke("What is the powerhouse of the cell?")
    assert isinstance(got, list)
    assert len(got) >= 1

    retriever_hi = _make_retriever(repo, embedder, kb["collection"], threshold=1.0)
    none = await retriever_hi.ainvoke("What is the powerhouse of the cell?")
    assert isinstance(none, list)
    assert len(none) == 0

async def test_env_model_path_is_absolute():
    """
    Ensure EMBEDDING_MODEL_PATH is absolute after our resolution step,
    matching the user's home-relative loading rules.
    """
    mp = os.getenv("EMBEDDING_MODEL_PATH")
    assert mp, "EMBEDDING_MODEL_PATH should be set by .env or environment"
    assert Path(mp).is_absolute(), "EMBEDDING_MODEL_PATH must be absolute after resolution"
