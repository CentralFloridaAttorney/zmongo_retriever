# ZMongo Toolbag — Developer Guide

A practical, programmer-facing guide to using **ZMongo**, **SafeResult**, **ZEmbedder**, **LocalVectorSearch**, and **ZRetriever** together for Mongo-backed, local-embedding retrieval.

---

## Contents

* [Quick Start](#quick-start)
* [Environment & Configuration](#environment--configuration)
* [SafeResult](#saferesult)
* [ZMongo (Async Mongo Helper + TTL Cache)](#zmongo-async-mongo-helper--ttl-cache)
* [ZEmbedder (Local llama.cpp Embeddings)](#zembedder-local-llamacpp-embeddings)
* [LocalVectorSearch (In-Memory Cosine / HNSW)](#localvectorsearch-inmemory-cosine--hnsw)
* [ZRetriever (LangChain-Compatible Retriever)](#zretriever-langchaincompatible-retriever)
* [End-to-End Example](#end-to-end-example)
* [Operational Tips & Troubleshooting](#operational-tips--troubleshooting)

---

## Quick Start

```bash
# 1) Set env (example)
export MONGO_URI="mongodb://127.0.0.1:27017"
export MONGO_DATABASE_NAME="test"
export EMBEDDING_MODEL_PATH="models/your-embedding-model.gguf"  # path is resolved from your HOME

# 2) Install deps
pip install motor pymongo bson python-dotenv llama-cpp-python numpy

# 3) Run a small sanity check (Python REPL)
python - <<'PY'
import asyncio
from zmongo_toolbag.zmongo import ZMongo

async def main():
    db = ZMongo()
    ok = await db.ping()
    print("Ping:", ok.success, ok.data)
    db.close()

asyncio.run(main())
PY
```

---

## Environment & Configuration

**Required**

* `MONGO_URI` (default: `mongodb://127.0.0.1:27017`)
* `MONGO_DATABASE_NAME` (default: `test`)
* `EMBEDDING_MODEL_PATH` (path to a local `.gguf` or `.ggml` embedding-capable model)

**Optional (ZMongo tuning)**

* `ZMONGO_CACHE_TTL` (default `300` seconds)
* `ZMONGO_NEGATIVE_CACHE_TTL` (default `30` seconds; capped by cache TTL)
* `ZMONGO_STRINGIFY_IDS` (`true`/`false`, default `true`)
* Pool/timeouts: `ZMONGO_APPNAME`, `ZMONGO_COMPRESSORS`, `ZMONGO_MAX_POOL`, `ZMONGO_MIN_POOL`, `ZMONGO_SRV_SEL_MS`, `ZMONGO_CONNECT_MS`, `ZMONGO_SOCKET_MS`, `ZMONGO_RETRY_READS`, `ZMONGO_RETRY_WRITES`

**Optional (dotenv)**

* The code will also attempt to load `~/.resources/.env_zai_core` and `~/.resources/.secrets` if present.

---

## SafeResult

A predictable, serializable wrapper for all operations.

### Create

```python
from zmongo_toolbag.data_processing import SafeResult

ok = SafeResult.ok({"x": 1})
fail = SafeResult.fail("Something went wrong", data={"y": 2})
```

### Inspect

```python
print(ok.success)     # True
print(ok.data)        # {'x': 1}
print(ok.error)       # None

print(fail.success)   # False
print(fail.error)     # "Something went wrong"
```

### Helpers

```python
# Dot-path lookup
res = SafeResult.ok({"a":{"b":[{"c": 123}]}})
print(res.get("a.b.0.c"))      # 123
print(res.get("a.b.1.c", 0))   # 0 (default)

# JSON
print(res.to_json())           # Pretty JSON string of res.data

# Round-trip BSON types back (e.g., _id strings -> ObjectId)
doc = SafeResult.ok({"_id": "68c0...abcd"})
original = doc.original()      # {'_id': ObjectId('68c0...abcd')}

# Flatten and rename keys for metadata export
res = SafeResult.ok({"case":{"title":"Foo"}} , metadata_keymap={"case.title":"doc_title"})
print(res.to_metadata())       # {'doc_title': 'Foo'}
```

---

## ZMongo (Async Mongo Helper + TTL Cache)

Lightweight, high-throughput async wrapper over Motor with per-collection TTL cache and unified `SafeResult` returns.

### Initialize

```python
from zmongo_toolbag.zmongo import ZMongo

zm = ZMongo()                   # uses env vars
# or: async with ZMongo() as zm: ...
```

### CRUD (all return SafeResult)

```python
# Insert one / many
ins1 = await zm.insert_document("users", {"name": "Alice", "age": 30})
insn = await zm.insert_documents("users", [{"name":"Bob"}, {"name":"Carol"}])

# Find one / many
f1   = await zm.find_document("users", {"_id": ins1.data["inserted_id"]})  # cache-enabled by default for _id
fm   = await zm.find_documents("users", {"age": {"$gte": 18}}, limit=50, sort={"age": 1})

# Update (auto-wraps dict payloads into {"$set": ...})
up1  = await zm.update_document("users", {"_id": ins1.data["inserted_id"]}, {"age": 31})
upn  = await zm.update_documents("users", {"name": {"$in": ["Bob","Carol"]}}, {"active": True})

# Delete
del1 = await zm.delete_document("users", {"_id": ins1.data["inserted_id"]})
deln = await zm.delete_documents("users", {"active": True})

# Bulk write (PyMongo ops)
from pymongo.operations import InsertOne, UpdateOne
ops  = [InsertOne({"n":1}), UpdateOne({"n":1}, {"$set":{"k":"v"}})]
bulk = await zm.bulk_write("bulk_demo", ops)

# Count / list / aggregate / ping
cnt  = await zm.count_documents("users", {})
cols = await zm.list_collections()
agg  = await zm.aggregate("users", [{"$match": {"age": {"$gte": 18}}}])
pong = await zm.ping()
```

### Cache behavior

* `_id` lookups (`{"_id": value}`) are cached; misses are negative-cached briefly.
* Single-doc writes invalidate that doc’s cache; many-doc writes/bulks drop the whole collection cache.
* `_id` is stringified by default in returned docs (`ZMONGO_STRINGIFY_IDS=false` to keep `ObjectId`).

---

## ZEmbedder (Local llama.cpp Embeddings)

Generates embeddings via **llama-cpp-python** and optionally **persists** chunked vectors back to Mongo.

### Initialize

```python
from zmongo_toolbag.zembedder import ZEmbedder
from zmongo_toolbag.zmongo import ZMongo

zm = ZMongo()
embedder = ZEmbedder(repository=zm, n_ctx=2048)   # needs EMBEDDING_MODEL_PATH or pass model_path=...
```

> The model is loaded once at init. Ensure the model fits in RAM/VRAM.

### Modes & Chunking

* `embedding_style="retrieval_query"` → single embedding for a short query.
* `embedding_style="retrieval_document"` → chunk long text and store vectors in Mongo.

Chunking:

* `chunk_style="paragraph"` (default) – split on blank lines.
* `chunk_style="fixed"` – sliding window by **words** (requires `chunk_size`, `overlap`).

### Query embedding

```python
vecs = await embedder.get_embedding(
    text="What is the powerhouse of the cell?",    # required
    # embedding_style defaults to "retrieval_query"
    as_safe_result=False                           # default for query style
)
qvec = vecs[0]
```

### Document embedding + persistence

```python
from bson import ObjectId

doc_id = ObjectId()
# Ensure document exists or provide text directly:
# await zm.insert_document("articles", {"_id": doc_id, "text": "Long content ..."})

res = await embedder.get_embedding(
    embedding_style="retrieval_document",
    collection="articles",
    document_id=doc_id,
    embedding_field="embeddings",
    text="Long content ...",          # or omit `text` and supply text_field="text"
    text_field="text",
    chunk_style="fixed", chunk_size=400, overlap=40,
    skip_if_present=True              # re-use existing vectors if already saved
)
if res.success:
    print("Saved", res.data["vectors_count"], "vectors to Mongo.")
```

**Returns** (document style): `SafeResult.ok({...})` with keys like
`embedding_style`, `text`, `vectors`, `vectors_count`, `dimensionality`, `from_cache`, plus
`collection`, `document_id`, `embedding_field`, `chunk_style`.
Vectors are also written to `collection/document_id[embedding_field]`.

---

## LocalVectorSearch (In-Memory Cosine / HNSW)

In-memory search over **chunked** embeddings stored in Mongo.

### Initialize

```python
from zmongo_toolbag.unified_vector_search import LocalVectorSearch

lvs = LocalVectorSearch(
    repository=zm,
    collection="articles",
    embedding_field="embeddings",
    chunked_embeddings=True,       # list-of-vectors per doc
    use_hnsw=False,                # requires hnswlib if True
    exact_rescore=True,            # when using HNSW
    score_mode="cosine"            # "cosine" | "cosine_0_1" | "distance" | "distance_0_1"
)
```

### Search

```python
hits_sr = await lvs.search(qvec, top_k=5)
if hits_sr.success:
    for h in hits_sr.data:
        print(h["retrieval_score"], h["doc_id"], h["text"][:80])
```

**Hit format** (per item):

```python
{
  "doc_id": "<str id>",
  "text": "<best chunk or document text>",
  "metadata": {...all original fields except the embedding field...},
  "retrieval_score": <float per score_mode>,
  # debug/back-compat:
  "raw_cosine": <float>, "chunk_index": <int>, "document": {...}
}
```

> The engine dedupes by document (keeps the best chunk per doc), sorts by cosine desc, and can optionally build an HNSW index.

---

## ZRetriever (LangChain-Compatible Retriever)

Wires together **ZMongo + ZEmbedder + LocalVectorSearch** to produce `langchain` `Document` objects for a query.

### Initialize

```python
from zmongo_toolbag.zretriever import ZRetriever

retriever = ZRetriever(
    repository=zm,
    embedder=embedder,
    vector_searcher=lvs,
    collection_name="articles",
    embedding_field="embeddings",
    content_field="text",
    top_k=5,
    similarity_threshold=0.8
)
```

### Use

```python
# Async
docs = await retriever.ainvoke("Which organelle is the powerhouse of the cell?")
# or Sync wrapper
docs = retriever.invoke("...")  # calls async under the hood

for d in docs:
    print(d.metadata.get("title"), d.metadata["retrieval_score"])
    print(d.page_content[:120], "...\n")
```

**What you get:** a list of `langchain.schema.Document` with:

* `page_content` = document’s `content_field` value (e.g., full `"text"`),
* `metadata` = other document fields (except content & embeddings) plus `retrieval_score`.

> Results below `similarity_threshold` are filtered out.

---

## End-to-End Example

```python
#!/usr/bin/env python3
import asyncio
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import ZEmbedder
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.zretriever import ZRetriever

async def main():
    zm = ZMongo()
    embedder = ZEmbedder(repository=zm, n_ctx=2048)

    coll = "knowledge_base"
    docs = [
        {"_id": ObjectId(), "title": "Biology",   "text": "Mitochondria are the powerhouse of the cell."},
        {"_id": ObjectId(), "title": "Astronomy", "text": "Jupiter is the fifth planet from the Sun."},
        {"_id": ObjectId(), "title": "History",   "text": "The Roman Empire spanned large parts of Europe."}
    ]
    await zm.insert_documents(coll, docs)

    # Embed & persist
    for d in docs:
        er = await embedder.get_embedding(
            embedding_style="retrieval_document",
            collection=coll, document_id=d["_id"], embedding_field="embeddings",
            text=d["text"], chunk_style="fixed", chunk_size=100, overlap=20
        )
        assert er.success, er.error

    # Vector search + retriever
    lvs = LocalVectorSearch(repository=zm, collection=coll, embedding_field="embeddings", chunked_embeddings=True)
    retriever = ZRetriever(repository=zm, embedder=embedder, vector_searcher=lvs,
                           collection_name=coll, embedding_field="embeddings", content_field="text",
                           top_k=3, similarity_threshold=0.75)

    query = "Which organelle powers the cell?"
    results = await retriever.ainvoke(query)
    print(f"Query: {query}\nHits: {len(results)}")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title')}  score={doc.metadata['retrieval_score']:.3f}")
        print("   ", doc.page_content)

    embedder.close()
    zm.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Operational Tips & Troubleshooting

* **Async everywhere:** All DB and search calls are `async`. In scripts, wrap with `asyncio.run(...)`. In frameworks, run inside the event loop.
* **Model path resolution:** `ZEmbedder` resolves `EMBEDDING_MODEL_PATH` from your home directory. Pass `model_path="C:/.../model.gguf"` explicitly on Windows if needed.
* **GPU offload:** `llama-cpp-python` can offload to GPU if your wheel supports it. Memory limits apply; adjust model size or quantization accordingly.
* **Concurrency:** The underlying llama model is not thread-safe for simultaneous embeds. Prefer a single embedder instance with serialized calls, or use a semaphore.
* **Embedding cache:** For documents, `skip_if_present=True` skips recomputation when vectors already exist at `embedding_field`.
* **ID normalization:** ZMongo will coerce valid hex strings in filters into `ObjectId`s to prevent collection scans; `_id` returned is stringified by default.
* **Cache invalidation:** Single-doc writes clear that `_id` entry; multi-doc writes/bulk clear the whole collection cache. If you need freshest reads, pass `cache=False` to `find_document(...)`.
* **Score scales:** `LocalVectorSearch.score_mode`:

  * `"cosine"` → raw cosine ∈ `[-1,1]` (higher = better)
  * `"cosine_0_1"` → mapped to `[0,1]`
  * `"distance"` → `1 − cosine` ∈ `[0,2]` (lower = better)
  * `"distance_0_1"` → `[0,1]` (lower = better)
* **Chunking choice:** Paragraph chunking is robust for prose; fixed window is predictable for long/loosely formatted text. Tune `chunk_size/overlap` for recall vs. speed.
* **LangChain:** `ZRetriever` implements `get_relevant_documents` and async `ainvoke`, so you can plug it into `RetrievalQA` directly.

---

*You’re all set. Drop this into your repo as `ZMongo-Toolbag-Guide.md`, tweak names/paths as needed, and build!*
