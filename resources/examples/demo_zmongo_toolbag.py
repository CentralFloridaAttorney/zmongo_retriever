#!/usr/bin/env python3
"""
ZMongo Toolbag — All-in-One Demo Script
=======================================

This script stitches together runnable examples for:
- SafeResult
- ZMongo (async Mongo helper with TTL cache)
- ZEmbedder (local llama.cpp embeddings)
- LocalVectorSearch (in-memory cosine / optional HNSW)
- ZRetriever (LangChain-compatible retriever)

Usage:
  1) Ensure MongoDB is reachable (defaults to mongodb://127.0.0.1:27017, DB "test").
  2) (Optional for embedding demos) Set EMBEDDING_MODEL_PATH to your .gguf model.
  3) pip install motor pymongo bson python-dotenv llama-cpp-python numpy
  4) Run: python all_in_one_zmongo_toolbag_demo.py

Notes:
  - Embedding demos are skipped if EMBEDDING_MODEL_PATH is not set or llama-cpp-python is unavailable.
  - All DB calls are async; we use asyncio.run(main()) at the bottom.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Third-party
from bson import ObjectId

# zmongo_toolbag imports (these must be available in your environment)
from zmongo_toolbag.data_processing import SafeResult, DataProcessor
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    EMBEDDING_STYLE_RETRIEVAL_QUERY,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    CHUNK_STYLE_FIXED,
    CHUNK_STYLE_PARAGRAPH,
)
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.zretriever import ZRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("all_in_one_demo")


# -------------------------------
# Helpers
# -------------------------------

def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


@dataclass
class DemoContext:
    zm: ZMongo
    embedder: Optional[ZEmbedder]
    kb_coll: str


# -------------------------------
# SafeResult Demo
# -------------------------------

def safe_result_demo() -> None:
    log.info("=== SafeResult Demo ===")
    ok = SafeResult.ok({"a": {"b": [{"c": 123}]}})
    fail = SafeResult.fail("Something went wrong", data={"y": 2})

    print("OK.success:", ok.success)
    print("OK.get('a.b.0.c'):", ok.get("a.b.0.c"))
    print("OK JSON:", ok.to_json())

    # Simulate round-trip of _id as string -> ObjectId
    s = SafeResult.ok({"_id": "68c0e9393d8d80265cb5e3a5", "foo": "bar"})
    restored = s.original()
    print("Original() restored _id type:", type(restored.get("_id")).__name__)

    # Metadata flatten+rename
    with_map = SafeResult.ok({"case": {"title": "Foo"}}, metadata_keymap={"case.title": "doc_title"})
    print("to_metadata():", json.dumps(with_map.to_metadata(), indent=2))

    print("FAIL.success:", fail.success)
    print("FAIL.error:", fail.error)
    print()


# -------------------------------
# ZMongo CRUD Demo
# -------------------------------

async def zmongo_crud_demo(zm: ZMongo) -> None:
    log.info("=== ZMongo CRUD Demo ===")
    coll = f"users_demo_{_rand_suffix()}"

    # Insert one
    ins1 = await zm.insert_document(coll, {"name": "Alice", "age": 30})
    print("insert_document:", ins1.model_dump())

    # Insert many
    insn = await zm.insert_documents(coll, [{"name": "Bob"}, {"name": "Carol"}])
    print("insert_documents:", insn.model_dump())

    # Find one (by _id, cache-enabled)
    if ins1.success:
        _id = ins1.data.get("inserted_id")
        f1 = await zm.find_one(coll, {"_id": _id})
        print("find_one (by _id):", f1.model_dump())

    # Find many (sorted)
    fm = await zm.find_many(coll, {"name": {"$in": ["Alice", "Bob", "Carol"]}}, sort={"name": 1})
    print("find_many:", {"success": fm.success, "n": len(fm.data or [])})

    # Update one (auto-wrap to $set)
    if ins1.success:
        _id = ins1.data.get("inserted_id")
        up1 = await zm.update_document(coll, {"_id": _id}, {"age": 31})
        print("update_document:", up1.model_dump())

    # Update many
    upn = await zm.update_documents(coll, {"name": {"$in": ["Bob", "Carol"]}}, {"active": True})
    print("update_documents:", upn.model_dump())

    # Count / list / aggregate
    cnt = await zm.count_documents(coll, {})
    print("count_documents:", cnt.model_dump())
    cols = await zm.list_collections()
    print("list_collections.success:", cols.success)

    agg = await zm.aggregate(coll, [{"$match": {"name": {"$exists": True}}}])
    print("aggregate.n:", len(agg.data or []))

    # Delete
    if ins1.success:
        _id = ins1.data.get("inserted_id")
        del1 = await zm.delete_document(coll, {"_id": _id})
        print("delete_document:", del1.model_dump())
    deln = await zm.delete_documents(coll, {"name": {"$in": ["Bob", "Carol"]}})
    print("delete_documents:", deln.model_dump())
    print()


# -------------------------------
# ZEmbedder Demos
# -------------------------------

def _can_embed() -> bool:
    # Skip embedder demos if no model path is set. ZEmbedder itself checks llama-cpp import.
    return bool(os.getenv("EMBEDDING_MODEL_PATH"))


async def zembedder_query_demo(ctx: DemoContext) -> None:
    if not ctx.embedder:
        log.warning("Skipping ZEmbedder query demo (no EMBEDDING_MODEL_PATH or embedder init failed).")
        return
    log.info("=== ZEmbedder Query Embedding Demo ===")
    query = "What is the powerhouse of the cell?"
    qvecs = await ctx.embedder.get_embedding(text=query, embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY)
    if qvecs and isinstance(qvecs, list) and qvecs[0]:
        print("Query vec dimension:", len(qvecs[0]))
    else:
        print("Query embedding failed or empty:", qvecs)
    print()


async def zembedder_document_demo(ctx: DemoContext) -> None:
    if not ctx.embedder:
        log.warning("Skipping ZEmbedder document demo (no EMBEDDING_MODEL_PATH or embedder init failed).")
        return
    log.info("=== ZEmbedder Document Embedding Demo ===")
    coll = f"articles_demo_{_rand_suffix()}"
    doc_id = ObjectId()

    # Insert a document with "text"
    await ctx.zm.insert_document(coll, {
        "_id": doc_id,
        "title": "Computing History",
        "text": (
            "The history of computing includes mechanical calculators, "
            "electromechanical devices, and electronic computers. "
            "The microprocessor enabled personal computers."
        )
    })

    # First-time embedding (compute + persist)
    res1 = await ctx.embedder.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=coll,
        document_id=doc_id,
        embedding_field="embeddings",
        text_field="text",
        chunk_style=CHUNK_STYLE_FIXED,
        chunk_size=60,
        overlap=10,
        skip_if_present=True,
    )
    print("Doc embed #1 success:", res1.success, "| vectors_count:", res1.data.get("vectors_count") if res1.success else None,
          "| from_cache:", res1.data.get("from_cache") if res1.success else None)

    # Second call (should hit cached vectors if present)
    res2 = await ctx.embedder.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=coll,
        document_id=doc_id,
        embedding_field="embeddings",
        text_field="text",
        skip_if_present=True,
    )
    print("Doc embed #2 success:", res2.success, "| vectors_count:", res2.data.get("vectors_count") if res2.success else None,
          "| from_cache:", res2.data.get("from_cache") if res2.success else None)
    print()


# -------------------------------
# LocalVectorSearch + ZRetriever Demos
# -------------------------------

async def local_vector_search_and_retriever_demo(ctx: DemoContext) -> None:
    if not ctx.embedder:
        log.warning("Skipping LocalVectorSearch/ZRetriever demo (no EMBEDDING_MODEL_PATH or embedder init failed).")
        return

    log.info("=== LocalVectorSearch + ZRetriever Demo ===")
    # Prepare a tiny knowledge base
    coll = ctx.kb_coll
    docs = [
        {"_id": ObjectId(), "title": "Biology", "text": "Mitochondria are the powerhouse of the cell."},
        {"_id": ObjectId(), "title": "Astronomy", "text": "Jupiter is the fifth planet from the Sun."},
        {"_id": ObjectId(), "title": "History", "text": "The Roman Empire influenced law and engineering."},
    ]
    ins = await ctx.zm.insert_documents(coll, docs)
    if not ins.success:
        print("Failed inserting KB docs:", ins.error)
        return

    # Embed each doc's text and persist
    for d in docs:
        er = await ctx.embedder.get_embedding(
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            collection=coll, document_id=d["_id"], embedding_field="embeddings",
            text=d["text"], chunk_style=CHUNK_STYLE_FIXED, chunk_size=80, overlap=10
        )
        if not er.success:
            print("Embedding failed:", d.get("title"), er.error)

    # Build LocalVectorSearch
    lvs = LocalVectorSearch(
        repository=ctx.zm,
        collection=coll,
        embedding_field="embeddings",
        chunked_embeddings=True,
        use_hnsw=False,
        exact_rescore=True,
        score_mode="cosine",  # cosine, cosine_0_1, distance, distance_0_1
    )

    # Search directly (embed a query and search)
    q = "Which organelle powers the cell?"
    qv = await ctx.embedder.get_embedding(text=q, embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY)
    if not qv or not qv[0]:
        print("Query embedding failed.")
        return
    hits_sr = await lvs.search(qv[0], top_k=5)
    print("LocalVectorSearch hits:", len(hits_sr.data) if hits_sr.success else 0)
    if hits_sr.success:
        for h in hits_sr.data:
            print(f"  score={h['retrieval_score']:.3f}  title={h['metadata'].get('title')}  text={h['text'][:60]}...")

    # ZRetriever (LangChain-compatible)
    retriever = ZRetriever(
        repository=ctx.zm,
        embedder=ctx.embedder,
        vector_searcher=lvs,
        collection_name=coll,
        embedding_field="embeddings",
        content_field="text",
        top_k=3,
        similarity_threshold=0.75,
    )
    docs_out = await retriever.ainvoke(q)
    print("ZRetriever docs:", len(docs_out))
    for i, d in enumerate(docs_out, 1):
        print(f"  {i}. {d.metadata.get('title')}  score={d.metadata['retrieval_score']:.3f}")
        print("     ", d.page_content)
    print()


# -------------------------------
# End-to-End Demo (All Steps)
# -------------------------------

async def end_to_end_demo(ctx: DemoContext) -> None:
    if not ctx.embedder:
        log.warning("Skipping End-to-End demo (no EMBEDDING_MODEL_PATH or embedder init failed).")
        return

    log.info("=== End-to-End Demo ===")
    coll = f"knowledge_base_{_rand_suffix()}"
    docs = [
        {"_id": ObjectId(), "title": "Biology", "text": "Mitochondria are the powerhouse of the cell."},
        {"_id": ObjectId(), "title": "Astronomy", "text": "Jupiter is the fifth planet from the Sun and the largest."},
        {"_id": ObjectId(), "title": "History", "text": "The Roman Empire spanned large parts of Europe."},
    ]
    ins = await ctx.zm.insert_documents(coll, docs)
    print("Inserted:", ins.success)

    # Embed
    for d in docs:
        er = await ctx.embedder.get_embedding(
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            collection=coll, document_id=d["_id"], embedding_field="embeddings",
            text=d["text"], chunk_style=CHUNK_STYLE_FIXED, chunk_size=100, overlap=20
        )
        if not er.success:
            print("Embedding failed:", d.get("title"), er.error)

    # Search + Retrieve
    lvs = LocalVectorSearch(
        repository=ctx.zm, collection=coll, embedding_field="embeddings", chunked_embeddings=True
    )
    retriever = ZRetriever(
        repository=ctx.zm, embedder=ctx.embedder, vector_searcher=lvs,
        collection_name=coll, embedding_field="embeddings", content_field="text",
        top_k=3, similarity_threshold=0.75
    )
    query = "Which organelle powers the cell?"
    results = await retriever.ainvoke(query)
    print(f"Query: {query}\nHits: {len(results)}")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title')}  score={doc.metadata['retrieval_score']:.3f}")
        print("   ", doc.page_content)
    print()


# -------------------------------
# Main
# -------------------------------

async def main():
    # Show basic env
    print("MONGO_URI:", os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
    print("MONGO_DATABASE_NAME:", os.getenv("MONGO_DATABASE_NAME", "test"))
    print("EMBEDDING_MODEL_PATH:", os.getenv("EMBEDDING_MODEL_PATH", "(not set)"))
    print()

    # 1) SafeResult demo (sync)
    safe_result_demo()

    # 2) ZMongo client
    zm = ZMongo()
    try:
        # Sanity ping
        ping = await zm.ping()
        print("Mongo ping:", ping.model_dump())

        # ZMongo CRUD
        await zmongo_crud_demo(zm)

        # 3) Initialize embedder if possible
        embedder: Optional[ZEmbedder] = None
        if _can_embed():
            try:
                embedder = ZEmbedder(repository=zm, n_ctx=2048)
            except Exception as e:
                log.warning("Embedder init failed, skipping embed demos: %s", e)
                embedder = None
        else:
            log.warning("EMBEDDING_MODEL_PATH not set; skipping embed demos.")

        ctx = DemoContext(zm=zm, embedder=embedder, kb_coll=f"kb_demo_{_rand_suffix()}")

        # 4) ZEmbedder demos
        await zembedder_query_demo(ctx)
        await zembedder_document_demo(ctx)

        # 5) LocalVectorSearch + ZRetriever
        await local_vector_search_and_retriever_demo(ctx)

        # 6) End-to-End
        await end_to_end_demo(ctx)

    finally:
        # Close embedder (closes its owned repo if any)
        if 'embedder' in locals() and embedder is not None:
            embedder.close()
        # Close Mongo
        zm.close()
        print("Closed connections. Done.")

if __name__ == "__main__":
    asyncio.run(main())
