# hello_zmongo.py

import asyncio
import time
import random
import string
import logging
from bson import ObjectId
from pymongo import InsertOne
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

logging.basicConfig(level=logging.INFO)
NUM_DOCS = 10000
COLLECTION_NAME = "perf_test"

def random_doc():
    return {
        "name": ''.join(random.choices(string.ascii_lowercase, k=10)),
        "value": random.randint(1, 1000),
        "flag": random.choice([True, False])
    }

async def perf_test_crud(zmongo: ZMongo):
    print(f"\nInserting {NUM_DOCS} documents...")
    docs = [InsertOne(random_doc()) for _ in range(NUM_DOCS)]
    start = time.time()
    await zmongo.bulk_write(COLLECTION_NAME, docs)
    insert_time = time.time() - start
    print(f"Insert time: {insert_time:.2f}s ({NUM_DOCS / insert_time:.2f} ops/sec)")

    # ✅ Explicitly override limit
    inserted = await zmongo.find_documents(COLLECTION_NAME, {}, limit=10000)
    assert len(inserted) == NUM_DOCS, f"Expected {NUM_DOCS}, got {len(inserted)}"

    print(f"\nQuerying documents with flag=True...")
    query = {"flag": True}
    start = time.time()
    results = await zmongo.find_documents(COLLECTION_NAME, query, limit=1000)
    query_time = time.time() - start
    print(f"Query time: {query_time:.2f}s ({len(results)} results)")

    print(f"\nUpdating 1,000 documents...")
    to_update = results[:1000]
    start = time.time()
    for doc in to_update:
        await zmongo.update_document(COLLECTION_NAME, {"_id": doc["_id"]}, {"$set": {"updated": True}})
    update_time = time.time() - start
    print(f"Update time: {update_time:.2f}s ({len(to_update) / update_time:.2f} ops/sec)")

    updated_count = len(await zmongo.find_documents(COLLECTION_NAME, {"updated": True}, limit=10000))
    assert updated_count >= len(to_update)

    print(f"\nDeleting all {NUM_DOCS} documents...")
    start = time.time()
    delete_result = await zmongo.db[COLLECTION_NAME].delete_many({})
    delete_time = time.time() - start
    print(f"Delete time: {delete_time:.2f}s ({delete_result.deleted_count / delete_time:.2f} ops/sec)")

    remaining = await zmongo.find_documents(COLLECTION_NAME, {}, limit=10000)
    assert len(remaining) == 0, f"{len(remaining)} documents were not deleted"

    remaining = await zmongo.find_documents(COLLECTION_NAME, {}, limit=10000)
    assert len(remaining) == 0

async def perf_test_mixed_read_write(zmongo: ZMongo):
    print(f"\nMixed Read/Write test with 1000 ops...")

    start = time.time()
    for i in range(500):
        doc = {"msg": f"entry_{i}", "val": i}
        await zmongo.insert_document(COLLECTION_NAME, doc)
        await zmongo.find_document(COLLECTION_NAME, {"val": i})
    elapsed = time.time() - start
    print(f"Mixed ops time: {elapsed:.2f}s ({1000 / elapsed:.2f} ops/sec)")

    await zmongo.db[COLLECTION_NAME].drop()

async def perf_test_embedding(zmongo: ZMongo):
    print(f"\nEmbedding test: 100 documents")
    embedder = ZMongoEmbedder(repository=zmongo, collection=COLLECTION_NAME)

    ids = []
    for i in range(100):
        result = await zmongo.insert_document(COLLECTION_NAME, {"text": f"text {i}"})
        ids.append(result.inserted_id)

    start = time.time()
    for i, doc_id in enumerate(ids):
        await embedder.embed_and_store(doc_id, f"This is a test embedding #{i}")
    elapsed = time.time() - start
    print(f"Embedding+store time: {elapsed:.2f}s ({100 / elapsed:.2f} embeds/sec)")

    await zmongo.db[COLLECTION_NAME].drop()

async def main():
    zmongo = ZMongo()
    await perf_test_crud(zmongo)
    await perf_test_mixed_read_write(zmongo)
    await perf_test_embedding(zmongo)
    await zmongo.close()
    print("\n✅ All performance tests passed.")

if __name__ == "__main__":
    asyncio.run(main())
