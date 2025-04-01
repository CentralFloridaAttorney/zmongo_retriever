# test_perf_hellow_zmongo.py

import asyncio
import time
from zmongo_toolbag.zmongo import ZMongo
from pymongo import InsertOne
import random
import string

NUM_DOCS = 10000
COLLECTION_NAME = "perf_test"

def random_doc():
    return {
        "name": ''.join(random.choices(string.ascii_lowercase, k=10)),
        "value": random.randint(1, 1000),
        "flag": random.choice([True, False])
    }

async def run_perf_test():
    zmongo = ZMongo()

    # --------- Insert Performance ----------
    print(f"\nInserting {NUM_DOCS} documents...")
    docs = [InsertOne(random_doc()) for _ in range(NUM_DOCS)]
    start = time.time()
    await zmongo.bulk_write(COLLECTION_NAME, docs)
    insert_time = time.time() - start
    print(f"Insert time: {insert_time:.2f}s ({NUM_DOCS / insert_time:.2f} ops/sec)")

    # --------- Query Performance ----------
    print(f"\nQuerying documents with flag=True...")
    query = {"flag": True}
    start = time.time()
    results = await zmongo.find_documents(COLLECTION_NAME, query, limit=1000)
    query_time = time.time() - start
    print(f"Query time: {query_time:.2f}s ({len(results)} results)")

    # --------- Update Performance ----------
    print(f"\nUpdating 1,000 documents...")
    to_update = results[:1000]
    start = time.time()
    for doc in to_update:
        await zmongo.update_document(COLLECTION_NAME, {"_id": doc["_id"]}, {"$set": {"updated": True}})
    update_time = time.time() - start
    print(f"Update time: {update_time:.2f}s ({1000 / update_time:.2f} ops/sec)")

    # --------- Delete Performance ----------
    print(f"\nDeleting all {NUM_DOCS} documents...")
    start = time.time()
    for doc in await zmongo.find_documents(COLLECTION_NAME, {}):
        await zmongo.delete_document(COLLECTION_NAME, {"_id": doc["_id"]})
    delete_time = time.time() - start
    print(f"Delete time: {delete_time:.2f}s ({NUM_DOCS / delete_time:.2f} ops/sec)")

    await zmongo.close()

if __name__ == "__main__":
    asyncio.run(run_perf_test())
