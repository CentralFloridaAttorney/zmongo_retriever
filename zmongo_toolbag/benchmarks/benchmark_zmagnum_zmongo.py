import asyncio
import time
import random
from pymongo import InsertOne

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmagnum import ZMagnum


async def benchmark_insert_documents(db_client, label: str, count: int = 10000, batch_size: int = 1000):
    collection = "benchmark_collection"
    documents = [{"_id": i, "value": random.random()} for i in range(count)]

    start = time.perf_counter()
    result = await db_client.insert_documents(collection, documents, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    inserted = result.get("inserted_count", 0)
    print(f"{label} - Inserted {inserted} documents in {elapsed:.4f} seconds")
    return {
        "label": label,
        "inserted": inserted,
        "time_seconds": elapsed,
        "docs_per_second": inserted / elapsed if elapsed > 0 else 0
    }


async def run_benchmark():
    zmongo = ZMongo()
    zmagnum = ZMagnum(disable_cache=True)

    results = []
    results.append(await benchmark_insert_documents(zmongo, "ZMongo"))
    results.append(await benchmark_insert_documents(zmagnum, "ZMagnum"))

    await zmongo.delete_all_documents("benchmark_collection")
    await zmagnum.delete_all_documents("benchmark_collection")

    await zmongo.close()
    await zmagnum.close()

    return results


results = asyncio.run(run_benchmark())
