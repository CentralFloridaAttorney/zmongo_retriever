import sys
import os
import asyncio
import time
import pandas as pd
from tabulate import tabulate

# Fix import path if running from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from zmongo_toolbag.zmagnum import ZMagnum
from zmongo_toolbag.zmongo import ZMongo


async def benchmark_insert(tool, label, collection, docs):
    await tool.delete_all_documents(collection)
    start = time.perf_counter()
    result = await tool.insert_documents(collection, docs)
    duration = time.perf_counter() - start
    return {
        "tool": label,
        "operation": "insert_documents",
        "count": result.get("inserted_count", 0),
        "duration_sec": round(duration, 4),
        "error": result.get("error", None)
    }


async def benchmark_find(tool, label, collection, query):
    start = time.perf_counter()
    doc = await tool.find_document(collection, query)
    duration = time.perf_counter() - start
    return {
        "tool": label,
        "operation": "find_document",
        "count": 1 if doc else 0,
        "duration_sec": round(duration, 6),
        "error": None if doc else "Not found"
    }


async def benchmark_update(tool, label, collection, query, update):
    start = time.perf_counter()
    result = await tool.update_document(collection, query, update)
    duration = time.perf_counter() - start
    return {
        "tool": label,
        "operation": "update_document",
        "count": result.get("modified_count", 0),
        "duration_sec": round(duration, 6),
        "error": None if result.get("modified_count", 0) > 0 else "No match"
    }


async def benchmark_delete(tool, label, collection, query):
    start = time.perf_counter()
    result = await tool.delete_document(collection, query)
    duration = time.perf_counter() - start
    return {
        "tool": label,
        "operation": "delete_document",
        "count": result.deleted_count if hasattr(result, "deleted_count") else 0,
        "duration_sec": round(duration, 6),
        "error": None if result.deleted_count else "No deletion"
    }


async def gather_benchmark_results():
    collection = "benchmark_test"
    test_docs = [{"_id": i, "value": i * 2} for i in range(1000)]
    find_query = {"_id": 500}
    update_data = {"$set": {"value": -1}}

    zmag = ZMagnum(disable_cache=False)
    zmongo = ZMongo()

    results = []

    # INSERT
    results.append(await benchmark_insert(zmag, "ZMagnum", collection, test_docs))
    results.append(await benchmark_insert(zmongo, "ZMongo", collection, test_docs))

    # FIND
    results.append(await benchmark_find(zmag, "ZMagnum", collection, find_query))
    results.append(await benchmark_find(zmongo, "ZMongo", collection, find_query))

    # UPDATE
    results.append(await benchmark_update(zmag, "ZMagnum", collection, find_query, update_data))
    results.append(await benchmark_update(zmongo, "ZMongo", collection, find_query, update_data))

    # DELETE
    results.append(await benchmark_delete(zmag, "ZMagnum", collection, find_query))
    results.append(await benchmark_delete(zmongo, "ZMongo", collection, find_query))

    await zmag.close()
    await zmongo.close()

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = asyncio.run(gather_benchmark_results())
    print("\n📊 Benchmark Results:\n")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
