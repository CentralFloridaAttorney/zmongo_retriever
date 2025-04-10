import asyncio
import time
import pandas as pd
from zmongo_toolbag.zmongo import ZMongo  # Ensure this import path matches your local project

async def benchmark_zmongo():
    zmongo = ZMongo()
    collection = "zmongo_benchmark"
    docs = [{"_id": i, "value": i * 3} for i in range(10000)]
    query = {"_id": 5000}
    update_data = {"$set": {"value": 654321}}

    results = []

    await zmongo.delete_all_documents(collection)

    # Insert
    start = time.perf_counter()
    insert_result = await zmongo.insert_documents(collection, docs)
    duration = time.perf_counter() - start
    results.append({
        "operation": "insert_documents",
        "duration_sec": round(duration, 4),
        "from": "mongo",
        "count": insert_result.get("inserted_count", 0),
        "error": insert_result.get("errors", None)
    })

    # Find (cold)
    start = time.perf_counter()
    doc = await zmongo.find_document(collection, query)
    duration = time.perf_counter() - start
    results.append({
        "operation": "find_document (cold)",
        "duration_sec": round(duration, 6),
        "from": "mongo",
        "found": bool(doc)
    })

    # Find (warm - cache)
    start = time.perf_counter()
    doc_cached = await zmongo.find_document(collection, query)
    duration = time.perf_counter() - start
    results.append({
        "operation": "find_document (warm)",
        "duration_sec": round(duration, 6),
        "from": "cache",
        "found": bool(doc_cached)
    })

    # Update
    start = time.perf_counter()
    update_result = await zmongo.update_document(collection, query, update_data)
    duration = time.perf_counter() - start
    results.append({
        "operation": "update_document",
        "duration_sec": round(duration, 6),
        "from": "mongo + cache",
        "matched": update_result.get("matched_count", 0),
        "modified": update_result.get("modified_count", 0)
    })

    # Delete
    start = time.perf_counter()
    delete_result = await zmongo.delete_document(collection, query)
    duration = time.perf_counter() - start
    results.append({
        "operation": "delete_document",
        "duration_sec": round(duration, 6),
        "from": "mongo + cache",
        "deleted": delete_result.deleted_count if hasattr(delete_result, "deleted_count") else 0
    })

    # Find after delete (should miss)
    start = time.perf_counter()
    doc_after_delete = await zmongo.find_document(collection, query)
    duration = time.perf_counter() - start
    results.append({
        "operation": "find_document (after delete)",
        "duration_sec": round(duration, 6),
        "from": "miss",
        "found": bool(doc_after_delete)
    })

    await zmongo.delete_all_documents(collection)
    await zmongo.close()

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    asyncio.run(benchmark_zmongo())
