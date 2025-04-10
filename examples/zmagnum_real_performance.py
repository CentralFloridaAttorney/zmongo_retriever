import asyncio
import time
import pandas as pd
from zmongo_toolbag.zmagnum import ZMagnum

async def benchmark_zmagnum():
    zmag = ZMagnum(disable_cache=False)
    collection = "zmagnum_benchmark"
    docs = [{"_id": i, "value": i * 3} for i in range(10000)]
    query = {"_id": 5000}
    update_data = {"$set": {"value": 123456}}

    results = []

    # Clean up
    await zmag.delete_all_documents(collection)

    # -------------------------------
    # Insert Documents
    # -------------------------------
    start = time.perf_counter()
    insert_result = await zmag.insert_documents(collection, docs)
    insert_time = time.perf_counter() - start

    results.append({
        "operation": "insert_documents",
        "duration_sec": round(insert_time, 4),
        "inserted_count": insert_result.get("inserted_count", 0),
        "from": "mongo",
        "error": insert_result.get("error")
    })

    # -------------------------------
    # Find Document (Cold - Mongo Read)
    # -------------------------------
    start = time.perf_counter()
    doc = await zmag.find_document(collection, query)
    cold_read_time = time.perf_counter() - start

    results.append({
        "operation": "find_document (cold)",
        "duration_sec": round(cold_read_time, 6),
        "found": bool(doc),
        "from": "mongo",
        "error": None if doc else "Not found"
    })

    # -------------------------------
    # Find Document Again (Warm - Cache Read)
    # -------------------------------
    start = time.perf_counter()
    doc_cached = await zmag.find_document(collection, query)
    warm_read_time = time.perf_counter() - start

    results.append({
        "operation": "find_document (warm)",
        "duration_sec": round(warm_read_time, 6),
        "found": bool(doc_cached),
        "from": "cache" if doc_cached else "mongo",
        "error": None if doc_cached else "Not found"
    })

    # -------------------------------
    # Update Document (Mongo + Cache Write)
    # -------------------------------
    start = time.perf_counter()
    update_result = await zmag.update_document(collection, query, update_data)
    update_time = time.perf_counter() - start

    results.append({
        "operation": "update_document",
        "duration_sec": round(update_time, 6),
        "matched": update_result.get("matched_count", 0),
        "modified": update_result.get("modified_count", 0),
        "from": "mongo + cache",
        "error": update_result.get("error")
    })

    # -------------------------------
    # Delete Document (Mongo + Cache Invalidation)
    # -------------------------------
    start = time.perf_counter()
    delete_result = await zmag.delete_document(collection, query)
    delete_time = time.perf_counter() - start

    results.append({
        "operation": "delete_document",
        "duration_sec": round(delete_time, 6),
        "deleted": delete_result.get("deleted_count", 0),
        "from": "mongo + cache",
        "error": delete_result.get("error")
    })

    # -------------------------------
    # Final Read (Cache should miss, Mongo should miss)
    # -------------------------------
    start = time.perf_counter()
    doc_after_delete = await zmag.find_document(collection, query)
    post_delete_read_time = time.perf_counter() - start

    results.append({
        "operation": "find_document (after delete)",
        "duration_sec": round(post_delete_read_time, 6),
        "found": bool(doc_after_delete),
        "from": "cache" if doc_after_delete else "miss",
        "error": None if doc_after_delete else "Not found"
    })

    # Cleanup
    await zmag.delete_all_documents(collection)
    await zmag.close()

    # Results as DataFrame
    df = pd.DataFrame(results)
    print("\n📊 ZMagnum Performance Benchmark\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(benchmark_zmagnum())
