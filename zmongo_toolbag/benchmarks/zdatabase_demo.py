import asyncio
import time
import pandas as pd

from zmongo_toolbag.zdatabase import ZDatabase


async def run_demo():
    db = ZDatabase(use_cache=True)
    collection = "zdatabase_benchmark"
    test_docs = [{"_id": i, "value": i * 2} for i in range(10_000)]
    target_query = {"_id": 500}
    update_data = {"value": 9999}

    results = []

    # Clean slate
    await db.delete_document(collection, {})  # Clear single to warm up
    await db.delete_document(collection, {"_id": {"$exists": True}})

    # Insert
    t0 = time.perf_counter()
    insert_result = await db.insert_documents(collection, test_docs)
    t1 = time.perf_counter()
    results.append({
        "operation": "insert_documents",
        "duration_sec": round(t1 - t0, 6),
        "from": "mongo",
        "inserted_count": insert_result.get("inserted_count", 0)
    })

    # Cold read
    t0 = time.perf_counter()
    doc = await db.find_document(collection, target_query)
    t1 = time.perf_counter()
    results.append({
        "operation": "find_document (cold)",
        "duration_sec": round(t1 - t0, 6),
        "from": "mongo",
        "found": doc is not None
    })

    # Warm read
    t0 = time.perf_counter()
    doc_cached = await db.find_document(collection, target_query)
    t1 = time.perf_counter()
    results.append({
        "operation": "find_document (warm)",
        "duration_sec": round(t1 - t0, 6),
        "from": "cache",
        "found": doc_cached is not None
    })

    # Update
    t0 = time.perf_counter()
    update_result = await db.update_document(collection, target_query, update_data)
    t1 = time.perf_counter()
    results.append({
        "operation": "update_document",
        "duration_sec": round(t1 - t0, 6),
        "from": "mongo + cache",
        **update_result
    })

    # Delete
    t0 = time.perf_counter()
    delete_result = await db.delete_document(collection, target_query)
    t1 = time.perf_counter()
    results.append({
        "operation": "delete_document",
        "duration_sec": round(t1 - t0, 6),
        "from": "mongo + cache",
        **delete_result
    })

    # Miss after delete
    t0 = time.perf_counter()
    final_lookup = await db.find_document(collection, target_query)
    t1 = time.perf_counter()
    results.append({
        "operation": "find_document (after delete)",
        "duration_sec": round(t1 - t0, 6),
        "from": "miss",
        "found": final_lookup is not None
    })

    await db.close()

    df = pd.DataFrame(results)
    print("\n📊 ZDatabase Performance Demo\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    asyncio.run(run_demo())
