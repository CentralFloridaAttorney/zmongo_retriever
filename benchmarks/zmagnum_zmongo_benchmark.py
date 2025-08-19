import asyncio
import time
from datetime import datetime

from zmongo_toolbag import ZMongo, ZMagnum


async def benchmark(lib_instance, lib_name, n_docs=1000):
    collection = "speed_test_collection"
    await lib_instance.delete_all_documents(collection)
    await lib_instance.clear_cache()
    docs = [{"name": f"Item {i}", "value": i, "timestamp": datetime.utcnow()} for i in range(n_docs)]

    t0 = time.perf_counter()
    inserted = await lib_instance.insert_documents(collection, docs)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    found = await lib_instance.find_documents(collection, {"value": {"$gte": 0}}, limit=n_docs)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    update_response = await lib_instance.update_document(collection, {"name": "Item 1"}, {"$set": {"value": 999}})
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    deleted = await lib_instance.delete_all_documents(collection)
    t7 = time.perf_counter()

    return {
        "library": lib_name,
        "insert_time": t1 - t0,
        "find_time": t3 - t2,
        "update_time": t5 - t4,
        "delete_time": t7 - t6,
        "inserted_count": inserted.get("inserted_count", 0),
        "find_count": len(found)
    }


async def main():
    n_docs = 1000
    results = []

    zmongo_inst = ZMongo()
    res1 = await benchmark(zmongo_inst, "ZMongo", n_docs)
    results.append(res1)
    zmongo_inst.close()

    zmagnum_inst = ZMagnum()
    res2 = await benchmark(zmagnum_inst, "ZMagnum", n_docs)
    results.append(res2)
    zmagnum_inst.close()

    print("Speed Test Results:")
    for res in results:
        print(f"Library: {res['library']}")
        print(f"  Insert Time: {res['insert_time']:.4f} sec ({res['inserted_count']} docs)")
        print(f"  Find Time: {res['find_time']:.4f} sec ({res['find_count']} docs)")
        print(f"  Update Time: {res['update_time']:.4f} sec")
        print(f"  Delete Time: {res['delete_time']:.4f} sec")
        print()


if __name__ == "__main__":
    asyncio.run(main())
