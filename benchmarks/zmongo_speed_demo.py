import asyncio
import random
import string
import time
import json
import csv
from datetime import datetime
from pymongo import UpdateOne
from zmongo_toolbag.zmongo import ZMongo  # Adjust the import path as needed

COLLECTION_NAME = "benchmark_test_zmongo"
NUM_DOCUMENTS = 5000
BATCH_SIZE = 1000
EXPORT_PREFIX = "zmongo_benchmark"

def generate_documents(num):
    return [
        {
            "_id": f"doc_{i}_{''.join(random.choices(string.ascii_letters, k=6))}",
            "name": ''.join(random.choices(string.ascii_letters, k=10)),
            "created_at": datetime.utcnow(),
            "value": random.randint(1, 100000)
        }
        for i in range(num)
    ]

def export_results_csv(data: dict, filename: str):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for key, value in data.items():
            writer.writerow([key, value])

def export_results_json(data: dict, filename: str):
    with open(filename, "w") as jsonfile:
        json.dump(data, jsonfile, indent=2, default=str)

async def run_zmongo_benchmark():
    z = ZMongo()
    docs = generate_documents(NUM_DOCUMENTS)

    print("⏳ Inserting documents...")
    start = time.perf_counter()
    insert_result = await z.insert_documents(COLLECTION_NAME, docs, batch_size=BATCH_SIZE, use_cache=False)
    elapsed_insert = time.perf_counter() - start

    print("🔁 Updating documents with bulk_write...")
    update_ops = [UpdateOne({"_id": d["_id"]}, {"$set": {"status": "processed"}}, upsert=True) for d in docs]
    start = time.perf_counter()
    await z.bulk_write(COLLECTION_NAME, update_ops)
    elapsed_update = time.perf_counter() - start

    print("📋 Fetching field names...")
    start = time.perf_counter()
    fields = await z.get_field_names(COLLECTION_NAME, sample_size=NUM_DOCUMENTS)
    elapsed_field = time.perf_counter() - start

    print("🧹 Cleaning up...")
    await z.delete_all_documents(COLLECTION_NAME)
    z.close()

    results = {
        "insert_time_sec": round(elapsed_insert, 4),
        "update_time_sec": round(elapsed_update, 4),
        "field_fetch_time_sec": round(elapsed_field, 4),
        "inserted_count": insert_result.get("inserted_count", 0),
        "index_field_names": fields
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{EXPORT_PREFIX}_{timestamp}.csv"
    json_filename = f"{EXPORT_PREFIX}_{timestamp}.json"

    export_results_csv(results, csv_filename)
    export_results_json(results, json_filename)

    print("\n📊 Benchmark Summary:")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\n📝 Results exported to: {csv_filename} and {json_filename}")

if __name__ == "__main__":
    asyncio.run(run_zmongo_benchmark())
