import asyncio
import random
import string
import time
import json
import csv
from datetime import datetime
from pymongo import InsertOne, UpdateOne

from zmongo_toolbag import ZMagnum

COLLECTION_NAME = "benchmark_test_zmagnum"
NUM_DOCUMENTS = 5000
BATCH_SIZE = 1000
EXPORT_PREFIX = "zmagnum_benchmark"

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

async def run_zmagnum_benchmark():
    z = ZMagnum(disable_cache=True)
    docs = generate_documents(NUM_DOCUMENTS)

    start = time.perf_counter()
    insert_result = await z.insert_documents(COLLECTION_NAME, docs, batch_size=BATCH_SIZE)
    elapsed_insert = time.perf_counter() - start

    update_ops = [UpdateOne({"_id": d["_id"]}, {"$set": {"status": "processed"}}, upsert=True) for d in docs]
    start = time.perf_counter()
    update_result = await z.bulk_write(COLLECTION_NAME, update_ops)
    elapsed_update = time.perf_counter() - start

    start = time.perf_counter()
    index_suggestion = await z.recommend_indexes(COLLECTION_NAME, sample_size=NUM_DOCUMENTS)
    elapsed_index = time.perf_counter() - start

    await z.db[COLLECTION_NAME].drop()
    await z.close()

    results = {
        "insert_time_sec": round(elapsed_insert, 4),
        "update_time_sec": round(elapsed_update, 4),
        "index_recommend_time_sec": round(elapsed_index, 4),
        "inserted_count": insert_result.get("inserted_count", 0),
        "bulk_matched_count": update_result.get("matched_count", 0),
        "bulk_modified_count": update_result.get("modified_count", 0),
        "bulk_upserted_count": update_result.get("upserted_count", 0),
        "index_recommendation_fields": list(index_suggestion.keys())
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
    asyncio.run(run_zmagnum_benchmark())
