import asyncio
import time
import json
import redis
import subprocess
import random
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, UpdateOne
from zmongo_toolbag.zmongo import ZMongo  # Adjust this path if needed

# Setup clients
zmongo = ZMongo()
motor_client = AsyncIOMotorClient(zmongo.MONGO_URI)
motor_db = motor_client[zmongo.MONGO_DB_NAME]
motor_col = motor_db["benchmark"]

pymongo_client = MongoClient(zmongo.MONGO_URI)
pymongo_col = pymongo_client[zmongo.MONGO_DB_NAME]["benchmark"]

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
mongo_shell_db = zmongo.MONGO_DB_NAME
mongo_shell_col = "benchmark"


def generate_documents(n):
    return [
        {"name": f"Item {i}", "value": random.randint(0, 1000), "timestamp": datetime.utcnow()}
        for i in range(n)
    ]


def write_json_file(data, path="mongo_input.json"):
    def convert(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, default=convert)



def clear_redis():
    for key in redis_client.keys("bench:item:*"):
        redis_client.delete(key)


# Custom encoder for handling datetime serialization
def custom_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 format
    raise TypeError(f"Type {type(obj)} not serializable")


async def benchmark_zmongo(n):
    docs = generate_documents(n)
    await zmongo.delete_all_documents("benchmark")
    await zmongo.clear_cache()

    t0 = time.perf_counter()
    await zmongo.insert_documents("benchmark", docs, use_sync=False)
    t1 = time.perf_counter()

    await zmongo.find_documents("benchmark", {"value": {"$gte": 0}}, limit=1000)
    t2 = time.perf_counter()

    for i in range(min(500, n)):
        await zmongo.update_document(
            "benchmark", {"name": f"Item {i}"}, {"$set": {"value": random.randint(1000, 9999)}}
        )
    t3 = time.perf_counter()

    return {
        "engine": "ZMongo",
        "insert_time": t1 - t0,
        "find_time": t2 - t1,
        "update_time": t3 - t2,
    }


async def benchmark_motor(n):
    docs = generate_documents(n)
    await motor_col.delete_many({})
    t0 = time.perf_counter()
    await motor_col.insert_many(docs)
    t1 = time.perf_counter()

    await motor_col.find({"value": {"$gte": 0}}).to_list(length=1000)
    t2 = time.perf_counter()

    for i in range(min(500, n)):
        await motor_col.update_one(
            {"name": f"Item {i}"}, {"$set": {"value": random.randint(1000, 9999)}}
        )
    t3 = time.perf_counter()

    return {
        "engine": "Motor (Raw)",
        "insert_time": t1 - t0,
        "find_time": t2 - t1,
        "update_time": t3 - t2,
    }


def benchmark_pymongo(n):
    docs = generate_documents(n)
    pymongo_col.delete_many({})
    t0 = time.perf_counter()
    pymongo_col.insert_many(docs)
    t1 = time.perf_counter()

    list(pymongo_col.find({"value": {"$gte": 0}}).limit(1000))
    t2 = time.perf_counter()

    updates = [
        UpdateOne({"name": f"Item {i}"}, {"$set": {"value": random.randint(1000, 9999)}})
        for i in range(min(500, n))
    ]
    pymongo_col.bulk_write(updates)
    t3 = time.perf_counter()

    return {
        "engine": "PyMongo (Raw)",
        "insert_time": t1 - t0,
        "find_time": t2 - t1,
        "update_time": t3 - t2,
    }


def benchmark_redis(n):
    docs = generate_documents(n)
    clear_redis()

    t0 = time.perf_counter()
    pipe = redis_client.pipeline()
    for doc in docs:
        # Use custom_encoder to handle datetime serialization
        pipe.set(f"bench:item:{doc['name']}", json.dumps(doc, default=custom_encoder))
    pipe.execute()
    t1 = time.perf_counter()

    keys = redis_client.keys("bench:item:*")[:1000]
    pipe = redis_client.pipeline()
    for key in keys:
        pipe.get(key)
    pipe.execute()
    t2 = time.perf_counter()

    pipe = redis_client.pipeline()
    for i in range(min(500, n)):
        key = f"bench:item:Item {i}"
        data = redis_client.get(key)
        if data:
            obj = json.loads(data)
            obj["value"] = random.randint(1000, 9999)
            pipe.set(key, json.dumps(obj, default=custom_encoder))
    pipe.execute()
    t3 = time.perf_counter()

    return {
        "engine": "Redis (JSON string)",
        "insert_time": t1 - t0,
        "find_time": t2 - t1,
        "update_time": t3 - t2,
    }


def benchmark_mongo_shell(n):
    docs = generate_documents(n)
    write_json_file(docs)

    subprocess.run(
        f"mongo {mongo_shell_db} --quiet --eval \"db.{mongo_shell_col}.deleteMany({{}});\"",
        shell=True,
    )

    t0 = time.perf_counter()
    subprocess.run(
        f"mongo {mongo_shell_db} --quiet --eval \"db.{mongo_shell_col}.insertMany(JSON.parse(cat('mongo_input.json')));\"",
        shell=True,
    )
    t1 = time.perf_counter()

    subprocess.run(
        f"mongo {mongo_shell_db} --quiet --eval \"db.{mongo_shell_col}.find({{value: {{$gte: 0}}}}).limit(1000).toArray();\"",
        shell=True,
    )
    t2 = time.perf_counter()

    subprocess.run(
        f'mongo {mongo_shell_db} --quiet --eval "for (let i = 0; i < 500; i++) {{ db.{mongo_shell_col}.updateOne({{name: \\"Item \\" + i}}, {{$set: {{value: Math.floor(Math.random()*9999)}}}}); }}"',
        shell=True,
    )
    t3 = time.perf_counter()

    return {
        "engine": "Mongo Shell",
        "insert_time": t1 - t0,
        "find_time": t2 - t1,
        "update_time": t3 - t2,
    }


async def main():
    sizes = [1000, 5000]
    all_results = []

    for size in sizes:
        print(f"\\n===== Benchmarking {size} documents =====")
        all_results.append(await benchmark_zmongo(size))
        all_results.append(await benchmark_motor(size))
        all_results.append(benchmark_pymongo(size))
        all_results.append(benchmark_redis(size))
        all_results.append(benchmark_mongo_shell(size))

    print("\\n===== FINAL RESULTS =====")
    for res in all_results:
        print(res)


if __name__ == "__main__":
    asyncio.run(main())
