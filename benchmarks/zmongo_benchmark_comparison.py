# First, generate a Python script that benchmarks both Redis and MongoDB shell inserts, finds, and updates.
# We'll simulate similar operations using redis-py and pymongo (but in shell mode via subprocess for mongo shell).
import os

benchmark_script = """
import time
import random
import json
import subprocess
import redis
from pymongo import MongoClient
from datetime import datetime

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME = "benchmark_shell"
COLLECTION_NAME = "mongo_shell_test"

# Connect clients
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[DB_NAME][COLLECTION_NAME]

# Document generator
def generate_docs(n):
    return [
        {
            "name": f"Item {i}",
            "value": random.randint(0, 1000),
            "timestamp": datetime.utcnow().isoformat()
        } for i in range(n)
    ]

# Mongo shell insert
def mongo_shell_insert(docs):
    json_data = json.dumps(docs)
    with open("mongo_input.json", "w") as f:
        f.write(json_data)
    cmd = f'mongo {DB_NAME} --eval "var collection=\\"{COLLECTION_NAME}\\"; var docs=cat(\\"mongo_input.json\\"); db[collection].insertMany(JSON.parse(docs));"'
    start = time.perf_counter()
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return time.perf_counter() - start

# Redis insert as JSON strings
def redis_insert(docs):
    start = time.perf_counter()
    pipeline = redis_client.pipeline()
    for doc in docs:
        key = f"redis:item:{doc['name']}"
        pipeline.set(key, json.dumps(doc))
    pipeline.execute()
    return time.perf_counter() - start

# Mongo shell find
def mongo_shell_find(limit):
    cmd = f'mongo {DB_NAME} --eval "var docs=db.{COLLECTION_NAME}.find().limit({limit}).toArray(); printjson(docs.length);"'
    start = time.perf_counter()
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return time.perf_counter() - start

# Redis find
def redis_find(limit):
    start = time.perf_counter()
    keys = redis_client.keys("redis:item:*")[:limit]
    pipeline = redis_client.pipeline()
    for key in keys:
        pipeline.get(key)
    pipeline.execute()
    return time.perf_counter() - start

# Mongo shell update
def mongo_shell_update(n):
    cmd = f'mongo {DB_NAME} --eval "for (var i = 0; i < {n}; i++) {{ db.{COLLECTION_NAME}.updateOne({{name: \\"Item \\"+i}}, {{$set: {{value: Math.floor(Math.random()*9999)}}}}); }}"'
    start = time.perf_counter()
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return time.perf_counter() - start

# Redis update
def redis_update(n):
    start = time.perf_counter()
    pipeline = redis_client.pipeline()
    for i in range(n):
        key = f"redis:item:Item {i}"
        existing = redis_client.get(key)
        if existing:
            doc = json.loads(existing)
            doc["value"] = random.randint(0, 9999)
            pipeline.set(key, json.dumps(doc))
    pipeline.execute()
    return time.perf_counter() - start

# Delete Redis keys
def redis_clear():
    for key in redis_client.keys("redis:item:*"):
        redis_client.delete(key)

# Main benchmark
def run():
    results = []
    for size in [1000, 5000, 10000]:
        docs = generate_docs(size)

        redis_clear()
        mongo_collection.delete_many({})

        print(f"\\nSize: {size}")
        results.append({ "type": "redis_insert", "size": size, "duration": redis_insert(docs) })
        results.append({ "type": "mongo_shell_insert", "size": size, "duration": mongo_shell_insert(docs) })

        results.append({ "type": "redis_find", "size": size, "duration": redis_find(1000) })
        results.append({ "type": "mongo_shell_find", "size": size, "duration": mongo_shell_find(1000) })

        results.append({ "type": "redis_update", "size": size, "duration": redis_update(500) })
        results.append({ "type": "mongo_shell_update", "size": size, "duration": mongo_shell_update(500) })

    print("\\nBenchmark Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    run()
"""

benchmark_file_path = "/home/overlordx/PycharmProjects/zmongo_retriever/zmongo_toolbag/benchmarks/zmongo_benchmarks.py"
with open(benchmark_file_path, "w") as f:
    f.write(benchmark_script)

benchmark_file_path
