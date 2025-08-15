import asyncio
import time
import random
import string
from pydantic import BaseModel, Field
import motor.motor_asyncio
import tracemalloc
import sys
import psutil

from zmongo_toolbag import ZMongo

try:
    import beanie
    from beanie import Document
except ImportError:
    beanie = None
try:
    import mongoengine
except ImportError:
    mongoengine = None

COLL = "benchmark_pets"
N = 50000

def random_pet(i):
    return {
        "name": f"pet_{i}_{''.join(random.choices(string.ascii_letters, k=8))}",
        "age": random.randint(1, 25),
        "_secret": ''.join(random.choices(string.ascii_lowercase, k=12)),
        "weight": random.uniform(3.5, 15.0),
        "desc": None if i % 2 == 0 else f"desc_{i}",
        "favorite": i % 7 == 0,
        "tags": [random.choice(["a","b","c","d"]) for _ in range(random.randint(0,3))],
    }

class PetModel(BaseModel):
    name: str
    age: int
    secret: str = Field(..., alias="_secret")
    weight: float
    desc: str | None = None
    favorite: bool = False
    tags: list[str] = []

def show_mem(msg=""):
    p = psutil.Process()
    m = p.memory_info().rss / 1024**2
    print(f"    [mem: {m:.1f}MB] {msg}")

def time_and_mem_async(func):
    async def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        show_mem(f"{func.__name__}: peak {peak/1024/1024:.1f}MB, current {current/1024/1024:.1f}MB, elapsed {elapsed:.2f}s")
        return result, elapsed, peak/1024/1024
    return wrapper

def time_and_mem_sync(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        show_mem(f"{func.__name__}: peak {peak/1024/1024:.1f}MB, current {current/1024/1024:.1f}MB, elapsed {elapsed:.2f}s")
        return result, elapsed, peak/1024/1024
    return wrapper

@time_and_mem_async
async def motor_benchmark(db):
    pets = [random_pet(i) for i in range(N)]
    res = await db[COLL].insert_many(pets)
    ins_ids = res.inserted_ids
    docs = [doc async for doc in db[COLL].find({}, limit=N)]
    await db[COLL].delete_many({})
    return len(ins_ids), len(docs)

@time_and_mem_async
async def zmongo_benchmark(zm):
    pets = [random_pet(i) for i in range(N)]
    res = await zm.insert_documents(COLL, pets)
    ins_ids = res.data.inserted_ids
    docs = (await zm.find_documents(COLL, {}, limit=N)).data
    await zm.delete_documents(COLL)
    return len(ins_ids), len(docs)

@time_and_mem_async
async def zmongo_model_benchmark(zm):
    pets = [PetModel(**random_pet(i)) for i in range(N)]
    res = await zm.insert_documents(COLL, pets)
    ins_ids = res.data.inserted_ids
    docs = (await zm.find_documents(COLL, {}, limit=N)).data
    await zm.delete_documents(COLL)
    return len(ins_ids), len(docs)

@time_and_mem_async
async def beanie_benchmark():
    class BeaniePet(Document):
        name: str
        age: int
        secret: str = Field(..., alias="_secret")
        weight: float
        desc: str | None = None
        favorite: bool = False
        tags: list[str] = []
        class Settings:
            name = COLL
    motor_db = motor.motor_asyncio.AsyncIOMotorClient()["benchmarkdb"]
    await beanie.init_beanie(database=motor_db, document_models=[BeaniePet])
    await BeaniePet.get_motor_collection().delete_many({})
    pets = [BeaniePet(**random_pet(i)) for i in range(N)]
    await BeaniePet.insert_many(pets)
    docs = await BeaniePet.find_all().to_list()
    await BeaniePet.get_motor_collection().delete_many({})
    return len(pets), len(docs)

@time_and_mem_sync
def mongoengine_benchmark():
    mongoengine.connect("test", host="mongodb://localhost:27017")
    class MongoEnginePet(mongoengine.Document):
        name = mongoengine.StringField()
        age = mongoengine.IntField()
        secret = mongoengine.StringField(db_field="_secret")
        weight = mongoengine.FloatField()
        desc = mongoengine.StringField(null=True)
        favorite = mongoengine.BooleanField(default=False)
        tags = mongoengine.ListField(mongoengine.StringField())
        meta = {"collection": COLL}
    MongoEnginePet.objects.delete()
    pets = [MongoEnginePet(**{k: v for k, v in random_pet(i).items() if k != "_secret"}, secret=random_pet(i)["_secret"]) for i in range(N)]
    MongoEnginePet.objects.insert(pets, load_bulk=False)
    docs = list(MongoEnginePet.objects)
    MongoEnginePet.objects.delete()
    return len(pets), len(docs)

async def main():
    import motor as motor_version
    print(f"\n=== MongoDB Python Benchmark (N = {N}) ===")
    print(f"Python: {sys.version.split()[0]}, Pydantic: {BaseModel.__module__}, Motor: {getattr(motor_version, '__version__', None)}")
    motor_db = motor.motor_asyncio.AsyncIOMotorClient()["benchmarkdb"]
    zm = ZMongo()

    print("\n[Motor raw]")
    (m_ins, m_find), m_time, m_mem = await motor_benchmark(motor_db)
    print(f"  insert_many: {m_time:.2f}s, docs: {m_ins} | find: {m_time:.2f}s, docs: {m_find}")

    print("\n[ZMongo dict]")
    (z_ins, z_find), z_time, z_mem = await zmongo_benchmark(zm)
    print(f"  insert_documents: {z_time:.2f}s, docs: {z_ins} | find_documents: {z_time:.2f}s, docs: {z_find}")

    print("\n[ZMongo model]")
    (zm_ins, zm_find), zm_time, zm_mem = await zmongo_model_benchmark(zm)
    print(f"  insert_documents(model): {zm_time:.2f}s, docs: {zm_ins} | find_documents: {zm_time:.2f}s, docs: {zm_find}")

    if beanie is not None:
        print("\n[Beanie]")
        (b_ins, b_find), b_time, b_mem = await beanie_benchmark()
        print(f"  insert_many: {b_time:.2f}s, docs: {b_ins} | find: {b_time:.2f}s, docs: {b_find}")
    else:
        print("\n[Beanie] Not installed.")

    if mongoengine is not None:
        print("\n[MongoEngine]")
        (me_ins, me_find), me_time, me_mem = mongoengine_benchmark()
        print(f"  insert_many: {me_time:.2f}s, docs: {me_ins} | find: {me_time:.2f}s, docs: {me_find}")
    else:
        print("\n[MongoEngine] Not installed.")

if __name__ == "__main__":
    asyncio.run(main())
