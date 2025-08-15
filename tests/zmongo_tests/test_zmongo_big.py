import pytest
import time
import motor.motor_asyncio
import pymongo
import random
import string
from pydantic import BaseModel, Field

from zmongo_toolbag import ZMongo

COLL = "perf_demo_pets_big"
N = 50000   # Test with 50,000 docs for realistic perf

# --- Helpers for random data ---
def random_pet(i):
    return {
        "name": f"pet_{i}_{''.join(random.choices(string.ascii_letters, k=8))}",
        "age": random.randint(1, 25),
        "_secret": ''.join(random.choices(string.ascii_lowercase, k=12)),
        "weight": random.uniform(3.5, 15.0),
    }

class PetModel(BaseModel):
    name: str
    age: int
    secret: str = Field(alias="_secret")
    weight: float

@pytest.mark.asyncio
async def test_zmongo_perf_big():
    zm = ZMongo()
    print("\n=== ZMongo Big Data Performance Test ===\n")

    # Clean start
    await zm.delete_documents(COLL)
    pets = [random_pet(i) for i in range(N)]

    # Insert many (dict)
    t0 = time.perf_counter()
    res1 = await zm.insert_documents(COLL, pets)
    t_insert = time.perf_counter() - t0
    print(f"insert_documents (dict x{N}): {t_insert:.2f}s, ids: {len(res1.data.inserted_ids)}")

    # Insert many (Pydantic model)
    pets_model = [PetModel(**random_pet(i)) for i in range(N, N*2)]
    t0 = time.perf_counter()
    res2 = await zm.insert_documents(COLL, pets_model)
    t_insert_model = time.perf_counter() - t0
    print(f"insert_documents (model x{N}): {t_insert_model:.2f}s, ids: {len(res2.data.inserted_ids)}")

    # Find many
    t0 = time.perf_counter()
    found = await zm.find_documents(COLL, {}, limit=N*2)
    t_find = time.perf_counter() - t0
    print(f"find_documents ({N*2}): {t_find:.2f}s, found: {len(found.data)}")

    # Raw Motor test
    raw_coll = motor.motor_asyncio.AsyncIOMotorClient()[zm.db.name][COLL]
    t0 = time.perf_counter()
    cursor = raw_coll.find({}, limit=N*2)
    raw = [doc async for doc in cursor]
    t_raw = time.perf_counter() - t0
    print(f"motor find (raw) ({N*2}): {t_raw:.2f}s, found: {len(raw)}")

    # PyMongo test
    sync_coll = pymongo.MongoClient()[zm.db.name][COLL]
    t0 = time.perf_counter()
    raw_sync = list(sync_coll.find({}, limit=N*2))
    t_pymongo = time.perf_counter() - t0
    print(f"pymongo find (sync) ({N*2}): {t_pymongo:.2f}s, found: {len(raw_sync)}")

    # Clean up
    await zm.delete_documents(COLL)
    assert len(found.data) == N*2
    assert len(raw) == N*2
    assert len(raw_sync) == N*2

    print("\nSummary:")
    print(f"insert_documents (dict): {t_insert:.2f}s | (model): {t_insert_model:.2f}s | zmongo find: {t_find:.2f}s")
    print(f"motor raw find: {t_raw:.2f}s | pymongo sync find: {t_pymongo:.2f}s")

