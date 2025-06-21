import pytest
import asyncio
import random
import string
from pydantic import BaseModel, Field
from zmongo_toolbag import ZMongo

def random_collection():
    return "test_zmongo_" + ''.join(random.choices(string.ascii_lowercase, k=8))


class Pet(BaseModel):
    name: str
    age: int
    secret: str = Field(..., alias="_secret")


@pytest.mark.asyncio
async def test_insert_and_find_dict():
    zm = ZMongo()
    coll = random_collection()
    doc = {"name": "Whiskers", "age": 3, "_secret": "purr"}
    ins = await zm.insert_document(coll, doc)
    assert ins.success
    _id = ins.data.inserted_id
    res = await zm.find_document(coll, {"_id": _id})
    found = res.data
    assert found["name"] == "Whiskers"
    assert found["age"] == 3
    # Check alias restored
    assert "_secret" in found
    # Clean up
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_insert_and_find_model():
    zm = ZMongo()
    coll = random_collection()
    pet = Pet(name="Rex", age=5, _secret="bowwow")   # Use secret=... not _secret=...
    ins = await zm.insert_document(coll, pet)
    assert ins.success
    _id = ins.data.inserted_id
    res = await zm.find_document(coll, {"_id": _id})
    found = res.data
    assert found["name"] == "Rex"
    assert found["_secret"] == "bowwow"
    await zm.delete_documents(coll)


@pytest.mark.asyncio
async def test_insert_many_and_find_many():
    zm = ZMongo()
    coll = random_collection()
    pets = [Pet(name=f"pet{i}", age=i, _secret=f"s{i}") for i in range(5)]
    ins = await zm.insert_documents(coll, pets)
    assert ins.success
    assert len(ins.data.inserted_ids) == 5
    found = await zm.find_documents(coll, {}, limit=10)
    names = [d["name"] for d in found.data]
    assert set(names) == set(f"pet{i}" for i in range(5))
    # Clean up
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_update_document():
    zm = ZMongo()
    coll = random_collection()
    doc = {"name": "Buddy", "age": 1, "_secret": "woof"}
    ins = await zm.insert_document(coll, doc)
    _id = ins.data.inserted_id
    upd = await zm.update_document(coll, {"_id": _id}, {"age": 2, "_secret": "yap"})
    assert upd.success
    res = await zm.find_document(coll, {"_id": _id})
    assert res.data["age"] == 2
    assert res.data["_secret"] == "yap"
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_delete_document_and_documents():
    zm = ZMongo()
    coll = random_collection()
    await zm.insert_documents(coll, [{"x": i} for i in range(3)])
    del_one = await zm.delete_document(coll, {"x": 0})
    assert del_one.success
    found = await zm.find_documents(coll, {}, limit=10)
    assert len(found.data) == 2
    del_all = await zm.delete_documents(coll)
    assert del_all.success
    found2 = await zm.find_documents(coll, {}, limit=10)
    assert len(found2.data) == 0

@pytest.mark.asyncio
async def test_cache_and_key_alias_restore():
    zm = ZMongo()
    coll = random_collection()
    doc = {"name": "Tiger", "age": 7, "_secret": "stripe"}
    await zm.insert_document(coll, doc)
    # The first find caches, second triggers cache hit
    res1 = await zm.find_document(coll, {"name": "Tiger"})
    res2 = await zm.find_document(coll, {"name": "Tiger"})
    assert res2.success
    # Check key alias restored both times
    assert res1.data["_secret"] == "stripe"
    assert res2.data["_secret"] == "stripe"
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_bulk_write():
    zm = ZMongo()
    coll = random_collection()
    from pymongo.operations import InsertOne, DeleteMany
    ops = [InsertOne({"foo": i}) for i in range(5)]
    bulk_res = await zm.bulk_write(coll, ops)
    assert bulk_res.success
    found = await zm.find_documents(coll, {}, limit=10)
    assert len(found.data) == 5
    # Now bulk delete
    ops2 = [DeleteMany({})]
    await zm.bulk_write(coll, ops2)
    found2 = await zm.find_documents(coll, {}, limit=10)
    assert len(found2.data) == 0

@pytest.mark.asyncio
async def test_aggregate_and_count():
    zm = ZMongo()
    coll = random_collection()
    await zm.insert_documents(coll, [{"cat": "A"}, {"cat": "B"}, {"cat": "A"}])
    # Aggregate count by category
    pipe = [{"$group": {"_id": "$cat", "count": {"$sum": 1}}}]
    agg = await zm.aggregate(coll, pipe)
    counts = {d["_id"]: d["count"] for d in agg.data}
    assert counts["A"] == 2
    assert counts["B"] == 1
    cnt = await zm.count_documents(coll, {"cat": "A"})
    assert cnt.data["count"] == 2
    await zm.delete_documents(coll)
