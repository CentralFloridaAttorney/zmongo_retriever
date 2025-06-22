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


import pytest
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo, _sanitize_dict


def random_collection():
    import random, string
    return "test_update_" + ''.join(random.choices(string.ascii_lowercase, k=8))


@pytest.mark.asyncio
async def test_update_documents_multiple_and_upsert():
    coll = random_collection()
    zm = ZMongo()
    # Insert multiple docs
    docs = [
        {"_id": ObjectId(), "name": "alice", "role": "user"},
        {"_id": ObjectId(), "name": "bob", "role": "user"},
    ]
    await zm.insert_documents(coll, docs)

    # Update all docs with role 'user' to role 'admin'
    result = await zm.update_documents(coll, {"role": "user"}, {"role": "admin"})
    assert result.success
    # Should have matched at least 2 docs (use UpdateResult)
    update_data = result.data
    assert hasattr(update_data, "matched_count")
    assert update_data.matched_count == 2

    # Confirm the docs are updated
    found = await zm.find_documents(coll, {"role": "admin"})
    assert found.success
    assert len(found.data) == 2
    for doc in found.data:
        assert doc["role"] == "admin"

    # Test upsert: update a doc that does not exist, with upsert=True
    upsert_name = "carol"
    upsert_result = await zm.update_documents(
        coll, {"name": upsert_name}, {"role": "user", "name": upsert_name}, upsert=True
    )
    assert upsert_result.success
    # Should have upserted one document
    upsert_data = upsert_result.data
    assert hasattr(upsert_data, "upserted_id")
    assert upsert_data.upserted_id is not None

    # Confirm upserted doc exists
    found = await zm.find_documents(coll, {"name": upsert_name})
    assert found.success
    assert len(found.data) == 1
    assert found.data[0]["role"] == "user"

    # Clean up
    await zm.delete_documents(coll)


@pytest.mark.asyncio
async def test_list_collections():
    zm = ZMongo()
    coll = random_collection()
    # Insert a document to ensure the collection exists
    await zm.insert_document(coll, {"foo": "bar"})

    # Now list collections
    result = await zm.list_collections()
    assert result.success
    collection_names = result.data
    assert isinstance(collection_names, list)
    assert coll in collection_names

    # Clean up
    await zm.delete_documents(coll)


@pytest.mark.asyncio
async def test_find_documents_with_sort():
    zm = ZMongo()
    coll = random_collection()
    values = [5, 2, 9, 1, 7]
    docs = [{"_id": i, "value": v} for i, v in enumerate(values)]
    await zm.insert_documents(coll, docs)

    # Ascending sort
    result_asc = await zm.find_documents(coll, {}, sort=[("value", 1)])
    assert result_asc.success
    asc_values = [doc["value"] for doc in result_asc.data]
    assert asc_values == sorted(values)

    # Descending sort
    result_desc = await zm.find_documents(coll, {}, sort=[("value", -1)])
    assert result_desc.success
    desc_values = [doc["value"] for doc in result_desc.data]
    assert desc_values == sorted(values, reverse=True)

    # Clean up
    await zm.delete_documents(coll)


def test_sanitize_dict_collision():
    # "_foo" needs to be sanitized, but "ufoo" already exists in d,
    # so the function should keep adding "u" until it finds a free key
    d = {
        "_foo": 123,      # will be sanitized
        "ufoo": "conflict",  # initial conflict
        "uufoo": "also_conflict", # also conflict
        "foo": "safe"
    }
    result = _sanitize_dict(d)
    # Should create "uuufoo" (three "u"s) as the sanitized key
    # All other keys should be unchanged except the keymap added
    assert "uuufoo" in result
    assert result["uuufoo"] == 123
    assert "ufoo" in result and result["ufoo"] == "conflict"
    assert "uufoo" in result and result["uufoo"] == "also_conflict"
    assert "foo" in result and result["foo"] == "safe"
    # Check that the keymap is correct
    keymap = result["__keymap"]
    assert keymap["uuufoo"] == "_foo"

    # (Optional) If you want to test that _restore_dict restores it
    from zmongo_toolbag.zmongo import _restore_dict
    restored = _restore_dict(result.copy())
    assert "_foo" in restored and restored["_foo"] == 123

from zmongo_toolbag.zmongo import _restore_dict

def test_restore_dict_no_keymap():
    # Dict with no __keymap should be returned unchanged
    d = {"foo": 1, "bar": 2}
    result = _restore_dict(d.copy())
    # Should be exactly the same as original
    assert result == d
    # Should not add or remove any keys
    assert set(result.keys()) == set(d.keys())
    # Should not mutate the input
    assert d == {"foo": 1, "bar": 2}


@pytest.mark.asyncio
async def test_aggregate_limit():
    zm = ZMongo()
    coll = random_collection()
    # Insert 20 docs
    await zm.insert_documents(coll, [{"val": i} for i in range(20)])
    # Simple aggregation pipeline to return all docs
    pipeline = [{'$match': {}}]

    # Test with explicit limit less than number of docs
    limit = 5
    result = await zm.aggregate(coll, pipeline, limit=limit)
    assert result.success
    docs = result.data
    assert isinstance(docs, list)
    assert len(docs) == limit  # Should not return more than the limit

    # Test with a limit higher than doc count
    big_limit = 30
    result2 = await zm.aggregate(coll, pipeline, limit=big_limit)
    assert result2.success
    docs2 = result2.data
    assert len(docs2) == 20  # Should return all docs, not more

    # Clean up
    await zm.delete_documents(coll)

@pytest.mark.asyncio
async def test_insert_documents_empty_list():
    zm = ZMongo()
    coll = random_collection()
    # Try inserting an empty list of documents
    result = await zm.insert_documents(coll, [])
    assert result.success
    # Should return a dict with "inserted_ids" as an empty list
    assert isinstance(result.data, dict)
    assert "inserted_ids" in result.data
    assert result.data["inserted_ids"] == []

import asyncio

from pydantic import BaseModel, Field

from zmongo_toolbag import ZMongo

import pytest
import asyncio
from pydantic import BaseModel, Field
from zmongo_toolbag.zmongo import ZMongo

@pytest.mark.asyncio
async def test_pydantic_alias_roundtrip():
    zm = ZMongo()
    class Model(BaseModel):
        field: str = Field(..., alias="_field")
    obj = Model(_field="val")
    res = await zm.insert_document("t", obj)
    doc = (await zm.find_document("t", {"_field": "val"})).original()
    assert doc["_field"] == "val"
    assert Model.parse_obj(doc).field == "val"
