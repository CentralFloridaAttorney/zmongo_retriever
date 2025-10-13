import pytest
import asyncio
from bson import ObjectId

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.safe_result import SafeResult


@pytest.fixture(scope="module")
def zmongo_instance():
    """Create a shared ZMongo instance for all sync + async tests."""
    zm = ZMongo()
    yield zm
    zm.close()


# ============================================================
# Sync Tests
# ============================================================

def test_sync_insert_and_find(zmongo_instance):
    zm = zmongo_instance
    collection = "sync_test_coll"
    doc = {"_id": "doc1", "name": "sync_test"}

    zm.delete_many(collection, {})  # cleanup

    insert_res = zm.insert_one(collection, doc)
    assert insert_res.success, f"Insert failed: {insert_res.error}"
    assert insert_res.data["inserted_id"] == "doc1"

    find_res = zm.find_one(collection, {"_id": "doc1"})
    assert find_res.success
    assert find_res.data["name"] == "sync_test"


def test_sync_update_and_delete(zmongo_instance):
    zm = zmongo_instance
    collection = "sync_update_coll"
    zm.delete_many(collection, {})

    zm.insert_one(collection, {"_id": "u1", "v": 1})
    upd = zm.update_one(collection, {"_id": "u1"}, {"$set": {"v": 2}})
    assert upd.success
    assert upd.data["modified_count"] == 1

    found = zm.find_one(collection, {"_id": "u1"})
    assert found.data["v"] == 2

    deleted = zm.delete_many(collection, {"_id": "u1"})
    assert deleted.data["deleted_count"] == 1


# ============================================================
# Async Tests
# ============================================================

@pytest.mark.asyncio
async def test_async_insert_and_find(zmongo_instance):
    zm = zmongo_instance
    coll = "async_test_coll"
    await zm.delete_many_async(coll, {})
    doc = {"_id": "async_doc", "name": "async_name"}

    ins = await zm.insert_one_async(coll, doc)
    assert ins.success, ins.error
    assert ins.data["inserted_id"] == "async_doc"

    found = await zm.find_one_async(coll, {"_id": "async_doc"})
    assert found.success
    assert found.data["name"] == "async_name"


@pytest.mark.asyncio
async def test_async_update_and_delete(zmongo_instance):
    zm = zmongo_instance
    coll = "async_update_coll"
    await zm.delete_many_async(coll, {})

    await zm.insert_one_async(coll, {"_id": "a1", "x": 1})
    upd = await zm.update_one_async(coll, {"_id": "a1"}, {"$set": {"x": 9}})
    assert upd.success
    assert upd.data["modified_count"] == 1

    found = await zm.find_one_async(coll, {"_id": "a1"})
    assert found.data["x"] == 9

    delres = await zm.delete_many_async(coll, {"_id": "a1"})
    assert delres.success
    assert delres.data["deleted_count"] == 1


@pytest.mark.asyncio
async def test_async_aggregation(zmongo_instance):
    zm = zmongo_instance
    coll = "async_agg_coll"
    await zm.delete_many_async(coll, {})
    docs = [{"_id": i, "group": "A" if i < 5 else "B", "val": i} for i in range(10)]
    for d in docs:
        await zm.insert_one_async(coll, d)

    pipeline = [
        {"$match": {"group": "A"}},
        {"$group": {"_id": "$group", "sum": {"$sum": "$val"}}}
    ]

    agg = await zm.aggregate_async(coll, pipeline)
    assert agg.success
    data = agg.data[0]
    assert data["_id"] == "A"
    assert data["sum"] == sum(range(5))


# ============================================================
# Error and Cache Tests
# ============================================================

@pytest.mark.asyncio
async def test_duplicate_key_error(zmongo_instance):
    zm = zmongo_instance
    coll = "async_dup_coll"
    await zm.delete_many_async(coll, {})

    doc = {"_id": "dup1", "v": 1}
    await zm.insert_one_async(coll, doc)
    dup = await zm.insert_one_async(coll, doc)
    assert not dup.success
    assert "duplicate key" in dup.error.lower()


def test_clear_cache(zmongo_instance):
    zm = zmongo_instance
    coll = "cache_test_coll"
    zm.caches[coll] = "dummy_cache"
    zm.clear_cache(coll)
    assert coll not in zm.caches


@pytest.mark.asyncio
async def test_safe_result_integration(zmongo_instance):
    zm = zmongo_instance
    coll = "async_safe_coll"
    await zm.delete_many_async(coll, {})
    doc = {"_id": ObjectId(), "a": 123}
    ins = await zm.insert_one_async(coll, doc)
    assert isinstance(ins, SafeResult)
    assert ins.success
    fnd = await zm.find_one_async(coll, {"_id": doc["_id"]})
    assert isinstance(fnd, SafeResult)
    assert fnd.success


@pytest.mark.asyncio
async def test_async_insert_and_update_many(zmongo_instance):
    zm = zmongo_instance
    coll = "async_many_coll"
    await zm.delete_many_async(coll, {})

    docs = [{"_id": f"d{i}", "v": i} for i in range(5)]
    ins = await zm.insert_many_async(coll, docs)
    assert ins.success
    assert len(ins.data["inserted_ids"]) == 5

    upd = await zm.update_many_async(coll, {"v": {"$lt": 3}}, {"$set": {"flag": True}})
    assert upd.success
    assert upd.data["modified_count"] == 3

    found = await zm.find_one_async(coll, {"_id": "d0"})
    assert found.success and found.data["flag"] is True


def test_sync_insert_and_update_many(zmongo_instance):
    zm = zmongo_instance
    coll = "sync_many_coll"
    zm.delete_many(coll, {})

    docs = [{"_id": f"s{i}", "v": i} for i in range(5)]
    ins = zm.insert_many(coll, docs)
    assert ins.success
    assert len(ins.data["inserted_ids"]) == 5

    upd = zm.update_many(coll, {"v": {"$gte": 2}}, {"$set": {"flag": True}})
    assert upd.success
    assert upd.data["modified_count"] == 3

    fnd = zm.find_one(coll, {"_id": "s2"})
    assert fnd.data["flag"] is True
