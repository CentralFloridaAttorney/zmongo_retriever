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

def test_list_collections_sync(zmongo_instance):
    zm = zmongo_instance
    zm.insert_one("col_test_list", {"x": 1})  # ensure at least one exists
    res = zm.list_collections()
    assert res.success
    assert "col_test_list" in res.data["collections"]


@pytest.mark.asyncio
async def test_list_collections_async(zmongo_instance):
    zm = zmongo_instance
    await zm.insert_one_async("col_test_async_list", {"y": 1})
    res = await zm.list_collections_async()
    assert res.success
    assert "col_test_async_list" in res.data["collections"]


def test_sync_timestamp(zmongo_instance):
    zm = zmongo_instance
    res = zm.sync_timestamp()
    assert res.success, res.error
    data = res.data
    assert "server_time" in data and "latency_seconds" in data
    assert isinstance(data["latency_seconds"], float)
    assert abs(data["offset_seconds"]) < 5  # clock drift sanity

@pytest.mark.asyncio
async def test_find_many_async(zmongo_instance):
    zm = zmongo_instance
    coll = "find_many_async_test"
    await zm.delete_many_async(coll, {})
    docs = [{"_id": f"a{i}", "x": i} for i in range(5)]
    await zm.insert_many_async(coll, docs)

    res = await zm.find_many_async(coll, {"x": {"$gte": 2}}, limit=3)
    assert res.success
    assert all(d["x"] >= 2 for d in res.data)
    assert len(res.data) <= 3


def test_find_many_sync(zmongo_instance):
    zm = zmongo_instance
    coll = "find_many_sync_test"
    zm.delete_many(coll, {})
    docs = [{"_id": f"s{i}", "x": i} for i in range(5)]
    zm.insert_many(coll, docs)

    res = zm.find_many(coll, {"x": {"$lt": 3}}, sort=[("x", 1)])
    assert res.success
    xs = [d["x"] for d in res.data]
    assert xs == sorted(xs)

def test_insert_or_update_sync(zmongo_instance):
    zm = zmongo_instance
    coll = "insert_or_update_sync_test"

    # Insert new document
    res1 = zm.insert_or_update(coll, {"_id": "doc1"}, {"value": 1})
    assert res1.success
    assert "upserted_id" in res1.data

    # Update existing document
    res2 = zm.insert_or_update(coll, {"_id": "doc1"}, {"value": 2})
    assert res2.success
    assert res2.data["modified_count"] >= 0

    # Verify document updated
    found = zm.find_one(coll, {"_id": "doc1"})
    assert found.success
    assert found.data["value"] == 2


@pytest.mark.asyncio
async def test_insert_or_update_async(zmongo_instance):
    zm = zmongo_instance
    coll = "insert_or_update_async_test"

    res1 = await zm.insert_or_update_async(coll, {"_id": "docA"}, {"x": 1})
    assert res1.success

    res2 = await zm.insert_or_update_async(coll, {"_id": "docA"}, {"x": 5})
    assert res2.success

    found = await zm.find_one_async(coll, {"_id": "docA"})
    assert found.success
    assert found.data["x"] == 5

def test_count_documents_sync(zmongo_instance):
    zm = zmongo_instance
    collection = "count_docs_test"
    zm.delete_many(collection, {})

    zm.insert_many(collection, [
        {"_id": "1", "cat": "A"},
        {"_id": "2", "cat": "B"},
        {"_id": "3", "cat": "A"},
    ])

    count_res = zm.count_documents(collection, {"cat": "A"})
    assert count_res.success
    assert count_res.data == 2
