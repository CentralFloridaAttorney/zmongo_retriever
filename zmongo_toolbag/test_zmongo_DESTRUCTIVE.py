import os
import asyncio
import pytest
import pytest_asyncio
from bson.objectid import ObjectId
from pymongo import UpdateOne, InsertOne

# Make sure zmongo and its dependencies are in the python path
# You might need to adjust this import based on your project structure
from zmongo import ZMongo
from safe_result import SafeResult

# --- Test Configuration ---
# Use a dedicated test database that can be safely dropped.
# Ensure your MongoDB instance is running.
TEST_DB_NAME = "zmongo_test_suite_db"
COLLECTION_NAME = "test_coll"


# --- Fixtures ---

@pytest_asyncio.fixture(scope="function")
async def zmongo_instance():
    """
    Provides a ZMongo instance connected to a clean test database for each test.
    This fixture ensures that each test function starts with a fresh slate.
    """
    # Set environment variables for the test database
    os.environ["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME

    # Initialize ZMongo with a short cache TTL for testing
    zmongo = ZMongo(cache_ttl=5)

    # --- Setup: Ensure a clean database before each test ---
    # Drop all collections in the test database
    collections = await zmongo.db.list_collection_names()
    for coll in collections:
        await zmongo.db.drop_collection(coll)

    # Clear any in-memory cache from previous runs
    await zmongo.clear_cache()

    yield zmongo

    # --- Teardown: Clean up after each test ---
    collections = await zmongo.db.list_collection_names()
    for coll in collections:
        await zmongo.db.drop_collection(coll)
    await zmongo.clear_cache()


# --- Test Cases ---

@pytest.mark.asyncio
async def test_initialization(zmongo_instance: ZMongo):
    """Tests that the ZMongo instance is created and connected."""
    assert zmongo_instance is not None
    assert zmongo_instance.db.name == TEST_DB_NAME


@pytest.mark.asyncio
async def test_insert_and_find_document(zmongo_instance: ZMongo):
    """Tests inserting a single document and finding it."""
    doc = {"name": "test_doc", "value": 123}
    insert_res = await zmongo_instance.insert_document(COLLECTION_NAME, doc)
    assert insert_res.success
    inserted_id = insert_res.data.inserted_id

    find_res = await zmongo_instance.find_document(COLLECTION_NAME, {"_id": inserted_id})
    assert find_res.success
    assert find_res.data["name"] == "test_doc"
    assert find_res.data["value"] == 123


@pytest.mark.asyncio
async def test_caching_and_cache_eviction(zmongo_instance: ZMongo):
    """Tests that documents are cached and cache is evicted on update/delete."""
    doc = {"name": "cached_doc", "value": 456}
    insert_res = await zmongo_instance.insert_document(COLLECTION_NAME, doc)
    inserted_id = insert_res.data.inserted_id

    # 1. Find the document to cache it
    await zmongo_instance.find_document(COLLECTION_NAME, {"_id": inserted_id})

    # 2. Verify it's in the cache
    cache = zmongo_instance._get_cache(COLLECTION_NAME)
    assert await cache.get(str(inserted_id)) is not None

    # 3. Update the document and check if cache is evicted
    await zmongo_instance.update_document(COLLECTION_NAME, {"_id": inserted_id}, {"$set": {"value": 789}})
    assert await cache.get(str(inserted_id)) is None, "Cache should be evicted after update"

    # 4. Re-cache and then delete
    await zmongo_instance.find_document(COLLECTION_NAME, {"_id": inserted_id})
    assert await cache.get(str(inserted_id)) is not None, "Document should be re-cached"
    await zmongo_instance.delete_document(COLLECTION_NAME, {"_id": inserted_id})
    assert await cache.get(str(inserted_id)) is None, "Cache should be evicted after delete"


@pytest.mark.asyncio
async def test_key_sanitization(zmongo_instance: ZMongo):
    """Tests that keys starting with '_' are sanitized and restored correctly."""
    doc = {"name": "private_fields", "_secret": "classified", "_internal_id": "abc"}
    insert_res = await zmongo_instance.insert_document(COLLECTION_NAME, doc)
    inserted_id = insert_res.data.inserted_id

    # Verify that the raw document in the DB has sanitized keys
    raw_doc = await zmongo_instance.db[COLLECTION_NAME].find_one({"_id": inserted_id})
    assert "usecret" in raw_doc
    assert "uinternal_id" in raw_doc
    assert "_secret" not in raw_doc

    # Verify that find_document restores the original keys
    find_res = await zmongo_instance.find_document(COLLECTION_NAME, {"_id": inserted_id})
    assert find_res.success
    assert find_res.data["_secret"] == "classified"
    assert find_res.data["_internal_id"] == "abc"
    assert "usecret" not in find_res.data


@pytest.mark.asyncio
async def test_update_and_upsert_document(zmongo_instance: ZMongo):
    """Tests updating and upserting a document."""
    doc = {"name": "updatable", "version": 1}
    insert_res = await zmongo_instance.insert_document(COLLECTION_NAME, doc)
    inserted_id = insert_res.data.inserted_id

    # Test standard update
    update_res = await zmongo_instance.update_document(COLLECTION_NAME, {"_id": inserted_id}, {"$set": {"version": 2}})
    assert update_res.success
    assert update_res.data.matched_count == 1

    find_res = await zmongo_instance.find_document(COLLECTION_NAME, {"_id": inserted_id})
    assert find_res.data["version"] == 2

    # Test upsert
    upsert_query = {"name": "new_doc"}
    upsert_data = {"$set": {"version": 1, "is_new": True}}
    upsert_res = await zmongo_instance.update_document(COLLECTION_NAME, upsert_query, upsert_data, upsert=True)
    assert upsert_res.success
    assert upsert_res.data.upserted_id is not None

    find_upserted_res = await zmongo_instance.find_document(COLLECTION_NAME, {"name": "new_doc"})
    assert find_upserted_res.data["is_new"] is True


@pytest.mark.asyncio
async def test_delete_documents(zmongo_instance: ZMongo):
    """Tests deleting one and multiple documents."""
    docs = [{"name": "to_delete", "id": 1}, {"name": "to_delete", "id": 2}]
    await zmongo_instance.insert_documents(COLLECTION_NAME, docs)

    # Delete one
    delete_one_res = await zmongo_instance.delete_document(COLLECTION_NAME, {"id": 1})
    assert delete_one_res.success
    assert delete_one_res.data.deleted_count == 1

    count_res = await zmongo_instance.count_documents(COLLECTION_NAME, {})
    assert count_res.data["count"] == 1

    # Delete remaining
    delete_many_res = await zmongo_instance.delete_documents(COLLECTION_NAME, {"name": "to_delete"})
    assert delete_many_res.success
    assert delete_many_res.data.deleted_count == 1

    count_res = await zmongo_instance.count_documents(COLLECTION_NAME, {})
    assert count_res.data["count"] == 0


@pytest.mark.asyncio
async def test_list_and_count_collections(zmongo_instance: ZMongo):
    """Tests listing collections and counting documents."""
    await zmongo_instance.insert_document("coll1", {"a": 1})
    await zmongo_instance.insert_document("coll2", {"b": 1})

    list_res = await zmongo_instance.list_collections()
    assert list_res.success
    assert set(list_res.data) == {"coll1", "coll2"}

    count_res = await zmongo_instance.count_documents("coll1", {})
    assert count_res.success
    assert count_res.data["count"] == 1


@pytest.mark.asyncio
async def test_bulk_write_operation(zmongo_instance: ZMongo):
    """Tests a complex bulk_write operation."""
    # Start with one document
    insert_res = await zmongo_instance.insert_document(COLLECTION_NAME, {"name": "bulk_doc", "status": "original"})
    doc_id = insert_res.data.inserted_id

    operations = [
        InsertOne({"name": "new_bulk_doc"}),
        UpdateOne({"_id": doc_id}, {"$set": {"status": "updated"}}),
    ]

    bulk_res = await zmongo_instance.bulk_write(COLLECTION_NAME, operations)
    assert bulk_res.success
    assert bulk_res.data.inserted_count == 1
    assert bulk_res.data.modified_count == 1

    updated_doc = await zmongo_instance.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert updated_doc.data["status"] == "updated"

    count_res = await zmongo_instance.count_documents(COLLECTION_NAME, {})
    assert count_res.data["count"] == 2


@pytest.mark.asyncio
async def test_aggregation_pipeline(zmongo_instance: ZMongo):
    """Tests a simple aggregation pipeline."""
    docs = [
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
        {"category": "B", "value": 30},
    ]
    await zmongo_instance.insert_documents(COLLECTION_NAME, docs)

    pipeline = [
        {"$group": {"_id": "$category", "total": {"$sum": "$value"}}},
        {"$sort": {"_id": 1}}
    ]

    agg_res = await zmongo_instance.aggregate(COLLECTION_NAME, pipeline)
    assert agg_res.success
    assert len(agg_res.data) == 2
    assert agg_res.data[0] == {"_id": "A", "total": 30}
    assert agg_res.data[1] == {"_id": "B", "total": 30}


@pytest.mark.asyncio
async def test_find_documents_with_sort_and_limit(zmongo_instance: ZMongo):
    """Tests find_documents with sorting and limit."""
    docs = [{"val": i} for i in range(10)]
    await zmongo_instance.insert_documents(COLLECTION_NAME, docs)

    # Test with limit
    find_res_limit = await zmongo_instance.find_documents(COLLECTION_NAME, {}, limit=5)
    assert find_res_limit.success
    assert len(find_res_limit.data) == 5

    # Test with sort
    find_res_sort = await zmongo_instance.find_documents(COLLECTION_NAME, {}, sort=[("val", -1)])
    assert find_res_sort.success
    assert find_res_sort.data[0]["val"] == 9
    assert find_res_sort.data[-1]["val"] == 0

