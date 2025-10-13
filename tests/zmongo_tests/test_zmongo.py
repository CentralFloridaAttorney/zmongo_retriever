import pytest
import os
import uuid
import time
from bson import ObjectId
from pymongo.operations import InsertOne, UpdateOne, DeleteOne

from zmongo_toolbag.zmongo import ZMongo

# --- Test Configuration ---
# Skip all tests if MONGO_URI is not set. This prevents failures in CI/CD.
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    pytest.skip("MONGO_URI environment variable not set, skipping integration tests.", allow_module_level=True)

TEST_DB_NAME = f"zmongo_test_suite_{uuid.uuid4().hex[:6]}"


# --- Fixtures ---

@pytest.fixture(scope="function")
def zmongo_instance():
    """
    Provides a ZMongo instance for a single test function and handles cleanup.
    This ensures complete isolation between tests.
    """
    # Use a unique DB for each test function to guarantee isolation
    db_name = f"test_db_{uuid.uuid4().hex[:6]}"
    os.environ["MONGO_DATABASE_NAME"] = db_name

    zm = ZMongo()
    yield zm, zm.db.name

    # Teardown: drop the entire test database
    zm.client.drop_database(db_name)
    zm.close()
    if "MONGO_DATABASE_NAME" in os.environ:
        del os.environ["MONGO_DATABASE_NAME"]


# --- Synchronous API Tests (`_sync` methods) ---

class TestZMongoSync:
    """Tests the synchronous (`_sync`) API meant for UI and legacy code."""

    def test_sync_insert_and_find(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "sync_test_coll"
        doc = {"_id": "doc1", "name": "sync_test"}

        insert_res = zm.insert_one_sync(collection, doc)
        assert insert_res.success

        find_res = zm.find_one_sync(collection, {"_id": "doc1"})
        assert find_res.success
        assert find_res.data["name"] == "sync_test"

    def test_sync_update_and_delete(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "sync_update_coll"
        zm.insert_one_sync(collection, {"_id": "doc2", "status": "active"})

        update_res = zm.update_one_sync(collection, {"_id": "doc2"}, {"$set": {"status": "inactive"}})
        assert update_res.success and update_res.data["modified_count"] == 1

        delete_res = zm.delete_one_sync(collection, {"_id": "doc2"})
        assert delete_res.success and delete_res.data["deleted_count"] == 1

        find_res = zm.find_one_sync(collection, {"_id": "doc2"})
        assert find_res.success and find_res.data is None


# --- Asynchronous API Tests (primary `async` methods) ---

@pytest.mark.asyncio
class TestZMongoAsync:
    """Tests the primary asynchronous API."""

    async def test_async_insert_and_find(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "async_test_coll"
        doc = {"name": "async_test"}

        insert_res = await zm.insert_one(collection, doc)
        assert insert_res.success
        doc_id = insert_res.data["inserted_id"]

        find_res = await zm.find_one(collection, {"_id": doc_id})
        assert find_res.success
        assert find_res.data["name"] == "async_test"

    async def test_async_insert_or_update(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "async_upsert_coll"

        # Test insert
        insert_res = await zm.insert_or_update(collection, {"_id": "upsert1", "version": 1})
        assert insert_res.success and insert_res.data["upserted_id"] is not None

        # Test update
        update_res = await zm.insert_or_update(collection, {"_id": "upsert1", "version": 2})
        assert update_res.success and update_res.data["modified_count"] == 1

        find_res = await zm.find_one(collection, {"_id": "upsert1"})
        assert find_res.data["version"] == 2

    async def test_async_bulk_write(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "async_bulk_coll"
        await zm.insert_one(collection, {"_id": "doc_to_delete"})

        operations = [
            InsertOne({"_id": "new_doc"}),
            UpdateOne({"_id": "new_doc"}, {"$set": {"updated": True}}),
            DeleteOne({"_id": "doc_to_delete"})
        ]

        bulk_res = await zm.bulk_write(collection, operations)
        assert bulk_res.success
        assert bulk_res.data["inserted_count"] == 1
        assert bulk_res.data["modified_count"] == 1
        assert bulk_res.data["deleted_count"] == 1


# --- Caching Logic Tests ---

@pytest.mark.asyncio
class TestZMongoCache:
    """Tests the caching functionality of the ZMongo class."""

    async def test_caching_and_invalidation(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "cache_test_coll"
        doc_id = ObjectId()

        # 1. Insert and cache the document
        await zm.insert_one(collection, {"_id": doc_id, "version": 1})
        res1 = await zm.find_one(collection, {"_id": doc_id}, cache=True)
        assert res1.data["version"] == 1

        # 2. Directly update in DB to make cache stale
        await zm.db[collection].update_one({"_id": doc_id}, {"$set": {"version": 99}})

        # 3. Fetch again, should get stale data from cache
        res2_cached = await zm.find_one(collection, {"_id": doc_id}, cache=True)
        assert res2_cached.data["version"] == 1

        # 4. Update via ZMongo, which should invalidate the cache
        await zm.update_one(collection, {"_id": doc_id}, {"version": 2})

        # 5. Fetch again, should get the fresh data (version 2)
        res3_fresh = await zm.find_one(collection, {"_id": doc_id}, cache=True)
        assert res3_fresh.data["version"] == 2

    async def test_cache_is_cleared_on_bulk_ops(self, zmongo_instance):
        zm, db_name = zmongo_instance
        collection = "cache_bulk_clear_coll"

        # Cache a document
        res = await zm.insert_one(collection, {"_id": "doc1"})
        await zm.find_one(collection, {"_id": "doc1"}, cache=True)
        assert collection in zm.caches

        # A bulk update should clear the entire collection's cache
        await zm.update_many(collection, {}, {"$set": {"updated": True}})
        assert collection not in zm.caches
