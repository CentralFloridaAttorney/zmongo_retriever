import time
import pytest
import os
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo


# Skip tests if MONGO_URI is not set
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    pytest.skip("MONGO_URI environment variable not set, skipping integration tests.", allow_module_level=True)


@pytest.fixture(scope="function")
def zm():
    """Provides a ZMongo instance for a single test function."""
    instance = ZMongo()
    yield instance
    instance.close()


@pytest.fixture(scope="function")
def zm_short_ttl():
    """Provides a ZMongo instance with a short TTL for expiration tests."""
    instance = ZMongo(cache_ttl=1)
    yield instance
    instance.close()


@pytest.fixture(scope="function")
def collection(zm, request):
    """
    Provides a unique collection name for a test and ensures it's dropped
    after the test completes.
    """
    collection_name = request.node.name
    yield collection_name
    zm.db.drop_collection(collection_name)


class TestZMongoCacheLogic:
    """Comprehensive tests for the caching logic in ZMongo."""

    def test_cache_ttl_expiration(self, zm_short_ttl, collection):
        """Verify that a cached item expires after the specified TTL."""
        doc = {"name": "ttl_doc", "version": 1}
        insert_res = zm_short_ttl.insert_document(collection, doc)
        assert insert_res.success
        doc_id_str = insert_res.data["inserted_id"]

        # Prime the cache
        find_res_1 = zm_short_ttl.find_document(collection, {"_id": doc_id_str}, cache=True)
        assert find_res_1.success
        assert find_res_1.data["version"] == 1

        # Update directly in DB (bypass cache)
        zm_short_ttl.update_document(collection, {"_id": doc_id_str}, {"$set": {"version": 2}})

        time.sleep(2)  # allow TTL (1s) to expire

        find_res_2 = zm_short_ttl.find_document(collection, {"_id": doc_id_str}, cache=True)
        assert find_res_2.success
        assert find_res_2.data["version"] == 2, "Should fetch updated version after TTL expiration"

    def test_cache_invalidation_on_delete_document(self, zm, collection):
        """Deleting a document should remove its cached entry."""
        doc = {"name": "doc_to_delete"}
        insert_res = zm.insert_document(collection, doc)
        doc_id_str = insert_res.data["inserted_id"]

        # Cache the document
        zm.find_document(collection, {"_id": doc_id_str}, cache=True)
        assert zm.run_sync(zm._cget(collection, doc_id_str)).success

        # Delete the document
        delete_res = zm.delete_document(collection, {"_id": doc_id_str})
        assert delete_res.success
        assert delete_res.data["deleted_count"] == 1

        # Cache entry should now be gone
        cached_after_delete = zm.run_sync(zm._cget(collection, doc_id_str))
        assert cached_after_delete.data is None, "Cache entry should be cleared after delete_document"

    def test_cache_cleared_on_update_documents(self, zm, collection):
        """Bulk updates should clear the entire collection cache."""
        docs = [{"name": f"doc_{i}"} for i in range(3)]
        first_doc = zm.insert_document(collection, docs[0])
        doc_id_0 = first_doc.data["inserted_id"]

        # Cache one document
        zm.find_document(collection, {"_id": doc_id_0}, cache=True)
        assert zm.run_sync(zm._cget(collection, doc_id_0)).success

        # Update many -> should clear collection cache
        zm.update_documents(collection, {}, {"$set": {"updated": True}})
        assert collection not in zm.caches, "Collection cache should be cleared after update_many"

    def test_cache_is_collection_specific(self, zm):
        """Clearing cache for one collection shouldn't affect another."""
        coll_A = "test_collection_A"
        coll_B = "test_collection_B"
        zm.db.drop_collection(coll_A)
        zm.db.drop_collection(coll_B)

        try:
            insert_A = zm.insert_document(coll_A, {"name": "doc_A"})
            id_A = insert_A.data["inserted_id"]
            zm.find_document(coll_A, {"_id": id_A}, cache=True)

            insert_B = zm.insert_document(coll_B, {"name": "doc_B"})
            id_B = insert_B.data["inserted_id"]
            zm.find_document(coll_B, {"_id": id_B}, cache=True)

            assert zm.run_sync(zm._cget(coll_A, id_A)).success
            assert zm.run_sync(zm._cget(coll_B, id_B)).success

            zm.delete_documents(coll_A, {})

            assert coll_A not in zm.caches
            assert coll_B in zm.caches
            assert zm.run_sync(zm._cget(coll_B, id_B)).success

        finally:
            zm.db.drop_collection(coll_A)
            zm.db.drop_collection(coll_B)
