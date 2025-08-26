import asyncio
import pytest
from bson.objectid import ObjectId
from pymongo.operations import DeleteMany, UpdateMany

from zmongo_retriever.zmongo_toolbag import ZMongo


# Pytest will automatically discover and use the pytest-asyncio plugin

@pytest.fixture(scope="function")
async def zm():
    """Provides a ZMongo instance for a single test function."""
    instance = ZMongo()
    yield instance
    instance.close()


@pytest.fixture(scope="function")
async def zm_short_ttl():
    """Provides a ZMongo instance with a short TTL for expiration tests."""
    instance = ZMongo(cache_ttl=1)
    yield instance
    instance.close()


@pytest.fixture(scope="function")
async def collection(zm):
    """
    Provides a unique collection name for a test and ensures it's dropped
    after the test completes.
    """
    collection_name = f"test_{asyncio.current_task().get_name()}"
    yield collection_name
    # Teardown: drop the collection
    await zm.db.drop_collection(collection_name)


class TestZMongoCacheLogic:
    """
    A dedicated test suite to fully evaluate the caching logic of the ZMongo class.
    This suite runs against a REAL MongoDB instance defined in your .env_local file.
    """

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, zm_short_ttl, collection):
        """Verify that a cached item expires after the specified TTL."""
        doc = {"name": "ttl_doc", "version": 1}
        insert_res = await zm_short_ttl.insert_document(collection, doc)
        assert insert_res.success
        # The returned ID is a string, which is fine for ZMongo methods
        doc_id_str = insert_res.data['inserted_id']

        # FIX: Convert string ID to ObjectId for direct database calls
        doc_id_obj = ObjectId(doc_id_str)

        # 1. Find the document to populate the cache
        find_res_1 = await zm_short_ttl.find_document(collection, {"_id": doc_id_str})
        assert find_res_1.data["version"] == 1.0

        # 2. Update the document directly in the database using the ObjectId
        await zm_short_ttl.db[collection].update_one({"_id": doc_id_obj}, {"$set": {"version": 2}})

        # 3. Wait for the cache TTL (1s) to expire
        await asyncio.sleep(2.0)  # A little over 1s is enough

        # 4. Find the document again. It should be fetched from the DB.
        find_res_2 = await zm_short_ttl.find_document(collection, {"_id": doc_id_str})
        assert find_res_2.success
        assert find_res_2.data["version"] == 2, "Should have fetched the new version after cache expired"

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_delete_document(self, zm, collection):
        """Verify that delete_document removes the specific item from the cache."""
        doc = {"name": "doc_to_delete"}
        insert_res = await zm.insert_document(collection, doc)
        doc_id = insert_res.data['inserted_id']

        # Cache the document
        await zm.find_document(collection, {"_id": doc_id})
        assert await zm._cget(collection, str(doc_id)) is not None

        # Delete the document
        delete_res = await zm.delete_document(collection, {"_id": doc_id})
        assert delete_res.success
        assert delete_res.data['deleted_count'] == 1

        # Verify it's no longer in the cache
        cached_val_after_delete = await zm._cget(collection, str(doc_id))
        assert cached_val_after_delete is None, "Cache entry should be gone after delete_document"

    @pytest.mark.asyncio
    async def test_cache_cleared_on_update_documents(self, zm, collection):
        """Verify that update_documents (many) clears the entire cache for that collection."""
        docs = [{"name": f"doc_{i}"} for i in range(3)]
        insert_res = await zm.insert_documents(collection, docs)
        doc_id_0 = insert_res.data['inserted_ids'][0]

        # Cache one of the documents
        await zm.find_document(collection, {"_id": doc_id_0})
        assert await zm._cget(collection, str(doc_id_0)) is not None

        # Run an update_many operation
        await zm.update_documents(collection, {}, {"$set": {"updated": True}})

        # The cache for the entire collection should be removed from the caches dictionary.
        assert collection not in zm.caches

    @pytest.mark.asyncio
    async def test_cache_cleared_on_delete_documents(self, zm, collection):
        """Verify that delete_documents (many) clears the entire cache for that collection."""
        docs = [{"name": f"doc_{i}"} for i in range(3)]
        insert_res = await zm.insert_documents(collection, docs)
        doc_id_0 = insert_res.data['inserted_ids'][0]
        doc_id_1 = insert_res.data['inserted_ids'][1]

        # Cache one document
        await zm.find_document(collection, {"_id": doc_id_0})
        assert await zm._cget(collection, str(doc_id_0)) is not None

        # Delete a different document using delete_many
        await zm.delete_documents(collection, {"_id": doc_id_1})

        # The cache for the entire collection should be removed.
        assert collection not in zm.caches

    @pytest.mark.asyncio
    async def test_cache_cleared_on_bulk_write(self, zm, collection):
        """Verify that bulk_write clears the entire cache for that collection."""
        doc = {"name": "bulk_doc"}
        insert_res = await zm.insert_document(collection, doc)
        doc_id = insert_res.data['inserted_id']

        # Cache the document
        await zm.find_document(collection, {"_id": doc_id})
        assert await zm._cget(collection, str(doc_id)) is not None

        # Perform a bulk write
        ops = [UpdateMany({}, {"$set": {"bulk_updated": True}})]
        await zm.bulk_write(collection, ops)

        # The cache for the entire collection should be removed.
        assert collection not in zm.caches

    @pytest.mark.asyncio
    async def test_cache_is_collection_specific(self, zm):
        """Verify that clearing the cache for one collection does not affect another."""
        coll_A = "test_collection_A"
        coll_B = "test_collection_B"
        await zm.db.drop_collection(coll_A)
        await zm.db.drop_collection(coll_B)

        try:
            # Insert and cache a document in Collection A
            insert_A = await zm.insert_document(coll_A, {"name": "doc_A"})
            id_A = insert_A.data['inserted_id']
            await zm.find_document(coll_A, {"_id": id_A})

            # Insert and cache a document in Collection B
            insert_B = await zm.insert_document(coll_B, {"name": "doc_B"})
            id_B = insert_B.data['inserted_id']
            await zm.find_document(coll_B, {"_id": id_B})

            # Confirm both are cached
            assert await zm._cget(coll_A, str(id_A)) is not None
            assert await zm._cget(coll_B, str(id_B)) is not None

            # Perform an action that clears the cache for Collection A
            await zm.delete_documents(coll_A, {})

            # Assert Collection A's cache is gone, but Collection B's remains.
            assert coll_A not in zm.caches
            assert coll_B in zm.caches, "Cache for Collection B should not be affected"
            assert await zm._cget(coll_B, str(id_B)) is not None, "Doc in B should still be cached"

        finally:
            # Cleanup extra collections
            await zm.db.drop_collection(coll_A)
            await zm.db.drop_collection(coll_B)
