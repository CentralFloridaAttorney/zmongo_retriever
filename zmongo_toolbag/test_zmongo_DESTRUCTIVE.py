import asyncio
import os

import pytest
import pytest_asyncio
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, InsertOne, DeleteOne

# Import the class to be tested
from zmongo import ZMongo

# --- IMPORTANT ---
# These tests are designed to run against a LIVE MongoDB database.
# Ensure you have a MongoDB instance running.
# The tests will create and destroy collections in the database specified by
# your environment variables or the default values in zmongo.py.
# It is STRONGLY recommended to use a dedicated test database.


# This fixture creates a new ZMongo instance for each test, ensuring isolation.
# It connects to a real MongoDB instance.
@pytest_asyncio.fixture
async def zmongo_instance():
    """
    Provides a ZMongo instance connected to a real MongoDB client for testing.
    This fixture ensures that each test runs with a clean database state.
    """
    # Set dummy environment variables for consistent testing if not already set
    os.environ["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    os.environ["MONGO_DATABASE_NAME"] = os.getenv("MONGO_TEST_DATABASE_NAME", "test_db")

    zmongo = ZMongo()

    # --- Test Setup ---
    # Clean up any collections from previous runs before the test starts
    # This ensures a clean slate for every test.
    db = zmongo.db
    await db.drop_collection("test_collection")
    await db.drop_collection("another_collection")
    await zmongo.clear_cache()

    yield zmongo  # This is where the test runs

    # --- Test Teardown ---
    # Clean up after the test is done
    await db.drop_collection("test_collection")
    await db.drop_collection("another_collection")
    await zmongo.close()

    # Clear environment variables set for the test
    if "MONGO_TEST_DATABASE_NAME" in os.environ:
        del os.environ["MONGO_DATABASE_NAME"]


# Mark all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio


async def test_initialization(zmongo_instance: ZMongo):
    """Tests that the ZMongo class initializes correctly with a real client."""
    assert zmongo_instance.MONGO_DB_NAME == os.getenv("MONGO_TEST_DATABASE_NAME", "test_db")
    assert zmongo_instance.db.name == os.getenv("MONGO_TEST_DATABASE_NAME", "test_db")
    assert isinstance(zmongo_instance.mongo_client, AsyncIOMotorClient)


async def test_insert_and_find_document(zmongo_instance: ZMongo):
    """Tests inserting a single document and then finding it."""
    collection_name = "test_collection"
    doc_to_insert = {"name": "Test User", "email": "test@example.com"}

    # Insert document
    result = await zmongo_instance.insert_document(collection_name, doc_to_insert)
    assert result["status"] == "success"
    assert "inserted_id" in result

    # Find the inserted document
    found_doc = await zmongo_instance.find_document(collection_name, {"email": "test@example.com"})
    assert found_doc is not None
    assert found_doc["name"] == "Test User"


async def test_find_document_caching(zmongo_instance: ZMongo):
    """Tests that finding a document uses the cache on the second call."""
    collection_name = "test_collection"
    doc_to_insert = {"name": "Cache Test", "value": 123}
    query = {"value": 123}

    await zmongo_instance.insert_document(collection_name, doc_to_insert)

    # First find - should hit the database
    found_doc_1 = await zmongo_instance.find_document(collection_name, query)
    assert found_doc_1 is not None

    # Manually clear the DB to prove the next find is from cache
    await zmongo_instance.delete_all_documents(collection_name)

    # Second find - should hit the cache
    found_doc_2 = await zmongo_instance.find_document(collection_name, query)
    assert found_doc_2 is not None
    assert found_doc_2["name"] == "Cache Test"


async def test_update_document(zmongo_instance: ZMongo):
    """Tests updating an existing document."""
    collection_name = "test_collection"
    doc = {"name": "Original Name", "counter": 1}
    insert_result = await zmongo_instance.insert_document(collection_name, doc)

    query = {"_id": ObjectId(insert_result["inserted_id"])}
    update_data = {"$set": {"name": "Updated Name"}, "$inc": {"counter": 1}}

    await zmongo_instance.update_document(collection_name, query, update_data)

    updated_doc = await zmongo_instance.find_document(collection_name, query)
    assert updated_doc["name"] == "Updated Name"
    assert updated_doc["counter"] == 2


async def test_delete_document(zmongo_instance: ZMongo):
    """Tests deleting a single document."""
    collection_name = "test_collection"
    doc = {"name": "To Be Deleted"}
    insert_result = await zmongo_instance.insert_document(collection_name, doc)

    query = {"_id": ObjectId(insert_result["inserted_id"])}

    # Ensure it exists before deletion
    assert await zmongo_instance.find_document(collection_name, query) is not None

    delete_result = await zmongo_instance.delete_document(collection_name, query)
    assert delete_result.deleted_count == 1

    # Ensure it's gone
    assert await zmongo_instance.find_document(collection_name, query) is None


async def test_insert_documents_async(zmongo_instance: ZMongo):
    """Tests bulk inserting documents asynchronously."""
    collection_name = "test_collection"
    docs = [{"item": f"item_{i}"} for i in range(50)]

    result = await zmongo_instance.insert_documents(collection_name, docs)

    assert result["inserted_count"] == 50
    count = await zmongo_instance.count_documents(collection_name)
    assert count == 50


async def test_insert_documents_sync(zmongo_instance: ZMongo):
    """Tests bulk inserting documents synchronously."""
    collection_name = "test_collection"
    docs = [{"item": f"sync_item_{i}"} for i in range(20)]

    # The sync method is called from an async context via run_in_executor
    result = await asyncio.get_running_loop().run_in_executor(
        None, zmongo_instance.insert_documents_sync, collection_name, docs
    )

    assert result["inserted_count"] == 20
    count = await zmongo_instance.count_documents(collection_name)
    assert count == 20


async def test_get_field_names(zmongo_instance: ZMongo):
    """Tests extracting unique field names from a collection."""
    collection_name = "test_collection"
    docs = [
        {"name": "A", "value": 1},
        {"name": "B", "extra_field": True},
        {"value": 3, "another_field": "hello"}
    ]
    await zmongo_instance.insert_documents(collection_name, docs)

    fields = await zmongo_instance.get_field_names(collection_name)

    # We don't care about the order, so we use sets for comparison
    expected_fields = {"_id", "name", "value", "extra_field", "another_field"}
    assert set(fields) == expected_fields


async def test_bulk_write(zmongo_instance: ZMongo):
    """Tests the bulk_write functionality with mixed operations."""
    collection_name = "test_collection"

    # Setup initial data
    initial_docs = [
        {"_id": 1, "name": "one"},
        {"_id": 2, "name": "two"},
    ]
    await zmongo_instance.insert_documents(collection_name, initial_docs)

    operations = [
        InsertOne({"_id": 3, "name": "three"}),
        UpdateOne({"_id": 1}, {"$set": {"name": "one_updated"}}),
        DeleteOne({"_id": 2}),
    ]

    result = await zmongo_instance.bulk_write(collection_name, operations)

    assert result["inserted_count"] == 1
    assert result["modified_count"] == 1
    assert result["deleted_count"] == 1

    # Verify the state of the collection
    doc1 = await zmongo_instance.get_document_by_id(collection_name, 1)
    doc2 = await zmongo_instance.get_document_by_id(collection_name, 2)
    doc3 = await zmongo_instance.get_document_by_id(collection_name, 3)

    assert doc1["name"] == "one_updated"
    assert doc2 is None
    assert doc3["name"] == "three"


async def test_get_document_by_id(zmongo_instance: ZMongo):
    """Tests retrieving a document by its string or ObjectId."""
    collection_name = "test_collection"
    doc = {"data": "some_data"}
    insert_result = await zmongo_instance.insert_document(collection_name, doc)
    doc_id_str = insert_result["inserted_id"]
    doc_id_obj = ObjectId(doc_id_str)

    # Test with string ID
    found_by_str = await zmongo_instance.get_document_by_id(collection_name, doc_id_str)
    assert found_by_str is not None
    assert found_by_str["_id"] == doc_id_obj

    # Test with ObjectId
    found_by_obj = await zmongo_instance.get_document_by_id(collection_name, doc_id_obj)
    assert found_by_obj is not None
    assert found_by_obj["_id"] == doc_id_obj


async def test_list_and_count_collections(zmongo_instance: ZMongo):
    """Tests listing collections and counting documents."""
    # Insert into two different collections
    await zmongo_instance.insert_document("test_collection", {"a": 1})
    await zmongo_instance.insert_document("another_collection", {"b": 2})
    await zmongo_instance.insert_document("another_collection", {"c": 3})

    # Test list_collections
    collections = await zmongo_instance.list_collections()
    assert set(collections) == {"test_collection", "another_collection"}

    # Test count_documents
    count1 = await zmongo_instance.count_documents("test_collection")
    count2 = await zmongo_instance.count_documents("another_collection")
    assert count1 == 1
    assert count2 == 2
