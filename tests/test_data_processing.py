import os
import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from bson import ObjectId
from dotenv import load_dotenv

from zmongo_toolbag.data_processing import SafeResult
from zmongo_toolbag.zmongo import ZMongo

load_dotenv(Path.home() / "resources" / ".env_local")
TEST_DB_NAME = "data_processing_test_db"
COLLECTION_NAME = "safetest_collection"
MONGO_URI = os.getenv("MONGO_URI")

pytestmark = pytest.mark.skipif(
    not MONGO_URI,
    reason="MONGO_URI must be set in the environment for live integration tests."
)


# --- Test Fixture ---

@pytest_asyncio.fixture
async def prepared_zmongo_instance():
    """
    Provides a ZMongo instance and pre-populates it with a complex document
    for testing SafeResult's discovery methods.
    """
    repo = ZMongo()


    # Clean up before the test
    # await client.drop_database(TEST_DB_NAME)

    # Define a complex, nested document to test against
    doc_id = ObjectId()
    nested_document = {
        "_id": doc_id,
        "author": "John Doe",
        "casebody": {
            "data": {
                "judges": ["Smith", "Jones"],
                "opinions": [
                    {
                        "author": "Judge Smith",
                        "text": "The quick brown fox jumps over the lazy dog."
                    }
                ]
            }
        },
        "citations": [
            {"type": "case", "cite": "123 U.S. 456"},
            {"type": "statute", "cite": "42 U.S.C. 1983"}
        ]
    }

    # Insert the document into the database
    await repo.insert_document(COLLECTION_NAME, nested_document)

    yield repo, doc_id  # Provide the repo and the ID to the test

    # Teardown: drop the database
    # await client.drop_database(TEST_DB_NAME)
    repo.close()


# --- Test Cases for SafeResult ---

@pytest.mark.asyncio
async def test_safetesult_get_method(prepared_zmongo_instance):
    """
    Tests the .get() method for retrieving nested data with dot notation.
    """
    repo, doc_id = prepared_zmongo_instance

    # Fetch the document using ZMongo to get a SafeResult
    find_result = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})
    assert find_result.success

    # Test retrieving various nested fields
    opinion_text = find_result.get("casebody.data.opinions.0.text")
    assert opinion_text == "The quick brown fox jumps over the lazy dog."

    second_judge = find_result.get("casebody.data.judges.1")
    assert second_judge == "Jones"

    first_citation_cite = find_result.get("citations.0.cite")
    assert first_citation_cite == "123 U.S. 456"

    # Test retrieving a non-existent key
    non_existent = find_result.get("casebody.data.non_existent_key", default="not_found")
    assert non_existent == "not_found"


@pytest.mark.asyncio
async def test_safetesult_to_json_method(prepared_zmongo_instance):
    """
    Tests the .to_json() method for serializing the result data.
    """
    repo, doc_id = prepared_zmongo_instance
    find_result = await repo.find_document(COLLECTION_NAME, {"_id": doc_id})

    # Generate JSON string
    json_output = find_result.to_json()

    # Verify it's a valid JSON string and contains expected data
    assert isinstance(json_output, str)
    data_from_json = json.loads(json_output)

    assert data_from_json["author"] == "John Doe"
    assert data_from_json["_id"] == str(doc_id)
    assert data_from_json["casebody"]["data"]["opinions"][0]["text"] == "The quick brown fox jumps over the lazy dog."


@pytest.mark.asyncio
async def test_safetesult_to_metadata_with_keymap(prepared_zmongo_instance):
    """
    Tests the .to_metadata() method with a keymap to flatten and rename keys.
    """
    repo, doc_id = prepared_zmongo_instance

    # Define a keymap to translate raw keys to user-friendly "tags"
    keymap = {
        "casebody.data.opinions.0.text": "opinion_text",
        "author": "document_author",
        "citations.1.cite": "second_citation"
    }

    # Fetch the document, this time passing the keymap
    find_result_with_map = await repo.find_document(
        COLLECTION_NAME,
        {"_id": doc_id},
        # Note: ZMongo.find_document needs to be updated to accept and pass this to SafeResult
        # For this test, we will manually create the SafeResult with the map
    )

    # Manually create a SafeResult with the keymap for the purpose of this test
    mapped_result = SafeResult.ok(find_result_with_map.data, metadata_keymap=keymap)

    # Generate metadata
    metadata = mapped_result.to_metadata()

    # Assert that keys are flattened and renamed correctly
    assert metadata["opinion_text"] == "The quick brown fox jumps over the lazy dog."
    assert metadata["document_author"] == "John Doe"
    assert metadata["second_citation"] == "42 U.S.C. 1983"

    # Assert that unmapped keys are still present with their original names
    assert metadata["casebody.data.judges.0"] == "Smith"

