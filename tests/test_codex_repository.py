import pytest
import os
import shutil
import time
from pathlib import Path
from bson import ObjectId

from zmongo_toolbag.codex_repository import CodexRepository

# --- Integration Test Setup ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    pytest.skip("MONGO_URI environment variable not set, skipping integration tests.", allow_module_level=True)

pytestmark = pytest.mark.integration

TEST_DB_NAME = "codex_repo_integration_test_final"
TEST_COLLECTION = "codex_items"
# NEW: Define a specific collection for codex tests to ensure isolation.
TEST_CODEX_COLLECTION = "test_legal_codex_collection"
TEST_BACKUP_ROOT = Path("./temp_backup_dir_for_tests")


@pytest.fixture(scope="module")
def repository():
    """Provides a CodexRepository instance connected once per test module."""
    os.environ["MONGO_DATABASE_NAME"] = TEST_DB_NAME
    # MODIFIED: Initialize the repository to use our dedicated test collection for codex documents.
    repo = CodexRepository(backup_root=TEST_BACKUP_ROOT, codex_collection=TEST_CODEX_COLLECTION)
    yield repo
    # Teardown: Close the connection and clean up the environment variable.
    repo.db.client.drop_database(TEST_DB_NAME)
    repo.close()
    if "MONGO_DATABASE_NAME" in os.environ:
        del os.environ["MONGO_DATABASE_NAME"]


@pytest.fixture(autouse=True)
def clean_environment(repository):
    """Ensures a clean state before each test by wiping relevant collections."""
    # MODIFIED: Ensure we clean the main test collection AND the specific codex test collection.
    repository.db.delete_all_documents(TEST_COLLECTION)
    repository.db.delete_all_documents(TEST_CODEX_COLLECTION) # This was the missing step.

    # Clean up backup directory
    if TEST_BACKUP_ROOT.exists():
        shutil.rmtree(TEST_BACKUP_ROOT)
    TEST_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    # Final cleanup after test if needed, though module-level teardown handles most of it.
    if TEST_BACKUP_ROOT.exists():
        shutil.rmtree(TEST_BACKUP_ROOT)


def test_repository_initialization(repository):
    assert repository.db is not None
    assert repository.backup_root.exists()
    assert repository.codex_collection == TEST_CODEX_COLLECTION


def test_insert_and_find_one(repository):
    doc = {"_id": ObjectId(), "name": "test_doc"}
    repository.insert(TEST_COLLECTION, doc)
    find_result = repository.find_one(TEST_COLLECTION, {"_id": doc["_id"]})
    assert find_result.success and find_result.data["name"] == "test_doc"


def test_update_and_find_all(repository):
    doc1 = {"_id": ObjectId(), "tag": "A"}
    doc2 = {"_id": ObjectId(), "tag": "B"}
    repository.insert(TEST_COLLECTION, doc1)
    repository.insert(TEST_COLLECTION, doc2)
    repository.update(TEST_COLLECTION, {"_id": doc1["_id"]}, {"$set": {"tag": "C"}})
    find_all_result = repository.find_all(TEST_COLLECTION)
    assert len(find_all_result.data) == 2
    updated_doc = next(d for d in find_all_result.data if d["_id"] == str(doc1["_id"]))
    assert updated_doc["tag"] == "C"


def test_delete(repository):
    doc = {"_id": ObjectId(), "status": "to_delete"}
    repository.insert(TEST_COLLECTION, doc)
    repository.delete(TEST_COLLECTION, {"_id": doc["_id"]})
    find_result = repository.find_one(TEST_COLLECTION, {"_id": doc["_id"]})
    assert find_result.success and find_result.data is None


def test_get_distinct_field_values(repository):
    repository.insert(TEST_COLLECTION, {"category": "X"})
    repository.insert(TEST_COLLECTION, {"category": "Y"})
    repository.insert(TEST_COLLECTION, {"category": "X"})
    distinct_result = repository.get_distinct_field_values(TEST_COLLECTION, "category")
    assert sorted(distinct_result.data) == ["X", "Y"]


def test_save_codex_and_backup(repository):
    """
    Test save_codex for both insert and update, and verify backup files.
    """
    codex_id = ObjectId()
    codex_doc = {"_id": codex_id, "meta_title": "My First Codex", "version": 1}

    # --- Test Insert ---
    save_result_insert = repository.save_codex(codex_doc)
    assert save_result_insert.success
    assert save_result_insert.data["upserted_id"] == str(codex_id)
    backup_files_after_insert = list(TEST_BACKUP_ROOT.glob(f"{codex_id}_*.json"))
    assert len(backup_files_after_insert) == 1

    time.sleep(1.1) # Ensure timestamp for the next backup is unique

    # --- Test Update ---
    codex_doc["version"] = 2
    save_result_update = repository.save_codex(codex_doc)
    assert save_result_update.success
    assert save_result_update.data["modified_count"] == 1
    backup_files_after_update = list(TEST_BACKUP_ROOT.glob(f"{codex_id}_*.json"))
    assert len(backup_files_after_update) == 2


def test_get_all_codex_summaries_and_load(repository):
    # This test will now pass because clean_environment wipes TEST_CODEX_COLLECTION
    codex1 = {"_id": ObjectId(), "meta_title": "Summary A", "content": "Full content A"}
    codex2 = {"_id": ObjectId(), "meta_title": "Summary B", "content": "Full content B"}
    repository.save_codex(codex1)
    repository.save_codex(codex2)

    summaries_result = repository.get_all_codex_summaries()
    assert summaries_result.success and len(summaries_result.data) == 2

    # The rest of the test remains the same
    first_summary_id = summaries_result.data[0]["_id"]
    load_result = repository.load_codex(first_summary_id)
    assert load_result.success and load_result.data is not None
    assert load_result.data["content"].startswith("Full content")
