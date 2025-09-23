# test_zonehotdb_lossless.py
# Python 3.10+
#
# =============================================================================
# Integration Test Suite for ZOneHotDB (Lossless Text Version)
# =============================================================================
"""
This script contains an integration test suite for the lossless text version
of the ZOneHotDB library. It connects to a real MongoDB instance to verify
the end-to-end functionality of encoding, storing, retrieving, and perfectly
reconstructing complex text.

Prerequisites:
- A running MongoDB instance.
- A .env file configured with MONGO_URI and MONGO_DATABASE_NAME.

Dependencies:
- pytest
- pytest-asyncio

Install with:
  pip install pytest pytest-asyncio python-dotenv

Run tests with:
  pytest test_zonehotdb_lossless.py
"""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# --- Ensure correct imports for the library ---
try:
    from onehotdb.zonehotdb import ZOneHotDB, VocabConfig
    from zmongo_toolbag.zmongo import ZMongo
except (ImportError, ModuleNotFoundError):
    # Adjust path for local testing if needed
    import sys

    # This assumes your script is in a 'tests' folder, and the library is one level up
    sys.path.append(str(Path(__file__).parent.parent))
    from onehotdb.zonehotdb import ZOneHotDB, VocabConfig
    from zmongo_toolbag.zmongo import ZMongo

# --- Test Configuration & Data ---

# Load environment variables for the database connection
load_dotenv()

# This is the ground truth for our reconstruction test.
# It includes various capitalization schemes, punctuation, and whitespace.
COMPLEX_TEXT_CONTENT = """
Welcome to ZOneHotDB, a test of mixedCase reconstruction!
This line is ALL CAPS.
    And this one is indented.
"""


# --- Pytest Fixture for Database Setup/Teardown ---

@pytest.fixture(scope="function")  # FIX: Changed scope from "module" to "function"
async def db():
    """
    Pytest fixture to set up a clean database connection for each test function
    and tear it down afterward.
    """
    # Ensure environment variables are set
    if not os.getenv("MONGO_URI") or not os.getenv("MONGO_DATABASE_NAME"):
        pytest.fail("MONGO_URI and MONGO_DATABASE_NAME must be set in your .env file.")

    # Configure for perfect reconstruction
    config = VocabConfig(remove_stopwords=False, encode_capitalization=True)

    db_instance = ZOneHotDB(
        documents_collection_name="test_documents",
        vocab_collection_name="test_vocabulary",
        config=config
    )

    print("\n--- Setting up test database ---")
    # Clean up any data from previous test runs
    await db_instance.repo.delete_documents(db_instance.documents_collection, {})
    await db_instance.repo.delete_documents(db_instance.lexicon.collection_name, {})

    yield db_instance

    print("\n--- Tearing down test database ---")
    db_instance.repo.close()


# --- Test Case ---

@pytest.mark.asyncio
async def test_full_reconstruction_cycle(db: ZOneHotDB):
    """
    The primary integration test. It performs the full encode-store-retrieve-
    reconstruct-verify cycle using a complex text string.
    """
    doc_id = "complex_text_doc"

    # 1. Encode and Store the document
    await db.encode_and_store_text(doc_id, COMPLEX_TEXT_CONTENT)

    # 2. Retrieve the encoded data from the database
    indices = await db.get_encoded_indices(doc_id)
    masks = await db.get_capitalization_mask(doc_id)

    assert len(indices) > 0, "No indices were stored for the document."
    assert len(masks) > 0, "No capitalization masks were stored for the document."
    assert len(indices) == len(masks), "Mismatch between number of indices and masks."

    # 3. Reconstruct the document
    word_tasks = [db.lexicon.get_word_from_index(idx) for idx in indices]
    words = await asyncio.gather(*word_tasks)

    reconstructed_tokens = []
    for i, word in enumerate(words):
        assert word is not None, f"Failed to find word for index {indices[i]}"

        mask_value = masks[i]
        reconstructed_token = word

        if mask_value == 'T':
            reconstructed_token = word.capitalize()
        elif mask_value == 'U':
            reconstructed_token = word.upper()
        elif mask_value.startswith('I,'):
            try:
                upper_indices = {int(idx) for idx in mask_value[2:].split(',')}
                char_list = list(word)
                for idx in upper_indices:
                    if idx < len(char_list):
                        char_list[idx] = char_list[idx].upper()
                reconstructed_token = "".join(char_list)
            except (ValueError, IndexError):
                pass  # Fallback to lowercase word on error

        reconstructed_tokens.append(reconstructed_token)

    reconstructed_text = "".join(reconstructed_tokens)

    # 4. Verify the reconstruction is a perfect character-for-character match
    assert reconstructed_text == COMPLEX_TEXT_CONTENT, "Reconstructed text does not match the original."

