"""
This module provides shared pytest fixtures to ensure test isolation,
primarily by managing the state of the MongoDB test database.
"""

import pytest_asyncio
from zmongo_toolbag.zmongo import ZMongo

# This is the shared collection name used by the retriever tests.
COLLECTION_NAME = "test"


@pytest_asyncio.fixture(scope="function")
async def clean_retriever_collection():
    """
    Pytest fixture that cleans the 'retriever_test_coll' before and after
    each test function.

    By adding this fixture as an argument to a test function, you guarantee
    that the test starts with an empty collection, preventing data from one
    test from interfering with another.

    Usage:
        async def test_my_retriever_logic(retriever_instance, clean_retriever_collection):
            # The collection is now guaranteed to be empty.
            ...
    """
    repo = None
    try:
        repo = ZMongo()
        # Setup: Drop the collection to ensure a clean start
        await repo.db.drop_collection(COLLECTION_NAME)

        # Yield control to the test function
        yield COLLECTION_NAME

    finally:
        # Teardown: Drop the collection again to be tidy after the test
        if repo:
            await repo.db.drop_collection(COLLECTION_NAME)
            repo.close()
