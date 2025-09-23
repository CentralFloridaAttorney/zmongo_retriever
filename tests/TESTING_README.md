How to Fix Your Retriever Tests (Cut-and-Paste Version)1. The Problem: Test Data LeakageYour retriever tests are failing because they are not isolated. Data created in one test is remaining in the database and interfering with the results of the next test. This is why tests that expect one result are getting two or more.2. The Solution: Automatic Database CleanupThe file zmongo_retriever/tests/test_helpers.py contains a pytest fixture called clean_retriever_collection. This fixture automatically connects to your test database and completely empties the retriever_test_coll collection before and after each test that uses it.This guarantees that every test starts with a clean slate.3. Applying the Fix (Copy-Paste Instructions)Follow these two steps for each of your failing test files (like test_zmongo_retriever_facts.py and test_zretriever.py).STEP 1: Add this import to the top of the test file.from .test_helpers 
```bash
import clean_retriever_collection

```


STEP 2: Add clean_retriever_collection to the signature of every failing test function.Simply find the async def test_... line for each failing test and add the fixture as a final argument. You do not need to use the argument inside the function.COPY-PASTE EXAMPLE:Here is what one of your failing tests, test_document_formatting_and_metadata, looks like before and after the fix.Before (in test_zmongo_retriever_facts.py):@pytest.mark.asynci
```bash
async def test_document_formatting_and_metadata(retriever_instance: ZRetriever, repository_instance, embedder_instance):
    """
    Tests that retrieved documents are correctly formatted into LangChain
    Documents with the right page_content and metadata.
    """
    doc_id = ObjectId()
    test_doc = {
        "_id": doc_id,
        "text": "This is the main content.",
        "author": "Test Author",
        "category": "Testing"
    }
    await populate_test_data(repository_instance, embedder_instance, [test_doc])

    results = await retriever_instance.ainvoke("A query for the main content")
    
    assert len(results) == 1
```

After (Paste this version into your file):@pytest.mark.asyncio

```bash 
async def test_document_formatting_and_metadata(retriever_instance: ZRetriever, repository_instance, embedder_instance, clean_retriever_collection):
    """
    Tests that retrieved documents are correctly formatted into LangChain
    Documents with the right page_content and metadata.
    """
    doc_id = ObjectId()
    test_doc = {
        "_id": doc_id,
        "text": "This is the main content.",
        "author": "Test Author",
        "category": "Testing"
    }
    await populate_test_data(repository_instance, embedder_instance, [test_doc])

    results = await retriever_instance.ainvoke("A query for the main content")
    
    assert len(results) == 1
```

Apply this same change to all other failing tests in your suite.