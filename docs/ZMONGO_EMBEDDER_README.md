# ZMongoRetriever System – Usage Guide

This guide demonstrates how to use the **ZMongo**, **ZMongoEmbedder**, and **ZRetriever** classes for document storage, retrieval, chunking, and embedding in your async Python applications.
**Tested code examples are included for every use case.**

---

## **Quick Start: Prerequisites**

* MongoDB running (locally or remote)
* Python packages:

  * `motor`
  * `bson`
  * `pytest`
  * `pytest-asyncio`
  * `langchain`
  * `llama-cpp-python`
* Your project structure includes `zmongo_toolbag.zmongo`, `zmongo_toolbag.zmongo_embedder`, `zmongo_toolbag.zmongo_retriever`

---

## **1. Inserting and Retrieving Documents**

```python
import asyncio
from bson.objectid import ObjectId
from zmongo_toolbag.zmongo import ZMongo

async def store_and_get():
    zm = ZMongo()
    coll = "my_test_coll"
    oid = ObjectId()
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": "Sentence one. Sentence two."}]}}
    }
    await zm.insert_document(coll, doc)
    result = await zm.find_document(coll, {"_id": oid})
    print(result.data)  # Document dict
    await zm.delete_documents(coll)

asyncio.run(store_and_get())
```

---

## **2. Chunking and Retrieving as LangChain Documents**

```python
from zmongo_toolbag.zmongo_retriever import ZRetriever
from bson.objectid import ObjectId

async def chunk_and_get():
    zm = ZMongo()
    coll = "test_retriever_123"
    retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False)
    oid = ObjectId()
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": "Sentence one. Sentence two. Sentence three."}]}}
    }
    await zm.insert_document(coll, doc)
    docs = await retriever.get_zdocuments([oid])
    for d in docs:
        print(d.page_content)  # Each chunk of text
    await zm.delete_documents(coll)

# asyncio.run(chunk_and_get())
```

---

## **3. Invoking Retriever (No Embeddings, Returns Documents/Chunks)**

```python
result = await retriever.invoke([oid])
assert isinstance(result, list)
assert all(isinstance(chunk, Document) for chunk in result)
```

---

## **4. Invoking Retriever (With Embeddings)**

```python
retriever = ZRetriever(collection=coll, repository=zm, use_embedding=True)
result = await retriever.invoke([oid])
assert isinstance(result, list)
for vector in result:
    assert isinstance(vector, list)
    assert all(isinstance(x, float) for x in vector)
```

---

## **5. Invoking Retriever With Multiple OIDs**

```python
oids = [ObjectId() for _ in range(2)]
for oid in oids:
    doc = {
        "_id": oid,
        "database_name": "testdb",
        "collection_name": coll,
        "casebody": {"data": {"opinions": [{"text": text}]}}
    }
    await zm.insert_document(coll, doc)

result = await retriever.invoke(oids)
assert isinstance(result, list)
for chunk_set in result:
    assert isinstance(chunk_set, list)
    for vector in chunk_set:
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
```

---

## **6. Retriever with max\_tokens\_per\_set=0**

* **Without embedding**: Returns `Document` objects.
* **With embedding**: Returns a list of embedding vectors (lists of floats).

```python
retriever = ZRetriever(collection=coll, repository=zm, use_embedding=False, max_tokens_per_set=0)
result = await retriever.invoke([oid])
assert all(isinstance(d, Document) for d in result)

retriever = ZRetriever(collection=coll, repository=zm, use_embedding=True, max_tokens_per_set=0)
result = await retriever.invoke([oid])
assert all(isinstance(vector, list) and all(isinstance(x, float) for x in vector) for vector in result)
```

---

## **7. Chunk Set Overlap and Token-based Splitting**

```python
from types import SimpleNamespace
retriever = ZRetriever(collection="fake", max_tokens_per_set=5, overlap_prior_chunks=2)
retriever.num_tokens_from_string = lambda t: 2  # Simulate each doc as 2 tokens
docs = [SimpleNamespace(page_content=f"text{i}") for i in range(5)]
sized = retriever.get_chunk_sets(docs)
assert len(sized) > 1
for i in range(1, len(sized)):
    overlap = retriever.overlap_prior_chunks
    prev = sized[i-1]
    curr = sized[i]
    assert prev[-overlap:] == curr[:overlap]
```

---

## **8. Handling Invalid Input or Edge Cases**

* **Invalid page content (not a string):**
  Skipped with a warning, not returned in output.

* **Invalid ObjectId:**
  Returns an empty result or skips as appropriate.

* **Empty result for not-found OID:**

  ```python
  result = await retriever.invoke([ObjectId()])
  assert result == [] or all(r == [] for r in result)
  ```

---

## **Summary Table**

| Use Case                               | Result Type                                          | Notes                                       |
| -------------------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| Insert & Retrieve                      | dict                                                 | Use `ZMongo` methods                        |
| Chunking only                          | List\[Document]                                      | Use `ZRetriever` with `use_embedding=False` |
| Embedding                              | List\[List\[float]]                                  | Use `use_embedding=True`                    |
| Multi-OID                              | List\[List\[List\[float]]] or List\[List\[Document]] | One set per OID                             |
| `max_tokens_per_set=0`, use\_embedding | List\[List\[float]]                                  | Embeddings for all chunks                   |
| `max_tokens_per_set=0`, no embedding   | List\[Document]                                      | Raw chunk Documents                         |
| Chunk overlap                          | List\[List\[Document]]                               | Overlap guaranteed                          |
| Not found / invalid input              | \[]                                                  | Empty result, never raises                  |

---

## **Troubleshooting**

* Always await async functions!
* Use correct OIDs when storing/retrieving.
* When in doubt, print `type(result)` and drill down to check your data shape.

---

## **See also**

* [LangChain Document docs](https://python.langchain.com/docs/api/langchain_core/langchain_core.documents.Document)
* [llama-cpp-python for local embeddings](https://github.com/abetlen/llama-cpp-python)
* [PyMongo/Motor docs](https://motor.readthedocs.io/en/stable/)


