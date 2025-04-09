# ZMongo Overview

`ZMongo` is an asynchronous MongoDB repository built on `motor` that simplifies data access for high-performance AI, retrieval, and simulation workflows. It provides intuitive methods for CRUD operations, bulk writes, caching, and vector embedding management, all while supporting async concurrency.

---

## 🔧 Key Features

- **Async MongoDB Access**
  - Powered by `motor`, designed for high-throughput workloads with connection pooling.

- **Smart Caching**
  - Automatically caches document reads using SHA-256 hash keys.

- **Core CRUD Methods**
  - `find_document`: Retrieve and cache a single document.
  - `find_documents`: Fetch multiple documents with flexible query options.
  - `insert_document`: Insert and cache new documents.
  - `update_document`: Update documents with optional `arrayFilters` and cache sync.
  - `delete_document`: Remove from DB and cache.
  - `delete_all_documents`: Efficiently wipe all documents in a collection.

- **Simulation Support**
  - `get_simulation_steps`: Retrieve and order simulation steps by `step` field.

- **Vector Embedding Storage**
  - `save_embedding`: Persist and cache ML-generated embeddings in document fields.

- **Bulk Operations**
  - `bulk_write`: Supports `InsertOne`, `UpdateOne`, `DeleteOne`, and `ReplaceOne` operations.

- **Cache & Teardown**
  - `clear_cache`: Reset internal cache.
  - `close`: Clean MongoDB client shutdown.

---

## ⚙️ Environment Setup

Create a `.env` file with the following:

```bash
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE_NAME=your_db_name
DEFAULT_QUERY_LIMIT=100
```

---

## 🧪 Example Usage

```python
from zmongo_toolbag.zmongo import ZMongo

zmongo = ZMongo()

async def demo():
    await zmongo.insert_document("users", {"name": "Alice"})
    user = await zmongo.find_document("users", {"name": "Alice"})
    print(user)
```

---

## 🧱 Tech Stack

- [`motor`](https://motor.readthedocs.io/) – Async MongoDB driver
- `bson` – Object serialization
- `pymongo` – Bulk operation models
- `dotenv` – Environment variable management

---

For more detail on using `page_content_key` for field-level extraction, see `PAGE_CONTENT_KEYS.md`.
