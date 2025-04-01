Here’s a clean and concise **overview summary** suitable for the main documentation page (`README.md` or equivalent) for the `ZMongo` class. Detailed functionality can be broken out into separate `.md` files for each feature as needed:

---

## `ZMongo` Overview

`ZMongo` is an asynchronous MongoDB repository class built on `motor` that simplifies and optimizes interactions with a MongoDB database. It provides a high-level interface for common database operations, offering out-of-the-box support for caching, bulk operations, and document serialization.

### 🔧 Features

- **Async MongoDB Access**: Powered by `motor`, supports high-concurrency scenarios with connection pooling.
- **Smart Caching**: Automatic in-memory caching of single-document reads, using hash-based cache keys.
- **CRUD Operations**:
  - `find_document`: Cached retrieval of a single document.
  - `find_documents`: Retrieve multiple documents with optional query options.
  - `insert_document`: Insert and auto-cache new documents.
  - `update_document`: Update with optional `arrayFilters`, and automatic cache sync.
  - `delete_document`: Deletes a document and purges it from the cache.
- **Bulk Write Support**: Efficient batch operations with `InsertOne`, `UpdateOne`, `DeleteOne`, `ReplaceOne`.
- **Embedding Management**: Save vector embeddings (e.g. for ML/AI use cases) into specified fields of documents.
- **Simulation Utilities**: Helper to retrieve ordered simulation steps by `simulation_id`.
- **Cache Control**: Easily clear internal cache with `clear_cache`.
- **Safe Shutdown**: Cleanly closes the database client connection on app teardown.

### 📦 Environment Configuration

Uses `.env` variables:
- `MONGO_URI` – Connection string for MongoDB.
- `MONGO_DATABASE_NAME` – Target database.
- `DEFAULT_QUERY_LIMIT` – Default document limit for queries (fallback: `100`).

### 🧱 Built With

- `motor` for async MongoDB operations
- `bson` for object/document serialization
- `dotenv` for config management
- `pymongo` for bulk operation models

---

Let me know if you'd like `.md` files generated for each of the core method groups (CRUD, bulk, caching, simulation, etc.) — I can scaffold those too.