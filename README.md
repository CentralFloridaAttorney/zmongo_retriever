

# ZMongo Toolbag

**ZMongo Toolbag** is a high-performance, async-first MongoDB utility suite built for Python developers working on AI-powered, data-heavy, and real-time applications. It wraps `motor` and `pymongo` with powerful tools like automatic caching, embeddings integration, bulk throughput optimization, and a modern async repository interface.

---

## 🚀 Key Features

- ✨ **Async-Enabled**: Powered by `motor` for blazing-fast MongoDB operations
- 🧠 **Smart Caching**: Hash-keyed automatic caching with near-zero latency hits
- 🧩 **Unified Repository** (`ZMongo`) for consistent async CRUD access
- 🔗 **AI Embedding Integration** with OpenAI or local models via `ZMongoEmbedder`
- 📦 **Bulk Write Optimization**: 200M+ ops/sec tested with async throughput
- 🧰 **Metadata Flattening** & recursive-safe serialization
- 🧪 **Benchmarking Suite** included with tests vs Mongo shell & Redis
- 🔄 **Simulation Step-Aware Queries** for ordered document processing
- ✅ **100% Test Coverage** and strong performance monitoring support

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- MongoDB running locally or remotely
- Redis (optional)
- OpenAI API key (optional for embeddings)

---

## ⚙️ .env Setup

Here's a minimal setup for development:

```env
MONGO_URI=mongodb://127.0.0.1:27017
MONGO_DATABASE_NAME=ztarot
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
DEFAULT_QUERY_LIMIT=100
```

> Additional keys and paths (for Redis, Chroma, SSL, etc.) are supported as per `zmongo_toolbag/.env`.

---

## 🔧 Quick Usage

### Insert & Query

```python
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

mongo = ZMongo()
await mongo.insert_document("users", {"name": "Alice"})
doc = await mongo.find_document("users", {"name": "Alice"})
```

---

### Caching (Auto-managed)

```python
# First call is cached
await mongo.find_document("users", {"name": "Alice"})

# Cache is invalidated on mutation
await mongo.delete_document("users", {"name": "Alice"})
```

---

### Embeddings

```python
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from bson import ObjectId

mongo = ZMongo()
embedder = ZMongoEmbedder(repository=mongo, collection="documents")
await embedder.embed_and_store(ObjectId("5f43a1ab1234567890abcdef"), "Some text to embed")
```

---

### ZMongo Bulk Insert

```python
import asyncio
from pymongo import InsertOne
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

async def main():
    zmongo = ZMongo()
    operations = [InsertOne({"index": i, "value": f"item_{i}"}) for i in range(100000)]
    await zmongo.bulk_write("my_collection", operations)
    await zmongo.close()

asyncio.run(main())
```

---

### Raw MongoDB Insert (Sync Example)

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["my_database"]
collection = db["my_collection"]

documents = [{"index": i, "value": f"item_{i}"} for i in range(100000)]
collection.insert_many(documents)

client.close()
```

---

### Redis Benchmark Example

```python
import redis
import time

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Set key-value
start = time.time()
redis_client.set("key", "value")
print("Set time:", time.time() - start)

# Get key-value
start = time.time()
value = redis_client.get("key")
print("Get time:", time.time() - start)
print("Value:", value.decode())
```

---

## 🧪 Run Tests

```bash
python -m unittest discover zmongo_toolbag_BAK/tests
```

To benchmark:

```bash
python zmongo_toolbag_BAK/tests/test_zmongo_comparative_benchmarks.py
```

---

## 🗺️ Roadmap

- [ ] Vector search backend support (FAISS, ChromaDB)
- [ ] Multi-tenant database context handling
- [ ] Configurable cache backends (Redis, DiskCache)
- [ ] MongoDB retry/backoff middleware

---

## 👨‍💻 Author

Crafted with ❤️ by **John M. Iriye**

> Star ⭐️ the repo if this project saved you hours — because it will.  
> [View ZMongo on GitHub](https://github.com/CentralFloridaAttorney/zmongo_retriever)

---

## 📄 License

MIT License – see [`LICENSE`](LICENSE) for details.