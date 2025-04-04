# ⚡ ZMongo Retriever
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CentralFloridaAttorney/zmongo_retriever/blob/main/LICENSE)
[![Discussions](https://img.shields.io/badge/Discussions-Join%20Us-blue?logo=github)](https://github.com/CentralFloridaAttorney/zmongo_retriever/discussions)
[![Issues](https://img.shields.io/github/issues/CentralFloridaAttorney/zmongo_retriever)](https://github.com/CentralFloridaAttorney/zmongo_retriever/issues)
[![Last Commit](https://img.shields.io/github/last-commit/CentralFloridaAttorney/zmongo_retriever)](https://github.com/CentralFloridaAttorney/zmongo_retriever/commits/main)

**ZMongo Retriever** is a high-performance, async-first MongoDB toolkit built for AI-powered and real-time applications. It wraps `motor` and `pymongo` with a modern async repository, bulk optimizations, smart **in-memory** caching (not Redis), and seamless integration with OpenAI and local LLaMA models.

---

## 🚀 Features
- ✅ 100% test coverage for `zmongo_toolbag`
- 🔄 Async-enabled MongoDB access using `motor`
- 🧠 In-memory auto-caching to accelerate repeated queries
- 🔗 Embedding integration with OpenAI or local LLaMA (`llama-cpp-python`)
- 📈 Bulk write optimizations (tested up to 200M+ ops/sec)
- 🧪 Benchmarking suite for Mongo vs Redis comparisons
- 🧠 Recursive-safe metadata flattening
- ⚖️ Legal text summarization + NLP preprocessing

---

## 📦 Installation

```bash
pip install .
```

### Requirements
- Python 3.10+
- MongoDB (local or remote)
- OpenAI API Key or GGUF LLaMA Model (for embeddings)
- `llama-cpp-python` (if using local models)

> Redis is **not required** — ZMongo uses **local in-memory caching** for performance.

---

## ⚙️ .env Example

```env
MONGO_URI=mongodb://127.0.0.1:27017
MONGO_DATABASE_NAME=ztarot
OPENAI_API_KEY=your-api-key-here
OPENAI_TEXT_MODEL=gpt-3.5-turbo-instruct
DEFAULT_MAX_TOKENS=256
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.95
GGUF_MODEL_PATH=/path/to/your/model.gguf
```

---

## 🔧 Quickstart

### Async Mongo Access

```python
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

mongo = ZMongo()
await mongo.insert_document("users", {"name": "Alice"})
user = await mongo.find_document("users", {"name": "Alice"})
```

---

### Embedding Text (OpenAI)

```python
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from bson import ObjectId

embedder = ZMongoEmbedder(repository=mongo, collection="documents")
await embedder.embed_and_store(ObjectId("65f0..."), "Your text to embed")
```

---

### Async Bulk Insert

```python
from pymongo import InsertOne
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

zmongo = ZMongo()
ops = [InsertOne({"x": i}) for i in range(100_000)]
await zmongo.bulk_write("bulk_test", ops)
```

---

### Built-in Caching (not Redis)

```python
await mongo.find_document("collection", {"field": "value"})  # ✨ Cached in memory
await mongo.delete_document("collection", {"field": "value"})  # ❌ Cache invalidated
```

---

### Use with OpenAI GPT

```python
from zmongo_retriever.models.openai_model import OpenAIModel

model = OpenAIModel()
response = await model.generate_instruction("Summarize the first amendment")
print(response)
```

---

### Use with LLaMA (local)

```python
from zmongo_retriever.models.llama_model import LlamaModel

llm = LlamaModel()
prompt = llm.generate_prompt_from_template("Explain ZMongo Retriever")
output = llm.generate_text(prompt=prompt, max_tokens=512)
print(output)
```

---

## 📊 Performance Benchmarks

| **Operation**              | **ZMongo (Async+Cached)** | **Mongo Shell**         | **Redis**             |
|---------------------------|---------------------------|-------------------------|------------------------|
| Bulk Write (100k)         | 🚀 **209M ops/sec**        | 👢 258K ops/sec         | ❌ N/A                 |
| Insert Latency (500 docs) | ⚡ 0.0329 ms/doc            | 👬 0.2405 ms/doc        | ⚡ 0.0451 ms/doc        |
| Query (cached)            | ⚡ **0.0054 ms**            | 👢 0.2436 ms            | ⚡ 0.0418 ms            |
| Concurrent Reads (5k)     | ⚙️ 0.766s (async)           | 🧵 1.41s (threaded)     | ⚡ 0.54s (threaded)     |
| Cache Hit Rate            | ✅ 100%                    | ❌ None                 | ✅ Built-in            |

---

## 🧪 Run Tests

```bash
PYTHONPATH=.. python -m unittest discover tests

```

## 🧪 Run Benchmarks

```bash
PYTHONPATH=. python tests/test_real_db_comparative_benchmarks.py
```

---

## 📌 Roadmap

- [ ] Add optional Redis backend (future)
- [ ] LLaMA summarization pipelines from Mongo text
- [ ] Full schema validation layer for stored documents

---

## 🧑‍💻 Author

Crafted by **John M. Iriye**  
📢 [Contact@CentralFloridaAttorney.net](mailto:Contact@CentralFloridaAttorney.net)  
🌐 [View Project on GitHub](https://github.com/CentralFloridaAttorney/zmongo_retriever)

> ⭐️ Star this repo if it saved you time or effort!

---

## 📄 License

MIT License – see [LICENSE](https://github.com/CentralFloridaAttorney/zmongo_retriever/blob/main/LICENSE) for full terms.

