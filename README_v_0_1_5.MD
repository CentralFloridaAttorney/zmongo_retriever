# ⚡ ZMongo Retriever

**ZMongo Retriever** is a high-performance, async-first MongoDB toolkit built for AI-powered and real-time applications.  
It wraps Motor with a modern async repository, true bulk optimizations, blazing-fast in-memory caching (not Redis), and seamless integration with OpenAI and local LLaMA models.

---

## 🚀 Features

- ✅ **100% test coverage** for all core modules (`zmongo_toolbag`)
- ⚡ **Async-enabled MongoDB access** using Motor
- 💾 **Smart in-memory auto-caching** (no Redis needed)
- 🔗 **Seamless embedding integration** with OpenAI or local LLaMA (llama-cpp-python)
- 📈 **Bulk write optimizations** — now up to **140,000+ ops/sec**!
- 🧪 **Benchmarks** comparing Mongo, Redis, and ZMongo (see below)
- 🧠 **Recursive-safe metadata flattening**
- ⚖️ **Legal text summarization + NLP preprocessing**
- 🛡️ **SafeResult**: Universal JSON serialization for all Mongo results
- 🔥 **Agent-native**: Designed for the ZAI Core Fleet (chat, embedding, database, web, file, agent maker)

---

## 📦 Installation

**From Source:**

```bash
git clone https://github.com/CentralFloridaAttorney/zmongo_retriever.git
cd zmongo_retriever
pip install .
````

**From PyPI:**

```bash
pip install --upgrade pip setuptools wheel
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple zmongo_retriever==0.1.5
```

### Requirements

* Python 3.10+
* MongoDB (local or remote)
* OpenAI API Key or GGUF LLaMA Model (for embeddings)
* `llama-cpp-python` (if using local models)
* **Redis is not required** — ZMongo uses local in-memory caching for performance.

---

## ⚙️ Example `.env`

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
from zmongo_toolbag.zmongo import ZMongo
import asyncio

async def main():
    mongo = ZMongo()
    await mongo.insert_document("users", {"name": "Alice"})
    user = await mongo.find_document("users", {"name": "Alice"})
    print(user.model_dump())

asyncio.run(main())
```

### Bulk Insert (True Async, High Speed)

```python
from zmongo_toolbag.zmongo import ZMongo
import asyncio

async def main():
    zmongo = ZMongo()
    docs = [{"x": i} for i in range(100_000)]
    result = await zmongo.insert_documents("bulk_test", docs)
    print(result.model_dump())

asyncio.run(main())
```

### Built-in Caching (not Redis)

```python
await mongo.find_document("collection", {"field": "value"})  # ✨ Cached in memory
await mongo.delete_document("collection", {"field": "value"})  # ❌ Cache invalidated
```

### Use with OpenAI GPT

```python
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
import asyncio

async def main():
    embedder = ZMongoEmbedder(collection="documents")
    doc_id = ...  # your ObjectId here
    await embedder.embed_and_store(doc_id, "Your text to embed")

asyncio.run(main())
```

### Use with LLaMA (local)

```python
from zmongo_toolbag.zmongo import ZMongo
from zai_toolbag.llama_model import LlamaModel
import asyncio
from datetime import datetime

async def main():
    zmongo = ZMongo()
    llama_model = LlamaModel()

    user_input = "Write a Dungeons & Dragons encounter using D20 rules..."
    prompt = llama_model.generate_prompt_from_template(user_input)
    output_text = llama_model.generate_text(prompt=prompt, max_tokens=3000)

    doc = {
        "type": "dnd_encounter",
        "prompt": user_input,
        "generated": output_text,
        "timestamp": datetime.now(),
        "model": "llama_model"
    }
    saved_doc = await zmongo.insert_document("dnd_encounters", doc)
    print("\n✅ Saved to MongoDB:", saved_doc.model_dump())

    await zmongo.close()

asyncio.run(main())
```

---

## 📈 Real-World Benchmark Comparison

**April 2025 Murder Mystery Suite (N=10,000):**

| Operation      | ZMongo (Async/Bulk) | ZMongo (Sync/Legacy) | PyMongo  | MongoEngine | SQLAlchemy | Mongo Shell | Redis  |
| -------------- | ------------------- | -------------------- | -------- | ----------- | ---------- | ----------- | ------ |
| Bulk Insert    | **142,844/sec**     | 4,000–8,000/sec      | 30–80k   | 20–50k      | 30–120k    | 204,995     | 17,956 |
| Per-Doc Insert | 3,079/sec           | 1,000–2,000/sec      | 2–5k     | 0.8–2k      | 1.5–6k     | 25–60k      | 5–20k  |
| Bulk Find      | **76,915/sec**      | 2,000–5,000/sec      | 10–40k   | 10–20k      | 10–70k     | 180,000     | 50,000 |
| Per-Doc Find   | 501/sec             | 400–900/sec          | 500–3k   | 800–1.5k    | 800–4k     | 5–10k       | 10–20k |
| Bulk Delete    | **40,009/sec**      | 2,000–5,000/sec      | 10–50k   | 10–30k      | 15–70k     | 120,000     | 12,000 |
| Per-Doc Delete | 3,668/sec           | 1,000–2,000/sec      | 2–5k     | 1–2.5k      | 1–4k       | 25–60k      | 5–20k  |
| Bulk Lookup    | 201/sec             | 50–150/sec           | 150–1.5k | 80–800      | 200–2k     | 5–12k       | n/a    |

> ZMongo’s async/bulk architecture is now up to **30x faster** than the prior version and outpaces most other Python-native wrappers.

---

## 🏆 Use Case Suitability

| Use Case              | ZMongo ✅ | Why                                                  |
| --------------------- | -------- | ---------------------------------------------------- |
| LLM/AI Workflows      | ✅✅✅      | Fast cached reads, embedding support, async-native   |
| Async Web Servers     | ✅✅       | Integrates with asyncio, high concurrent performance |
| LegalTech / NLP Tools | ✅✅       | Metadata/flattening, optimized for text, summaries   |
| Edge AI & Agents      | ✅✅       | In-memory perf, agent orchestration                  |
| Bulk ETL Ingestion    | 🟡       | Batch ops support, Mongo shell is still fastest      |
| Analytics Dashboards  | 🟡✅      | Great for caching reads, Redis faster for pub/sub    |

---

## 🧪 Run Tests

```bash
PYTHONPATH=.. python -m unittest discover tests
```

## 🧪 Run Benchmarks

```bash
PYTHONPATH=. python tests/test_real_db_comparative_benchmarks.py
PYTHONPATH=. python tests/test_zmongo_comparative_benchmarks.py
```

---

## 📌 Roadmap

* Add optional Redis backend (future)
* LLaMA summarization pipelines from Mongo text
* Full schema validation layer for stored documents
* Aggregation/lookup speed improvements

---

## 🧑‍💼 Author

Crafted by **John M. Iriye**

Contact: [Contact@CentralFloridaAttorney.net](mailto:Contact@CentralFloridaAttorney.net)
🌐 [View Project on GitHub](https://github.com/CentralFloridaAttorney/zmongo_retriever)

⭐️ *Star this repo if it saved you time or effort!*

---

## 📄 License

MIT License – see LICENSE for full terms.

---
