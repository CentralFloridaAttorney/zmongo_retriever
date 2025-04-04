


# ⚡ ZMongo Retriever

**ZMongo Retriever** is a high-performance, async-first MongoDB toolkit built for AI-powered and real-time applications. It wraps `motor` and `pymongo` with a modern async repository, bulk optimizations, smart caching, and seamless integration with OpenAI and local LLaMA models.

---

## 🚀 Features

- 🔄 **Async-Enabled MongoDB Access** with `motor`
- 🧠 **Auto-Caching** for repeated query acceleration
- 🔌 **Embeddings Integration** with OpenAI or LLaMA (via `llama-cpp-python`)
- 📈 **Bulk Write Optimizations** up to 200M+ ops/sec tested
- 🧪 **Benchmarking Suite** with Redis and PyMongo comparisons
- 🧰 **Recursive-safe Metadata Flattening**
- 🛠️ **Full Test Coverage** + Jupyter/Script compatibility
- 🧑‍⚖️ **Legal Research-Oriented** summarization pipelines

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- MongoDB (local or remote)
- Redis (optional for caching)
- OpenAI API Key or GGUF LLaMA Model
- `llama-cpp-python` (if using local models)

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

embedder = ZMongoEmbedder(repository=mongo, collection="documents")
await embedder.embed_and_store(ObjectId("..."), "Your text to embed")
```

---

### Async Bulk Insert

```python
from pymongo import InsertOne
zmongo = ZMongo()
ops = [InsertOne({"x": i}) for i in range(100_000)]
await zmongo.bulk_write("bulk_test", ops)
```

---

### Redis Caching

```python
await mongo.find_document("collection", {"field": "value"})  # Cached
await mongo.delete_document("collection", {"field": "value"})  # Invalidates cache
```

---

### Use with OpenAI GPT

```python
from your_module.openai_model import OpenAIModel

model = OpenAIModel()
response = await model.generate_instruction("Summarize the first amendment")
print(response)
```

---

### Use with LLaMA (local)

```python
from llama_model import LlamaModel

llm = LlamaModel()
prompt = llm.generate_prompt_from_template("Explain ZMongo Retriever")
output = llm.generate_text(prompt=prompt, max_tokens=512)
print(output)
```

---

## 📊 Performance Benchmarks

| **Operation**              | **ZMongo (Async+Cached)** | **Mongo Shell**         | **Redis**             |
|---------------------------|---------------------------|-------------------------|------------------------|
| Bulk Write (100k)         | 🚀 **209M ops/sec**        | 🐢 258K ops/sec         | ❌ N/A                 |
| Insert Latency (500 docs) | ⚡ 0.0329 ms/doc            | 🐌 0.2405 ms/doc        | ⚡ 0.0451 ms/doc        |
| Query (cached)            | ⚡ **0.0054 ms**            | 🐢 0.2436 ms            | ⚡ 0.0418 ms            |
| Concurrent Reads (5k)     | ⚙️ 0.766s (async)           | 🧵 1.41s (threaded)     | ⚡ 0.54s (threaded)     |
| Cache Hit Rate            | ✅ 100%                    | ❌ None                 | ✅ Built-in            |

---

## 🧪 Run Tests

```bash
python -m unittest discover tests
```

## 🧪 Run Benchmarks

```bash
python tests/test_zmongo_comparative_benchmarks.py
```

---

## 📌 Roadmap

- [ ] Add asyncio-based Redis backend integration
- [ ] LLaMA summarization pipelines from Mongo text
- [ ] Full schema validation layer for stored documents

---

## 🧑‍💻 Author

Crafted by **John M. Iriye**  
Email: [Contact@CentralFloridaAttorney.net](mailto:Contact@CentralFloridaAttorney.net)

> Star ⭐️ the repo if this saved you time.  
> Visit: [github.com/CentralFloridaAttorney/zmongo_retriever](https://github.com/CentralFloridaAttorney/zmongo_retriever)

---

## 📄 License

MIT License – see [`LICENSE`](LICENSE) for full terms.

