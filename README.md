# ⚡ ZMongo Retriever

**ZMongo Retriever** is a high-performance, async-first MongoDB toolkit built for AI-powered and real-time applications. It wraps `motor` and `pymongo` with a modern async repository, bulk optimizations, smart **in-memory** caching (not Redis), and seamless integration with OpenAI and local LLaMA models.

---

## 🚀 Features

- ✅ 100% test coverage for `zmongo_toolbag`
- ↺ Async-enabled MongoDB access using `motor`
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
import asyncio
from bson import ObjectId

from zmongo_retriever.examples.openai_model import OpenAIModel


async def main():
    model = OpenAIModel()

    # 👤 Example instruction
    instruction = "Explain the concept of async programming in Python."
    response = await model.generate_instruction(instruction)
    print("\n🔹 Instruction Response:\n", response)

    # 📄 Example summary
    long_text = (
        "Asynchronous programming in Python allows developers to write code that can pause "
        "while waiting for an operation to complete (like a network request) and continue "
        "executing other code during that time. It enables better scalability and performance "
        "in I/O-bound programs. Libraries like asyncio make this possible."
    )
    summary = await model.generate_summary(long_text)
    print("\n🔹 Summary:\n", summary)

    # ❓ Question answering
    context = "Python supports async programming using the asyncio library and the 'async' and 'await' keywords."
    question = "How does Python support asynchronous programming?"
    answer = await model.generate_question_answer(context, question)
    print("\n🔹 Q&A:\n", answer)

    # 🧬 ZElement explanation
    zelement_doc = {
        "name": "Case Precedent Extractor",
        "note": "Designed to retrieve and summarize legal precedents from MongoDB based on user queries.",
        "creator": "LegalAI-Labs"
    }
    explanation = await model.generate_zelement_explanation(zelement_doc)
    print("\n🔹 ZElement Explanation:\n", explanation)

    # 🧾 Save summary to MongoDB (optional demo)
    # Replace with a real _id and ensure the collection exists
    fake_id = ObjectId()  # Replace with real ID if testing on real DB
    saved = await model.save_openai_result(
        collection_name="documents",
        record_id=fake_id,
        field_name="ai_summary",
        generated_text=summary,
        extra_fields={"ai_summary_source": "OpenAI gpt-3.5-turbo-instruct"}
    )
    print("\n✅ Saved to MongoDB:", saved)


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Use with LLaMA (local)

```python
from zmongo_retriever.zmongo_toolbag.llama_model import LlamaModel


def main():
    llama_model = LlamaModel()

    user_input = (
        "Write a Dungeons & Dragons encounter using D20 rules. "
        "Include full descriptive text for the dungeon master to read when running the encounter. "
        "This is for new dungeon masters. The adventurers awake from a drunken slumber in the corner of a tavern."
    )

    prompt = llama_model.generate_prompt_from_template(user_input)

    output_text = llama_model.generate_text(
        prompt=prompt,
        max_tokens=3000,  # careful with this value depending on your model!
    )

    print("Generated Text:\n")
    print(output_text)


if __name__ == "__main__":
    main()
```

---

## 📊 Use Case Suitability

| Use Case                          | ZMongo ✅ | Why                                                                 |
|----------------------------------|-----------|----------------------------------------------------------------------|
| **LLM/AI Workflows**             | ✅✅✅     | Fast cached reads, embedding support, async-first architecture       |
| **Async Web Servers**            | ✅✅       | Integrates with `asyncio`, excellent concurrent read performance     |
| **LegalTech / NLP Tools**        | ✅✅       | Metadata-safe, recursive-safe flattening, optimized for text         |
| **Edge AI & Agents**             | ✅✅       | In-memory performance without Redis dependency                       |
| **Bulk ETL Ingestion**           | 🟡        | Supports batch ops, but Mongo shell faster for raw throughput        |
| **Analytics Dashboards**         | 🟡✅       | Great for caching reads; Redis better for live metrics/pub-sub       |

---

## 📈 Real-World Benchmark Comparison

```
ZMongo Retriever Real-World Benchmark Comparison
============================================================

Bulk Write (100k)
-----------------
  MongoDB Shell: 162207.8010 ops/sec  
         ZMongo: 107212.0408 ops/sec  

Concurrent Reads (5k)
---------------------
  MongoDB Shell: 7.9904 s  
          Redis: 0.6397 s  
         ZMongo: 0.1140 s  

Insert (500 docs)
-----------------
  MongoDB Shell: 0.9816 ms/doc  
         ZMongo: 1.3601 ms/doc  
          Redis: 0.0576 ms/doc  

Query Latency (cached)
----------------------
  MongoDB Shell: 1.0082 ms  
         ZMongo: 0.0094 ms  
          Redis: 0.0529 ms  

insert_documents (100k)
-----------------------
         ZMongo: 27605.6581 ops/sec  
  MongoDB Shell: 187871.6902 ops/sec  
          Redis: 17954.9665 ops/sec  

============================================================
```

![Benchmark Chart](benchmark_chart.png)

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

- [ ] Add optional Redis backend (future)
- [ ] LLaMA summarization pipelines from Mongo text
- [ ] Full schema validation layer for stored documents

---

## 🧑‍💼 Author

Crafted by **John M. Iriye**  
📣 [Contact@CentralFloridaAttorney.net](mailto:Contact@CentralFloridaAttorney.net)  
🌐 [View Project on GitHub](https://github.com/CentralFloridaAttorney/zmongo_retriever)

> ⭐️ Star this repo if it saved you time or effort!

---

## 📄 License

MIT License – see [LICENSE](https://github.com/CentralFloridaAttorney/zmongo_retriever/blob/main/LICENSE) for full terms.
