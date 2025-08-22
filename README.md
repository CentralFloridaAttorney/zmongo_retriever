
# 🦦 ZMongo Retriever

## Please pardon the dust — this is a work in progress!  I recently changed the output of ZMongo to be a SafeResult object, which is a serializable wrapper around the result that can be used in tests and other places.  I will update the README soon to reflect this change.

A modern Python toolkit for **MongoDB-powered retrieval-augmented generation (RAG)**.  
It wraps MongoDB access with a clean async API, integrates with **Google Gemini embeddings**, supports **fast local vector search (with optional HNSW acceleration)**, and plugs directly into **LangChain retrievers** for end-to-end AI workflows.  

---

## ✨ Features

- **ZMongo**:  
  A robust async MongoDB client wrapper with:
  - Built-in caching layer
  - Clean `SafeResult` wrapper for predictable results
  - CRUD operations, aggregation, bulk writes
  - BSON → JSON conversion and back

- **SafeResult**:  
  A serializable, test-friendly result wrapper with:
  - `.ok()` / `.fail()` convenience methods
  - `.original()` to restore BSON/keys
  - `.model_dump()`, `.to_json()`, `.to_metadata()` for inspection

- **ZMongoEmbedder**:  
  - Uses **Google Gemini** to generate embeddings
  - Cache-first design: stores/reuses chunk embeddings in MongoDB
  - Batch-friendly API for texts and documents
  - `embed_and_store()` writes embeddings directly to your docs

- **LocalVectorSearch**:  
  - Fast cosine similarity search over stored embeddings
  - Handles chunked embeddings and normalization
  - Optional **HNSW acceleration** (via `hnswlib`)
  - Supports exact rescoring and candidate re-ranking

- **ZMongoRetriever (LangChain integration)**:  
  - Implements LangChain’s `BaseRetriever`
  - Retrieves documents above a similarity threshold
  - Strips embeddings from metadata, adds `retrieval_score`
  - End-to-end demo included

- **ZMongoSystemManager (GUI)**:  
  - Tkinter-based MongoDB manager
  - Backup & restore collections to JSON
  - Inspect DB stats
  - Browse and restore from snapshots

---

## 🚀 Quickstart

### 1. Install
```bash
git clone https://github.com/CentralFloridaAttorney/zmongo_retriever.git
cd zmongo_retriever
pip install -r requirements.txt
````

### 2. Configure

Create `~/resources/.env_local` with:

```ini
MONGO_URI=mongodb://127.0.0.1:27017
MONGO_DATABASE_NAME=test
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Retriever Demo

```bash
python zretriever.py
```

This will:

* Insert a small knowledge base into MongoDB
* Embed and store each fact
* Query with LangChain’s retriever
* Print the top matches with scores

---

## 🧩 Example

```python
# openai_main.py

import asyncio
from datetime import datetime
from bson.objectid import ObjectId

from examples.openai_model import OpenAIModel
from zmongo import ZMongo

this_zmongo = ZMongo()


async def log_to_zmongo(op_type: str, prompt: str, result: str, meta: dict = None) -> bool:
    doc = {
        "operation": op_type,
        "prompt": prompt,
        "result": result,
        "timestamp": datetime.now(),
        "meta": meta or {}
    }
    insert_result = await this_zmongo.insert_document("openai_logs", doc)
    return True if insert_result else False


async def main():
    model = OpenAIModel()

    # 👤 Instruction
    instruction = "Explain how to use ZMongo to query all documents where status is 'active'."
    instruction_response = await model.generate_instruction(instruction)
    print("\n🔹 Instruction Response:\n", instruction_response)
    await log_to_zmongo("instruction", instruction, instruction_response)

    # 📄 Summary
    long_text = (
        "ZMongo is an asynchronous MongoDB client wrapper that simplifies insert, update, find, and bulk operations. "
        "It integrates seamlessly with async frameworks and is designed to work well with AI workflows."
    )
    summary_response = await model.generate_summary(long_text)
    print("\n🔹 Summary Response:\n", summary_response)
    await log_to_zmongo("summary", long_text, summary_response)

    # ❓ Q&A
    context = (
        "ZMongo uses Python's Motor driver under the hood and provides utility methods for easy querying, "
        "bulk inserts, updates, and logging. It supports coroutine-based design patterns."
    )
    question = "What async features make ZMongo a good choice for AI applications?"
    qa_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    qa_response = await model.generate_question_answer(context, question)
    print("\n🔹 Q&A Response:\n", qa_response)
    await log_to_zmongo("question_answer", qa_prompt, qa_response)

    # 🧬 ZElement Explanation
    zelement_doc = {
        "name": "ZMongo Query Helper",
        "note": "Simplifies MongoDB operations for async apps.",
        "creator": "Business Process Applications, Inc."
    }
    explanation_response = await model.generate_zelement_explanation(zelement_doc)
    print("\n🔹 ZElement Explanation:\n", explanation_response)
    await log_to_zmongo("zelement_explanation", str(zelement_doc), explanation_response)

    # 🧾 Simulate saving result into documents
    fake_id = ObjectId()
    save_success = await model.save_openai_result(
        collection_name="documents",
        record_id=fake_id,
        field_name="ai_generated_summary",
        generated_text=summary_response,
        extra_fields={"ai_summary_source": "OpenAI Chat Completion"}
    )
    print("\n✅ Saved to documents collection:", save_success)


if __name__ == "__main__":
    asyncio.run(main())

```

---

## 🛠 Components

* [`data_processing.py`](./data_processing.py) → `SafeResult`, `DataProcessor`
* [`zmongo.py`](./zmongo.py) → Async MongoDB wrapper with caching
* [`zmongo_embedder.py`](./zmongo_embedder.py) → Gemini embeddings + cache-first storage
* [`unified_vector_search.py`](./unified_vector_search.py) → Local cosine search with optional HNSW
* [`zmongo_retriever.py`](./zmongo_retriever.py) → LangChain retriever implementation
* [`zmongo_system_manager.py`](./zmongo_system_manager.py) → Tkinter GUI for DB management

---

## 🧪 Tests

Run the test suite with:

```bash
pytest tests
```

---

## 📦 Roadmap

* [ ] Add restore modes ("Add and Update", "Update without Adding") in `ZMongoSystemManager`
* [ ] Support multiple embedding providers (OpenAI, Cohere, etc.)
* [ ] Add hybrid (BM25 + vector) search

---

## 📜 License

MIT License © 2025 [CentralFloridaAttorney](https://github.com/CentralFloridaAttorney)

---

## 🤝 Contributing

Pull requests are welcome!
If you’d like to extend functionality (e.g., support for other embedding providers), open an issue first to discuss.



