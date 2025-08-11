# ZMongoEmbedder: Async Embedding with MongoDB and OpenAI

`ZMongoEmbedder` integrates OpenAI embedding generation into MongoDB workflows via the `ZMongo` repository. It allows asynchronous embedding of documents with caching and persistent storage.

## ✨ Features

- **Asynchronous Embedding** with OpenAI using `openai.AsyncOpenAI`
- **Embedding Caching** to avoid redundant API calls
- **MongoDB Storage** via `ZMongo.save_embedding`
- **Hash-based Deduplication** using SHA-256
- **Simple API** for calling and storing embeddings

---

## ⚙️ Setup

### 1. Requirements
- Python 3.8+
- `openai`, `motor`, `bson`, `python-dotenv`

### 2. Environment Variables

Place the following in your `.env` file:

```env
OPENAI_API_KEY_APP=your-openai-api-key
EMBEDDING_MODEL=text-embedding-ada-002
```

You should also configure `ZMongo` with:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE_NAME=yourDatabase
```

---

## 🚀 Usage

```python
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo.embedder import ZMongoEmbedder

import asyncio
from bson.objectid import ObjectId

zmongo = ZMongo()
embedder = ZMongoEmbedder(repository=zmongo, collection="yourCollection")

async def main():
    text = "Example text to embed."
    doc_id = ObjectId("65123abc...")
    await embedder.embed_and_store(document_id=doc_id, text=text)

asyncio.run(main())
```

---

## 🔍 API Overview

### `embed_text(text: str) -> List[float]`
- Embeds text
- Reuses cached embedding from `_embedding_cache` if available

### `embed_and_store(document_id: ObjectId, text: str, embedding_field: str = "embedding")`
- Embeds and stores the vector in the specified field in MongoDB

---

## 🧠 Internals
- Uses `SHA-256` hashes to index repeated content
- Uses `openai.AsyncOpenAI.embeddings.create()` to call the API
- Stores the raw text and hash in `_embedding_cache` collection

---

## 📦 Example: Caching Check
```python
await embedder.embed_text("The quick brown fox")
# On subsequent calls, logs: 🔁 Reusing cached embedding for text hash: ...
```

---

## 🛠️ Troubleshooting
- Check `.env` for correct API keys
- Ensure `ZMongo` is initialized and points to a running MongoDB
- Use `try/except` for graceful API failure handling

---

## 🏁 Conclusion
`ZMongoEmbedder` simplifies and accelerates embedding pipelines using OpenAI and MongoDB. It is ideal for scalable NLP and vector database tasks where caching and structured storage are essential.

---

For JSON key selection and advanced field control, see `PAGE_CONTENT_KEYS.md`.
