---

# 🧠 ZRetriever: Easy Chunked Retrieval from MongoDB for AI & LLM Workflows

The `ZRetriever` class helps you **grab documents from MongoDB**, **break them into smart chunks**, and **get them ready for embeddings or large language models (LLMs)**. It's perfect for building AI-powered apps like summarizers, search engines, or any NLP pipeline.

---

## 🌟 What Can It Do?

✅ Retrieve MongoDB documents by ID  
✅ Select specific fields using `page_content_key`  
✅ Split long text into manageable **chunks**  
✅ Group chunks by **token count** to fit model limits  
✅ Works with **OpenAI** and **Ollama (e.g., Mistral)** embeddings  
✅ Outputs LangChain-compatible `Document` objects

---

## 🛠️ Getting Started

### Step 1: Initialize the Retriever

```python
from zmongo_toolbag.zmongo_retriever import ZRetriever

retriever = ZRetriever(
    repository=zmongo_instance,  # An instance of your ZMongo connection
    overlap_prior_chunks=1,  # Optional: How many chunks should overlap for context
    max_tokens_per_set=4096,  # Token budget per batch of chunks
    chunk_size=512,  # Number of characters per chunk
    embedding_length=1536,  # For your embedding model (optional)
    encoding_name='cl100k_base',  # Token encoding (e.g., for OpenAI)
    use_embedding=True  # Toggle embedding-aware behavior
)
```

---

## 🔍 Step 2: Extract Documents by ID

Use the `get_zdocuments()` method to fetch documents and convert them into LangChain `Document` objects.

```python
documents = await retriever.get_zdocuments(
    collection="zcases",
    object_ids="60b621fe9b8b9e3a3e4b4321",
    page_content_key="casebody.data.opinions.0.text"
)
```

👆 This grabs one document and pulls the opinion text using the path provided in `page_content_key`.

---

## 🧱 Step 3: Smart Chunking

If you want to group the documents into token-limited sets (for better LLM handling), use:

```python
chunk_sets = retriever.get_chunk_sets(documents)
```

Each set will respect the `max_tokens_per_set` you configured.

---

## ⚙️ Step 4: Or Just Use `.invoke()` for Everything

Shortcut method that runs both retrieval and chunking in one go:

```python
results = await retriever.invoke(
    collection="zcases",
    object_ids="60b621fe9b8b9e3a3e4b4321",
    page_content_key="casebody.data.opinions.0.text"
)
```

Returns a list of token-safe `Document` sets, ready for embedding or LLM input.

---

## 🧮 Bonus: Count Tokens in a String

You can also calculate how many tokens a string will use (e.g., for OpenAI limits):

```python
tokens = retriever.num_tokens_from_string("This is my sample text.")
```

---

## 🧠 Embedding Models

ZRetriever includes built-in support for:

- `OpenAIEmbeddings()` (default)
- `OllamaEmbeddings(model="mistral")` — great for local models!

You can swap models by modifying `retriever.embedding_model`.

---

## 🏷️ Metadata on Chunks

Each LangChain `Document` returned includes helpful metadata:

```json
{
  "source": "mongodb",
  "database_name": "your-db",
  "collection_name": "your-collection",
  "document_id": "60b621fe9b8b9e3a3e4b4321"
}
```

You can also pass your own metadata using the `existing_metadata` argument.

---

## 📦 Requirements

Make sure you’ve installed:

```bash
pip install langchain langchain-community tiktoken bson
```

---

## ✅ Best For…

- Chunking MongoDB documents for **LLMs**
- Preparing embeddings for **semantic search**
- Splitting long legal/academic texts
- Efficient **token-aware prompt construction**

---

💡 **Pro Tip**: Combine `ZRetriever` with `ZMongoEmbedder` to embed and store vector data directly into your MongoDB documents.