# ZMongoRetriever

`ZMongoRetriever` is a Python library designed to facilitate the retrieval, processing, and encoding of documents from MongoDB collections. It's especially suited for handling large datasets that require chunking and embedding for advanced machine learning applications. Through an elegant interface, it supports document splitting, custom encoding with OpenAI models, and direct integration with MongoDB databases.

## Features

- **Document Retrieval:** Seamlessly fetch documents from MongoDB collections.
- **Dynamic Chunking:** Split documents into manageable chunks based on character count or embedding size.
- **Embedding Support:** Encode document chunks using OpenAI's embedding models for deep learning tasks.
- **Flexible Configuration:** Customize chunk sizes, token overlaps, and database connections to fit your project needs.
- **Metadata Conversion:** Convert JSON to structured metadata for enhanced document handling.

## Installation

Before you begin, ensure you have MongoDB and Python 3.6+ installed on your system. Clone this repository or download the `ZMongoRetriever` module directly. Dependencies can be installed via pip:

```bash
pip install -r requirements.txt
```

## Environment Variable File

You must have a file named '.env' with the appropriate values for the following:

```angular2html
OPENAI_API_KEY=___
```


## Quick Start

To get started with `ZMongoRetriever`, follow these steps:

1. **Initialize MongoDB Connection:**

```python
from pymongo import MongoClient
from zconstants import MONGO_URI

client = MongoClient(MONGO_URI)
```

2. **Create an Instance of ZMongoRetriever:**

```python
from zmongo_retriever import ZMongoRetriever

retriever = ZMongoRetriever(mongo_uri=MONGO_URI, db_name='your_database', collection_name='your_collection')
```

3. **Retrieve and Process Documents:**

```python
object_ids = ["ObjectId1", "ObjectId2"]
documents = retriever.invoke(object_ids=object_ids, page_content_key_index=116)
```

## Advanced Usage

### Encoding Document Chunks

Enable encoding to process document chunks with OpenAI's embeddings:

```python
retriever.use_encoding = True
encoded_chunks = retriever.invoke(object_ids=object_ids, page_content_key_index=116)
```

### Custom Chunking and Overlaps

Customize the chunk size and token overlap for nuanced control over document processing:

```python
retriever.chunk_size = 1024  # Characters
retriever.overlap_prior_chunks = 2  # Number of chunks repeated in a subsequent Document list
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or add new features.

## License

Distributed under the MIT License. See `LICENSE` for more information.
