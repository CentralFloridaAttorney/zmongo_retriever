# ZMongoRetriever API

The ZMongoRetriever API is a sophisticated Python library designed for retrieving, processing, and encoding documents stored in MongoDB. It supports chunking documents into manageable sizes, encoding them for further machine learning tasks, and organizing chunks based on token limits.

## Features

- Efficient retrieval of documents from MongoDB.
- Automatic chunking of documents into smaller pieces.
- Optional encoding of document chunks with customizable embedding models.
- Token limit management for chunk sets, ideal for processing with machine learning models.
- Metadata generation for each document chunk.

## Installation

To install ZMongoRetriever, use GitHub's CLI tool `gh`:

```sh
gh repo clone CentralFloridaAttorney/zmongo_retriever
cd zmongo_retriever
pip install -r requirements.txt
```

## Quick Start

```python
from zmongoretriever import ZMongoRetriever

# Initialize the ZMongoRetriever with MongoDB connection details
retriever = ZMongoRetriever(
    db_name="your_database",
    collection_name="your_collection",
    page_content_field="your_document_field"
)

# Fetch, process, and encode document chunks from MongoDB
documents = retriever.invoke(object_ids=["60b4fa10d8d8c2c7b8e4fa7e"])

for document in documents:
    print(document.page_content, document.metadata)
```

## API Reference

### Initialization

```python
ZMongoRetriever(
    overlap_prior_chunks=0,
    max_tokens_per_set=4096,
    chunk_size=512,
    embedding_length=1536,
    db_name=None,
    mongo_uri=None,
    collection_name=None,
    page_content_field=None,
    encoding_name='cl100k_base',
    use_encoding=False
)
```

- **`overlap_prior_chunks`**: Number of chunks to overlap for context continuity.
- **`max_tokens_per_set`**: Maximum tokens per chunk set; less than 1 returns all chunks in a single list.
- **`chunk_size`**: Character count per chunk.
- **`embedding_length`**: Length of the embedding vector if encoding is used.
- **`use_encoding`**: Enables or disables chunk encoding.

### Methods

- **`invoke`**: Main method to retrieve and process documents by object IDs.
  
  ```python
  invoke(object_ids, existing_metadata=None)
  ```

- **`get_zcase_chroma_retriever`**: Fetches documents and compiles them into a unified Chroma database.

  ```python
  get_zcase_chroma_retriever(object_ids, database_dir)
  ```

- **`get_chunk_sets`**: Organizes document chunks into token-limited sets.

  ```python
  get_chunk_sets(chunks)
  ```

- **`_create_default_metadata`**: Generates default metadata for document chunks.

  ```python
  _create_default_metadata(mongo_object)
  ```

## License

Distributed under the MIT License. See `LICENSE` for more information.