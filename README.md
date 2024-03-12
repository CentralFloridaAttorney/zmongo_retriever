# ZMongoRetriever API

The ZMongoRetriever API is a powerful Python tool designed for efficient retrieval, processing, and encoding of documents stored in MongoDB collections. It supports chunking of documents into manageable pieces, optional encoding for machine learning models, and organizing chunks into sets based on token limits.

## Features

- **Document Retrieval**: Fetch documents using MongoDB object IDs.
- **Automatic Chunking**: Split documents into smaller, manageable chunks.
- **Optional Encoding**: Encode chunks using a specified embedding model.
- **Token Limit Management**: Organize chunks into sets that adhere to a maximum token count, suitable for model inputs.

## Installation

To install ZMongoRetriever, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourgithub/zmongoretriever.git
cd zmongoretriever
pip install -r requirements.txt
```

## Quick Start

Below is a simple example to demonstrate how to use the ZMongoRetriever API:

```
# Initialize the retriever with MongoDB connection details
zmongo_retriever = ZMongoRetriever(
    mongo_uri="mongodb://localhost:27017",
    db_name="your_database_name",
    collection_name="your_collection_name",
    page_content_field="your_field_name"
)

# Retrieve and process documents
documents = zmongo_retriever.invoke(object_ids=["your_object_id_here"])

# Explore the processed chunks
for doc in documents:
    print(doc)
```

## Configuration

The `ZMongoRetriever` class can be customized with several parameters:

- `overlap_prior_chunks`: Number of tokens to overlap between chunk sets.
- `max_tokens_per_set`: Maximum number of tokens allowed per set of chunks.
- `chunk_size`: Number of characters per chunk.
- `embedding_length`: Length of the embedding vector (used when encoding is enabled).
- `use_encoding`: Flag to enable or disable chunk encoding.

Refer to the class documentation for more details on each parameter.

## Contributing

Contributions to the ZMongoRetriever project are welcome. Here's how you can contribute:

1. **Fork the Repository**: Create a fork of our repository on GitHub.
2. **Create a Feature Branch**: Make your changes in a new git branch.
3. **Submit a Pull Request**: Submit your changes for review.

Please ensure your code adheres to the project's coding standards and include tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.