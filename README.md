# ZMongoRetriever Utility

## Overview

ZMongoRetriever is a Python utility for efficiently fetching, processing, and splitting large text documents from MongoDB collections. It's designed to handle documents of any size, split them into manageable chunks, and enrich each chunk with metadata for subsequent analysis or processing. With a built-in token estimator, ZMongoRetriever ensures that chunks are compatible with processing limits of downstream applications, such as machine learning models or text analysis tools.

## Key Features

- **Flexible Document Retrieval**: Fetch documents using MongoDB Object IDs.
- **Automatic Document Splitting**: Dynamically splits documents into smaller chunks.
- **Metadata Enrichment**: Enhances chunks with default and custom metadata.
- **Token Estimation**: Evaluates the token count of chunks to manage processing limits.

## Installation

Ensure Python 3.6+ and MongoDB are installed. Install required Python packages with pip:

```bash
pip install pymongo langchain_text_splitters tiktoken
```

Clone the repository:

```bash
git clone https://github.com/yourgithub/ZMongoRetriever.git
cd ZMongoRetriever
```

## Usage


```python
from zmongo_retriever import ZMongoRetriever

#Initialize ZMongoRetriever with MongoDB connection details and collection information:
retriever = ZMongoRetriever(
                max_tokens_per_set=4096,
                chunk_size=512,
                db_name='your_database_name',
                mongo_uri='mongodb://localhost:27017',
                collection_name='your_collection_name',
                page_content_field='field_containing_text'
            )
object_ids = ["5fc778bfc2e344001f81ae89", "5fc778c0c2e344001f81ae8a"]
documents = retriever.invoke(object_ids=object_ids)

# Invoke the retriever with a list of MongoDB Object IDs:
for i, group in enumerate(documents):
    print(f"Group {i+1} - Total Documents: {len(group)}")
    for doc in group:
        print(f"Metadata: {doc.metadata}, Content Preview: {doc.page_content[:100]}...")
```


### Adjusting Token Limits

Modify `max_tokens_per_set` to fit the processing limit of your downstream application:

```python
retriever = ZMongoRetriever(max_tokens_per_set=2048, chunk_size=512)
```

## Components

### Document

Represents a chunk of text and its associated metadata, facilitating easy access and processing.

### ZTokenEstimator

Estimates the number of tokens in a chunk, ensuring compatibility with processing limits.

## Contributing

We welcome contributions! For improvements or bug fixes, please open an issue or submit a pull request.

## License

This project is available under the MIT License. See the LICENSE file for more details.