# ZMongoRetriever

## Introduction

`ZMongoRetriever` is a Python class designed to facilitate the retrieval and processing of documents stored in a MongoDB collection. It leverages MongoDB queries to fetch documents based on specific criteria and utilizes text splitting techniques to handle long documents efficiently. This guide will help you understand how to implement and use its functions effectively.

## Setup

### Requirements

- Python 3.x
- MongoDB
- LangChain
- dotenv: For loading environment variables
- pymongo: MongoDB driver for Python

### Installation

Ensure you have MongoDB running and accessible. Install the required Python packages by running:

```bash
pip install python-dotenv pymongo langchain llama-cpp-python openai
```

### Environment Configuration

Create a `.env` file in your project directory and define the following variables:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE_NAME=your_database_name
DEFAULT_COLLECTION_NAME=your_default_collection_name
MODEL_PATH=path_to_your_model
```

Load the environment variables in your script:

```python
from dotenv import load_dotenv

load_dotenv('.env_zmongo_retriever')
```

## Usage

### Initializing ZMongoRetriever

To start using `ZMongoRetriever`, you need to initialize it with your MongoDB collection details. Optionally, you can specify the field within your documents that contains the content you're interested in (e.g., `opinion`).

```python
from zmongo_retriever import ZMongoRetriever

retriever = ZMongoRetriever(collection_name='zcases', page_content_field='opinion')
```

### Fetching Documents

To fetch documents based on a text search or by document ID:

```python
# Fetch documents by text search
documents = retriever.invoke(query="search term")

# Fetch documents by MongoDB ObjectID
documents_by_id = retriever.invoke(query='65cf9acdb347eec24fd6b02a', query_by_id=True)
```

### Processing Documents

After retrieving documents, you might want to split long texts into smaller chunks for further processing. This is handled automatically by the `ZMongoRetriever` using the `RecursiveCharacterTextSplitter`, which divides texts into manageable pieces.

### Creating Default Metadata

`ZMongoRetriever` also facilitates creating default metadata for each document. This metadata includes information like the document's source, unique identifier, and collection name. This is especially useful for tracking and managing documents across different stages of processing.

### Example

Here is a complete example that retrieves documents by ID, splits them into chunks, and prints the results:

```python
from zmongo_retriever import ZMongoRetriever
collection_name = 'zcases'
page_content_field = 'opinion'
document_id = '65cf9acdb347eec24fd6b02a'

retriever = ZMongoRetriever(collection_name=collection_name, page_content_field=page_content_field)
documents_by_id = retriever.invoke(document_id, query_by_id=True)

for doc_chunks in documents_by_id:
    for chunk in doc_chunks:
        print(chunk.page_content)
```

## Conclusion

`ZMongoRetriever` is a powerful tool for working with document-based data stored in MongoDB. By leveraging MongoDB's query capabilities and handling large documents through text splitting, it simplifies the process of retrieving and processing documents for various applications, including NLP and data analysis tasks. Remember to customize the class and methods according to your specific requirements and data structure.