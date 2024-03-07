# ZMongo Retriever

ZMongo Retriever is a Python utility for retrieving documents from MongoDB collections. It includes functionality to fetch documents based on queries and split them into smaller chunks for processing.

## Installation

To use ZMongo Retriever, you'll need to install the required dependencies:

```bash
pip install langchain_text_splitters pymongo bson
```

## Usage

### Importing ZMongo Retriever

```python
from bson import ObjectId
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
```

### Initializing ZMongo Retriever

```python
retriever = ZMongoRetriever()
```

### Retrieving Documents

```python
# Fetch documents based on a query
documents = retriever.invoke("your_query_here")

# Fetch document by ID
document = retriever.invoke("your_document_id", query_by_id=True)
```

## Example

```python
# Import the required modules
from bson import ObjectId
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from zmongo_retriever import ZMongoRetriever
# Initialize ZMongo Retriever
mongo_uri = 'mongodb://localhost:27017' # Your mongo_uri
this_collection_name = 'zcases'  # Your MongoDB collection
this_page_content_field = 'opinion'  # Specify the field to use as page_content
document_id = '65d995ee2051723e1bb6f154'  # Example ObjectId('_id') value
chunk_size = 1024 # larger values for chunk_size may solve problems with exceeding your token limit


retriever = ZMongoRetriever(mongo_uri=mongo_uri,
                            chunk_size=chunk_size,
                            collection_name=this_collection_name,
                            page_content_field=this_page_content_field)
documents = retriever.invoke(document_id, query_by_id=True)
# Print the retrieved documents
for document_chunk in documents:
    for chunk in document_chunk:
        print(chunk.page_content)
```

## Documentation

### `ZMongoRetriever` Class

#### Methods:

- `__init__(chunk_size=1024, db_name='zcases', mongo_uri='mongodb://localhost:49999', collection_name='zcases', page_content_field='opinion')`: Initializes the ZMongoRetriever instance.

- `invoke(query, query_by_id=False)`: Invokes the retriever to fetch documents based on the provided query.

### `Document` Class

Represents a document fetched from MongoDB.

#### Attributes:

- `page_content`: The content of the document.
- `metadata`: Metadata associated with the document.

### `create_default_metadata` Method

Creates default metadata for a langchain document.

#### Arguments:

- `mongo_object (dict)`: The MongoDB document from which metadata is derived.

#### Returns:

- `dict`: A dictionary containing default metadata.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by John M. Iriye