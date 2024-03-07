# ZMongoRetriever API Documentation

Welcome to the ZMongoRetriever API documentation. ZMongoRetriever is a powerful utility designed for retrieving and processing documents from MongoDB collections, specifically tailored for working with large text documents that may need to be split into manageable chunks before processing.

## Overview

ZMongoRetriever allows you to query MongoDB collections by document ID or text search, returning the specified field's content as a list of `Document` objects. These objects contain manageable chunks of the original text, along with custom metadata for further processing.

## Class: Document

The `Document` class encapsulates a piece of text (a document) and its associated metadata.

### Parameters:
- `page_content` (str): The content of the document.
- `this_metadata` (dict, optional): Metadata associated with the document. Defaults to `None`.

## Function: get_opinion_from_zcase

Extracts the opinion text from a case document.

### Parameters:
- `zcase` (dict): A MongoDB document representing a legal case.

### Returns:
- `str`: The text of the first opinion found in the document, or an error message if no opinion is found.

## Class: ZMongoRetriever

Responsible for retrieving documents from a MongoDB collection and processing them into manageable chunks.

### Parameters:
- `chunk_size` (int, optional): The maximum size of each text chunk. Defaults to `1024`.
- `db_name` (str): The name of the MongoDB database.
- `mongo_uri` (str): The MongoDB connection URI.
- `collection_name` (str): The name of the MongoDB collection to query.
- `page_content_field` (str): The document field containing the text to be processed.

### Methods:

#### \_get_relevant_document

Queries the MongoDB collection for documents and processes the specified content field into chunks.

##### Parameters:
- `query` (str): The query term or document ID.
- `query_by_id` (bool, optional): Whether to query by document ID (`True`) or perform a text search (`False`). Defaults to `False`.
- `existing_metadata` (dict, optional): Additional metadata to combine with the document's default metadata. Defaults to `None`.

##### Returns:
- `List[Document]`: A list of `Document` objects containing chunks of the queried document's content and combined metadata.

#### \_create_default_metadata

Generates default metadata for a document based on MongoDB object details.

##### Parameters:
- `mongo_object` (dict): The MongoDB document from which to derive metadata.

##### Returns:
- `dict`: Default metadata for the document.

#### invoke

The primary method used to query documents and return processed chunks.

##### Parameters:
- `query` (str): The query term or document ID.
- `query_by_id` (bool, optional): Whether to query by document ID or perform a text search. Defaults to `False`.
- `existing_metadata` (dict, optional): Additional metadata to combine with each document's default metadata. Defaults to `None`.

##### Returns:
- `List[Document]` or `Document`: A list of `Document` objects containing text chunks and metadata if multiple documents are found, or a single `Document` object if a specific document is queried by ID.

## Usage Example

```python
# Initialize ZMongoRetriever with your MongoDB details
retriever = ZMongoRetriever(
    db_name='my_database',
    mongo_uri='mongodb://localhost:27017',
    collection_name='my_collection',
    page_content_field='my_text_field'
)

# Query a specific document by ID
documents = retriever.invoke('5f3a5c861d94b1cff536f7db', query_by_id=True)

# Query documents by a text search
documents = retriever.invoke('search term', query_by_id=False, existing_metadata={'your_metadata_key': 'Your metadata related to the key.'})

for doc in documents[0]:  # documents[0] contains the chunks of the first document found
    print(doc.page_content)
    print(doc.metadata)
```

## Conclusion

ZMongoRetriever simplifies the process of retrieving and handling large text documents from MongoDB, making it an invaluable tool for applications requiring pre-processing of text data for analysis or machine learning purposes.