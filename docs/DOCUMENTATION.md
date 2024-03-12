## Class `ZMongoRetriever`

This class is designed to retrieve and process documents from a MongoDB collection, potentially splitting them into smaller chunks and optionally encoding them using a specified embedding model.

### Constructor

```python
def __init__(self, overlap_prior_chunks=0, max_tokens_per_set=4096, chunk_size=512, embedding_length=1536, db_name=None, mongo_uri=None, collection_name=None, page_content_field=None, encoding_name='cl100k_base')
```

Initializes a new instance of the `ZMongoRetriever` class with specified configuration for MongoDB connection, document processing, and optional encoding.

#### Parameters:
- `overlap_prior_chunks` (int): Number of tokens to overlap with prior chunks to ensure continuity in embeddings. Default is 0.
- `max_tokens_per_set` (int): Maximum number of tokens to be included in a single set of documents or chunks. Default is 4096.
- `chunk_size` (int): Size of each chunk (in number of characters) into which documents are split. Default is 512.
- `embedding_length` (int): Length of the embedding vector, used if encoding is enabled. Default is 1536.
- `db_name` (str, optional): Name of the MongoDB database. Defaults to 'zcases'.
- `mongo_uri` (str, optional): URI for connecting to MongoDB. Defaults to 'mongodb://localhost:49999'.
- `collection_name` (str, optional): Name of the MongoDB collection to retrieve documents from. Defaults to 'zcases'.
- `page_content_field` (str, optional): Field name in collection documents containing the text content. Defaults to 'opinion'.
- `encoding_name` (str): Name of the encoding to use for embeddings. Default is 'cl100k_base'.

### Methods

#### `get_chunk_sets`
```python
def get_chunk_sets(self, chunks)
```
Organizes chunks of document content into sets, ensuring each set's total token count does not exceed the specified maximum.

#### Parameters:
- `chunks` (list[Document]): List of Document instances representing chunks of document content.

#### Returns:
- A list of lists, where each inner list represents a set of chunks whose combined token count does not exceed the predefined maximum.

#### `_create_default_metadata`
```python
def _create_default_metadata(self, mongo_object)
```
Generates a default metadata dictionary for a given MongoDB document object.

#### Parameters:
- `mongo_object` (dict): The MongoDB document object from which to extract metadata.

#### Returns:
- A dictionary containing key metadata about the MongoDB document.

#### `num_tokens_from_string`
```python
def num_tokens_from_string(self, page_content) -> int
```
Returns the number of tokens in a text string.

#### Parameters:
- `page_content` (str): Text content to count tokens.

#### Returns:
- The number of tokens in the given text string.

#### `get_zdocuments`
```python
def get_zdocuments(self, object_ids, page_content_key_index=116, existing_metadata=None)
```
Retrieves, processes, optionally encodes, and splits documents into manageable chunks, wrapping each chunk with metadata into a Document instance.

#### Parameters:
- `object_ids` (str, list[str]): MongoDB ObjectIds of the documents to retrieve.
- `page_content_key_index` (int): Index of the key used to get the `page_content` in the document.
- `existing_metadata` (dict, optional): Existing metadata to be combined with the document's metadata.

#### Returns:
- A list of Document instances, each representing a chunk of the original document's content, combined with metadata.

#### `invoke`
```python
def invoke(self, object_ids, page_content_key_index=116, existing_metadata=None)
```
Retrieves and processes documents identified by MongoDB object IDs, applying encoding and splitting into chunks as configured.

#### Parameters:
- `object_ids` (str or list[str]): Object IDs for the documents to be retrieved and processed.
- `page_content_key_index` (int): Index of the key used to get the `page_content` in the document.
- `existing_metadata` (dict, optional): Metadata to be merged with each document's metadata.

#### Returns:
- A list or a list of lists of document chunks, organized according to the `max_tokens_per_set` configuration.


#### `get_zcase_chroma_retriever`
```python
def get_zcase_chroma_retriever(self, object_ids, database_dir, page_content_key_index=116)
```
Retrieves and processes documents from MongoDB records identified by `object_ids`, splits them into manageable chunks if necessary, and compiles them into a list of Chroma databases. Each database contains a chunked document. This method aims to minimize redundant API calls for embedding by reusing existing Chroma databases when available. New documents or chunks are processed and added to a combined Chroma database, which is then returned for further use.

##### Parameters:
- `object_ids` (list): A list of object IDs representing the documents to be retrieved and processed.
- `database_dir` (str): The directory name under which the combined Chroma database should be stored.
- `page_content_key_index` (int): The index of the key used to retrieve the `page_content` in the document, according to the list returned by `get_keys_from_json(json_object)`.

##### Returns:
- `list`: A list containing the combined Chroma database instances.

##### Description:
This method involves loading existing Chroma databases for each object ID if available. Otherwise, the corresponding document is fetched, split into chunks, and a new Chroma database is created and persisted. Finally, all data, both from existing and newly created databases, are consolidated into a single combined list of Chroma databases for efficiency and convenience.

The process begins with checking for an existing Chroma database for each object ID. If one exists, it's loaded; otherwise, the document is fetched from MongoDB, processed, and added to a new or existing Chroma database. The method ultimately returns a list of Chroma database instances that represent the processed and chunked documents.