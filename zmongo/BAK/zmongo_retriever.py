import json
import logging
import os
from datetime import datetime
import tiktoken
from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

from llama_index.embeddings import ollama

load_dotenv()
def get_keys_from_json(json_object):
    this_metadata = convert_json_to_metadata(json_object=json_object)
    return list(this_metadata.keys())


def get_value(json_data, key):
    """Retrieves a value from nested JSON data using a dot-separated key."""
    keys = key.split('.')
    value = json_data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        elif isinstance(value, list) and k.isdigit():
            index = int(k)
            value = value[index] if 0 <= index < len(value) else None
        else:
            return None  # Key not found or invalid access
    return value


def get_by_key_from_json(json_object, key_sequence):
    """
    Traverse a nested JSON object using a key sequence in the format of
    "first_get_value.second_get_value...", where parts of the sequence
    can also indicate indexes in lists by using integer values.
    """
    key_parts = key_sequence.split('.')
    current_value = json_object

    for part in key_parts:
        try:
            # Attempt to convert part into an integer for list indexing
            index = int(part)
            current_value = current_value[index]  # Access list element
        except ValueError:
            # Part is not an integer, access dictionary value
            if part in current_value:
                current_value = current_value[part]
            else:
                # Return a default value or raise an error if the key doesn't exist
                return None  # or raise KeyError(f"Key {part} does not exist.")
        except (IndexError, TypeError):
            # IndexError for list index out of range, TypeError for incorrect access type
            return None  # or handle the error differently

    return current_value


def convert_json_to_metadata(json_object, existing_metadata=None, metadata_prefix=''):
    """
    Recursively convert JSON data to metadata format suitable for passing to the retriever.

    Args:
        json_object (dict or list): JSON data to be converted to metadata. Can be either a dictionary or a list.
        existing_metadata (dict, optional): Existing metadata dictionary to add the converted data. Default is None.
        metadata_prefix (str, optional): Prefix for keys to distinguish nested fields. Default is an empty string.

    Returns:
        dict: Metadata dictionary containing converted JSON data.
    """
    if existing_metadata is None:
        existing_metadata = {}

    if isinstance(json_object, dict):
        for key, value in json_object.items():
            new_prefix = f"{metadata_prefix}.{key}" if metadata_prefix else key
            # Recursive processing for nested structures
            convert_json_to_metadata(value, existing_metadata, new_prefix)
    elif isinstance(json_object, list):
        for idx, item in enumerate(json_object):
            item_prefix = f"{metadata_prefix}.{idx}" if metadata_prefix else str(idx)
            # Recursive processing for items within lists
            convert_json_to_metadata(item, existing_metadata, item_prefix)
    else:
        # Directly add non-iterable items to the metadata
        existing_metadata[metadata_prefix] = str(json_object)

    return existing_metadata


def convert_object_to_json(data):
    """
    Convert a potentially nested list (or any data structure) to a JSON string.
    Handles custom objects by attempting to convert them to dictionaries.
    """
    serializable_data = convert_object_to_serializable(data)
    this_json = json.dumps(serializable_data, indent=4)
    return json.loads(this_json)


def convert_object_to_serializable(obj):
    """
    Recursively convert objects in the data structure to JSON serializable types,
    handling MongoDB ObjectId, datetime objects, custom objects, tuples, and more.
    """
    if isinstance(obj, dict):
        return {key: convert_object_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_object_to_serializable(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return {attr: convert_object_to_serializable(getattr(obj, attr)) for attr in dir(obj)
                if not attr.startswith(('__', '_')) and not callable(getattr(obj, attr))}
    else:
        # Fallback for other types that json.dumps cannot serialize directly
        try:
            # Attempt to use the default serializer; if this fails, convert to string
            return json.dumps(obj, default=str)
        except TypeError:
            return str(obj)


class Document:
    def __init__(self, page_content, this_metadata=None):
        self.page_content = page_content
        self.metadata = this_metadata if this_metadata else {}


class ZMongoRetriever:
    """
    ZMongoRetriever is designed to retrieve and process documents from a MongoDB collection,
    potentially splitting them into smaller chunks and optionally encoding them using a specified embedding model.

    Parameters:
        overlap_prior_chunks (int): Number of tokens to overlap with prior chunks to ensure continuity in embeddings. Default is 0.
        max_tokens_per_set (int): Maximum number of tokens to be included in a single set of documents or chunks. Default is 4096. Values less than 1 will result in all chunks being returned in a single list.
        chunk_size (int): Size of each chunk (in number of characters) into which documents are split. Default is 512.
        embedding_length (int): The length of the embedding vector. This is used if encoding is enabled. Default is 1536.
        db_name (str, optional): Name of the MongoDB database. Defaults to 'zcases'.
        mongo_uri (str, optional): URI for connecting to MongoDB. Defaults to 'mongodb://localhost:49999'.
        collection_name (str, optional): Name of the collection within the MongoDB database to retrieve documents from. Defaults to 'zcases'.
        page_content_key (str, optional): Field name in the collection documents that contains the text content. Defaults to 'opinion'.
        encoding_name (str): Name of the encoding to use for embeddings. Default is 'cl100k_base'.
        use_embedding (bool): Flag to enable or disable the use of embeddings for chunking. Default is False.

    Attributes:
        client (MongoClient): The MongoDB client instance.
        db (Database): The MongoDB database instance.
        collection (Collection): The MongoDB collection instance from which documents are retrieved.
        splitter (RecursiveCharacterTextSplitter): The text splitter used for dividing documents into smaller chunks.
        embedding_model (OpenAIEmbeddings): The model used for generating embeddings, configured with an API key.
    """

    def __init__(self,
                 overlap_prior_chunks=0,
                 max_tokens_per_set=4096,
                 chunk_size=512,
                 embedding_length=1536,
                 db_name=os.getenv("MONGO_DB_NAME"),
                 mongo_uri=os.getenv("MONGO_URI"),
                 collection_name=os.getenv("DEFAULT_COLLECTION_NAME"),
                 page_content_key=os.getenv("PAGE_CONTENT_KEY"),
                 encoding_name=os.getenv("EMBEDDING_ENCODING"),
                 use_embedding=False):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.page_content_key = page_content_key
        self.encoding_name = encoding_name
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.chunk_size = chunk_size  # Note: If use_embedding then chunk_size = embedding_length
        self.max_tokens_per_set = max_tokens_per_set
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self.overlap_prior_chunks = overlap_prior_chunks

        self.ollama_embedding_model = ollama.OllamaEmbedding(model_name=os.getenv('Embeddings_Model_Name', 'llama2'),)
        self.openai_embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = self.openai_embedding_model



    def get_chunk_sets(self, chunks):
        """
        Organizes chunks of document content into sets, where each set has a total token count
        that does not exceed a specified maximum. This method also supports overlapping of chunks
        between sets to ensure continuity and context are preserved when processing chunk sets separately.

        This approach is particularly useful for processing or analysis tasks that have input size
        constraints, such as feeding chunks into machine learning models that can only handle a certain
        number of tokens at a time.

        Parameters:
            chunks (list[Document]): A list of Document instances representing chunks of document content
                that need to be organized into sets.

        Returns:
            list[list[Document]]: A list of lists, where each inner list represents a set of chunks
                (Document instances) whose combined token count does not exceed the predefined maximum.
                Chunks at the boundary of two sets may overlap based on the `overlap_prior_chunks` setting.

        The method iterates over each chunk, counting the tokens it contains, and aggregates chunks
        into current working set until adding another chunk would exceed the maximum token limit.
        Once this limit is reached, the current set is finalized and a new set is started, potentially
        with an overlap of chunks for continuity. This process continues until all chunks have been
        allocated to a set, ensuring all document content is accounted for without exceeding token limits.
        """
        max_tokens = self.max_tokens_per_set
        sized_chunks = []
        current_chunks = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.num_tokens_from_string(page_content=chunk.page_content)
            if current_tokens + chunk_tokens <= max_tokens:
                current_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                overlap_start = max(0, len(current_chunks) - self.overlap_prior_chunks)
                sized_chunks.append(current_chunks[:])
                # Reinitialize current_chunks with the overlapped chunks for continuity.
                current_chunks = current_chunks[overlap_start:]
                # Recalculate the total token count for the new starting set.
                current_tokens = sum(self.num_tokens_from_string(page_content=c.page_content) for c in current_chunks)
                current_chunks.append(chunk)
                current_tokens += chunk_tokens

        # Ensure the last set of chunks is added to the return value.
        if current_chunks:
            sized_chunks.append(current_chunks)

        return sized_chunks

    def _create_default_metadata(self, mongo_object):
        """
        Generates a default metadata dictionary for a given MongoDB document object.

        This method creates a standardized structure of metadata that includes information
        about the source of the document, its originating database and collection, and specific
        document identifiers. This metadata is intended to be used alongside document content
        for downstream processing or analysis, providing context about the origin and location
        of the data.

        Parameters:
            mongo_object (dict): The MongoDB document object from which to extract metadata.

        Returns:
            dict: A dictionary containing key metadata about the MongoDB document, including the
                  source ('mongodb'), database name, collection name, document ID, and the field
                  name containing the page content.

        The returned metadata includes the database name, collection name, and document ID as
        essential identifiers. Additionally, it specifies the `page_content_field` to indicate
        which field of the document contains the main content of interest. This helps in maintaining
        a consistent data structure and facilitates easier integration with document processing
        and analysis pipelines.
        """
        return {
            "source": "mongodb",
            "database_name": self.db_name,
            "collection_name": self.collection_name,
            "document_id": str(mongo_object.get("_id", "N/A")),
            "page_content_field": self.page_content_key
        }

    def num_tokens_from_string(self, page_content) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(page_content))
        return num_tokens

    def get_zdocuments(self, object_ids, page_content_key="casebody.data.opinions.0.text",
                       existing_metadata=None):
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        these_zdocuments = []
        for object_id in object_ids:
            try:
                this_mongo_record = self.collection.find_one({'_id': ObjectId(object_id)})
                if not this_mongo_record:
                    print(f"No record found with ID: {object_id}")
                    return None
                page_content = get_value(json_data=this_mongo_record, key=page_content_key)

                # Ensure page_content is a string; if not, log an error and skip processing this document.
                if not isinstance(page_content, str):
                    print(f"Page content for ID {object_id} is not a string. Skipping document.")
                    continue

                chunks = self.splitter.split_text(page_content)

                # Create and combine metadata.
                metadata = self._create_default_metadata(mongo_object=convert_object_to_json(this_mongo_record))
                combined_metadata = existing_metadata or {}
                combined_metadata.update(metadata)
                for chunk in chunks:
                    these_zdocuments.append(Document(page_content=chunk, this_metadata=combined_metadata))

            except InvalidId as e:
                print(f"Error with ID {object_id}: {e}")

        return these_zdocuments

    def invoke(self, object_ids, page_content_key="casebody.data.opinions.0.text", existing_metadata=None):
        """
        Retrieves and processes a set of documents identified by their MongoDB object IDs,
        optionally applying encoding and splitting them into manageable chunks. It then
        groups these chunks into sets that do not exceed a predefined maximum token limit.

        This method serves as a high-level interface for fetching and preparing documents
        from the database for downstream processing or analysis. It ensures that each document
        is fetched, processed into chunks (with optional encoding), and then these chunks are
        aggregated into larger sets based on a maximum token count constraint.

        Parameters:
            object_ids (str or list[str]): A single object ID or a list of object IDs for the documents
                to be retrieved and processed.
            page_content_key (str): The path-like key according to the list returned by get_keys_from_json(json_object) is used to get the page_content in the Document.
            existing_metadata (dict, optional): Metadata to be merged with each document's metadata.
                This can include additional context or information necessary for processing. Defaults to None.

        Returns:
            list: Depending on the `max_tokens_per_set` configuration, this method returns either a flat list
                of all document chunks or a list of lists, where each inner list represents a set of chunks
                that together do not exceed the maximum token limit.

        The process involves iterating over each provided object ID, retrieving the document associated
        with that ID, and then processing it into chunks. These chunks are then optionally grouped into
        sets that comply with the `max_tokens_per_set` constraint, facilitating easier handling in scenarios
        where token count is a limiting factor (e.g., input size constraints of machine learning models).
        """
        # Ensure object_ids is a list
        if not isinstance(object_ids, list):
            object_ids = [object_ids]

        documents = []
        for object_id in object_ids:
            # It seems there's a typo: 'get_zdocuments' should probably be 'get_zdocument'
            doc_chunks = self.get_zdocuments(object_ids=object_id,
                                             page_content_key=page_content_key,
                                             existing_metadata=existing_metadata)
            if doc_chunks:
                documents.extend(doc_chunks)

        # Handling based on the max_tokens_per_set limit
        if self.max_tokens_per_set < 1:
            return documents
        else:
            # Aggregate document chunks into sets that comply with the max token limit
            context_sized_chunks = self.get_chunk_sets(documents)
            return context_sized_chunks


# Import your retriever and embedder
from zmongo.BAK.zmongo_embedder import ZMongoEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

import os
import asyncio


async def main():
    # Initialize the retriever
    retriever = ZMongoRetriever(
        overlap_prior_chunks=3,
        max_tokens_per_set=-1,
        chunk_size=512,
        use_embedding=True
    )

    # Fetch sample documents from MongoDB to get ObjectIds.
    sample_docs_cursor = retriever.collection.find({}, {"_id": 1}).limit(5)
    sample_docs = sample_docs_cursor.to_list(length=5)
    sample_object_ids = [str(doc["_id"]) for doc in sample_docs]

    if not sample_object_ids:
        print("No documents found in the collection.")
        return
    else:
        print(f"Fetched ObjectIds: {sample_object_ids}")

        # Retrieve and process the documents using their ObjectIds.
        documents = retriever.invoke(
            object_ids=sample_object_ids,
            page_content_key=os.getenv("PAGE_CONTENT_KEY", "default_key")
        )

        if not documents:
            print("No documents were retrieved or processed.")
            return
        else:
            print(f"Retrieved {len(documents)} document chunks:")
            for i, document in enumerate(documents):
                content = getattr(document, "page_content", "")
                metadata = getattr(document, "metadata", "N/A")
                print(f"\nDocument {i + 1}:")
                print(f"Page Content (truncated): {content[:200]}...")
                print(f"Metadata: {metadata}")

    # Initialize the embedder
    collection_name = os.getenv('DEFAULT_COLLECTION_NAME', 'default_collection')
    page_content_key = os.getenv('PAGE_CONTENT_KEY', 'content_key')

    embedder = ZMongoEmbedder(
        collection_name=collection_name,
        page_content_keys=[page_content_key]
    )
    sample_text = "This is a sample text to demonstrate embedding."

    try:
        # Attempt to get the embedding
        embedding_vector = await embedder.get_embedding(sample_text)
        if not embedding_vector:
            raise ValueError("Embedding vector is empty or None.")


        print("\nSample text embedding vector (truncated):", embedding_vector[:5])
    except Exception as e:
        print(f"Failed to retrieve embedding: {e}")


if __name__ == "__main__":
    asyncio.run(main())
