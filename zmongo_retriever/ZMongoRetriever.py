import json
from datetime import datetime

import tiktoken
from bson.errors import InvalidId
from bson.objectid import ObjectId
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient


def get_opinion_from_zcase(zcase):
    try:
        casebody = zcase.get('casebody')
        data = casebody.get('data')
        opinions = data.get('opinions')
        this_opinion = opinions[0]
        opinion_text = this_opinion.get('text')
        return opinion_text
    except Exception as e:
        return f"No opinion for ObjectId: {zcase.get('_id')}"


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
    def __init__(self, overlap_prior_chunks=0, max_tokens_per_set=4096, chunk_size=512, db_name=None, mongo_uri=None,
                 collection_name=None, page_content_field=None, encoding_name='cl100k_base'):
        self.mongo_uri = mongo_uri or 'mongodb://localhost:49999'
        self.db_name = db_name or 'zcases'
        self.collection_name = collection_name or 'zcases'
        self.page_content_field = page_content_field or 'opinion'
        self.encoding_name = encoding_name
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.chunk_size = chunk_size
        self.max_tokens_per_set = max_tokens_per_set
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self.overlap_prior_chunks = overlap_prior_chunks

    def process_chunks(self, chunks):
        max_tokens = self.max_tokens_per_set
        sized_chunks = []
        current_chunks = []
        current_tokens = 0

        for i, chunk in enumerate(chunks):
            chunk_tokens = self.num_tokens_from_string(page_content=chunk.page_content)
            if current_tokens + chunk_tokens <= max_tokens:
                # Add chunk to current_chunks
                current_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                overlap_start = max(0, len(current_chunks) - self.overlap_prior_chunks)
                # Append current_chunks to
                sized_chunks.append(current_chunks[:])
                current_chunks = current_chunks[overlap_start:]
                current_tokens = sum(self.num_tokens_from_string(page_content=c.page_content) for c in current_chunks)
                current_chunks.append(chunk)
                current_tokens += chunk_tokens

        if current_chunks:
            sized_chunks.append(current_chunks)

        return sized_chunks

    def _create_default_metadata(self, mongo_object):
        return {
            "source": "mongodb",
            "database_name": self.db_name,
            "collection_name": self.collection_name,
            "document_id": str(mongo_object.get("_id", "N/A")),
            "page_content_field": self.page_content_field
        }

    def num_tokens_from_string(self, page_content) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(page_content))
        return num_tokens

    def get_zdocument(self, object_id, existing_metadata=None):
        try:
            this_mongo_record = self.collection.find_one({'_id': ObjectId(object_id)})
            if not this_mongo_record:
                print(f"No record found with ID: {object_id}")
                return None

            page_content = get_opinion_from_zcase(this_mongo_record) if self.page_content_field == 'opinion' else (
                this_mongo_record.get(self.page_content_field, "Content not found"))
            chunks = self.splitter.split_text(page_content)
            metadata = self._create_default_metadata(this_mongo_record)
            combined_metadata = existing_metadata or {}
            combined_metadata.update(metadata)
            return [Document(chunk, combined_metadata) for chunk in chunks]
        except InvalidId as e:
            print(f"Error with ID {object_id}: {e}")
            return None

    def invoke(self, object_ids, existing_metadata=None):
        # Ensure zcase_ids is a list
        if not isinstance(object_ids, list):
            object_ids = [object_ids]

        documents = []
        for object_id in object_ids:
            doc_chunks = self.get_zdocument(object_id=object_id, existing_metadata=existing_metadata)
            if doc_chunks:
                documents.extend(doc_chunks)

        context_sized_chunks = self.process_chunks(documents)
        return context_sized_chunks


if __name__ == "__main__":
    retriever = ZMongoRetriever(overlap_prior_chunks=3, max_tokens_per_set=2048, chunk_size=512)
    these_object_ids = ["65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47", "65eab5493c6a0853d9a9cce7",
                        "65eab55e3c6a0853d9a9cd54", "65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47",
                        "65eab5493c6a0853d9a9cce7", "65eab55e3c6a0853d9a9cd54"]
    these_documents = retriever.invoke(object_ids=these_object_ids)
    for i, group in enumerate(these_documents):
        print(f"Group {i + 1} - Total Documents: {len(group)}")
        for doc in group:
            print(f"page_content: {doc.page_content}... metadata: {doc.metadata}")
