from zmongo_retriever.Document import Document
from zmongo_retriever.ZTokenEstimator import ZTokenEstimator
from bson.errors import InvalidId
from bson.objectid import ObjectId
from zmongo_retriever.data_tools import get_opinion_from_zcase
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ZMongoRetriever:
    def __init__(self, max_tokens_per_set=4096, chunk_size=512, db_name=None, mongo_uri=None, collection_name=None, page_content_field=None):
        self.mongo_uri = mongo_uri or 'mongodb://localhost:49999'
        self.db_name = db_name or 'zcases'
        self.collection_name = collection_name or 'zcases'
        self.page_content_field = page_content_field or 'opinion'
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.chunk_size = chunk_size
        self.max_tokens_per_set = max_tokens_per_set
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    def _create_default_metadata(self, mongo_object):
        return {
            "source": "mongodb",
            "database_name": self.db_name,
            "collection_name": self.collection_name,
            "document_id": str(mongo_object.get("_id", "N/A")),
            "page_content_field": self.page_content_field
        }

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

    def process_chunks(self, chunks):
        max_tokens = self.max_tokens_per_set
        sized_chunks = []
        current_chunk = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = ZTokenEstimator().estimate_tokens(chunk.page_content)
            if current_tokens + chunk_tokens <= max_tokens:
                current_chunk.append(chunk)
                current_tokens += chunk_tokens
            else:
                sized_chunks.append(current_chunk)
                current_chunk = [chunk]
                current_tokens = chunk_tokens

        if current_chunk:
            sized_chunks.append(current_chunk)

        return sized_chunks

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
    retriever = ZMongoRetriever(max_tokens_per_set=2048, chunk_size=512)
    these_object_ids = ["65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47", "65eab5493c6a0853d9a9cce7", "65eab55e3c6a0853d9a9cd54", "65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47", "65eab5493c6a0853d9a9cce7", "65eab55e3c6a0853d9a9cd54"]
    these_documents = retriever.invoke(object_ids=these_object_ids)
    for i, group in enumerate(these_documents):
        print(f"Group {i+1} - Total Documents: {len(group)}")
        for doc in group:
            print(f"Metadata: {doc.metadata}, Content Preview: {doc.page_content[:100]}...")