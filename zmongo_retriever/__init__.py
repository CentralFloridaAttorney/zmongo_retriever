import os

from bson import ObjectId
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv('.env_zmongo_retriever')


class Document:
    def __init__(self, page_content, this_metadata={}):
        self.page_content = page_content
        self.metadata = this_metadata


def get_opinion_from_zcase(zcase):
    casebody = zcase.get('casebody')
    data = casebody.get('data')
    opinions = data.get('opinions')
    this_opinion = opinions[0]
    opinion_text = this_opinion.get('text')
    return opinion_text


class ZMongoRetriever:
    def __init__(self,
                 chunk_size=1024,
                 db_name=os.getenv('MONGO_DATABASE_NAME'),
                 mongo_uri=os.getenv('MONGO_URI'),
                 collection_name=os.getenv('DEFAULT_COLLECTION_NAME'),
                 page_content_field='page_content'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name or os.getenv('DEFAULT_COLLECTION_NAME', 'user')
        self.page_content_field = page_content_field
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)  # Adjust chunk_size as needed

    def _get_relevant_document(self, query, query_by_id=False):
        documents = []
        if query_by_id:
            cursor = self.collection.find({'_id': ObjectId(query)})
        else:
            cursor = self.collection.find({"$text": {"$search": query}})

        for doc in cursor:
            if self.page_content_field == 'opinion':
                page_content = get_opinion_from_zcase(doc)
            else:
                page_content = doc.get(self.page_content_field, "Content not found")
            chunks = self.splitter.split_text(page_content)
            this_metadata = self.create_default_metadata(doc)
            # these_documents = [Document(page_content=chunk, this_metadata=this_metadata) for chunk in chunks]
            these_documents = [Document(page_content=chunk) for chunk in chunks]
            documents.append(these_documents)
        return documents

    def create_default_metadata(self, zobject):
        """
        Creates default metadata for a langchain document.

        Args:
            zobject (dict): The MongoDB document from which metadata is derived.

        Returns:
            dict: A dictionary containing default metadata.
        """
        metadata = {
            "source": "local",  # Indicate the source of the document
            "document_id": str(zobject.get("_id", "N/A")),  # Unique identifier of the document
            "collection_name": self.collection.name,  # Collection from which the document originates
        }
        return metadata

    def invoke(self, query, query_by_id=False):
        document = self._get_relevant_document(query, query_by_id)
        return document
