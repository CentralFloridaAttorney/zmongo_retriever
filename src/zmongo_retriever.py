from bson.errors import InvalidId
from bson.objectid import ObjectId
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient


class Document:
    def __init__(self, page_content, this_metadata=None):
        self.page_content = page_content
        self.metadata = this_metadata


def get_opinion_from_zcase(zcase):
    try:
        casebody = zcase.get('casebody')
        data = casebody.get('data')
        opinions = data.get('opinions')
        this_opinion = opinions[0]
        opinion_text = this_opinion.get('text')
        return opinion_text
    except Exception as e:
        return f"No opinion: {str(e)}"


class ZMongoRetriever:
    def __init__(self,
                 chunk_size=1024,
                 db_name='zcases',
                 mongo_uri='mongodb://localhost:49999',
                 collection_name='zcases',
                 page_content_field='opinion'
                 ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.page_content_field = page_content_field
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size)  # Changing chunk_size may fix token size problems

    def _get_relevant_document(self, query, query_by_id=False, existing_metadata=None):
        if existing_metadata is None:
            existing_metadata = {}

        documents = []
        try:
            if query_by_id:
                cursor = self.collection.find({'_id': ObjectId(query)})
            else:
                cursor = self.collection.find({"$text": {"$search": query}})
        except InvalidId as e:
            print(f"Error: {e}")
            default_metadata = self._create_default_metadata(mongo_object={query: 'Not found.'})
            default_document = Document(page_content="No Page Found", this_metadata=default_metadata)
            return default_document

        for doc in cursor:
            # Add custom handling for other fields when needed
            if self.page_content_field == 'opinion':
                page_content = get_opinion_from_zcase(doc)
            else:
                page_content = doc.get(self.page_content_field, "Content not found")
            chunks = self.splitter.split_text(page_content)
            new_metadata = self._create_default_metadata(doc)
            combined_metadata = {**existing_metadata, **new_metadata}
            these_documents = [Document(page_content=chunk, this_metadata=combined_metadata) for chunk in chunks]
            documents.append(these_documents)
        return documents

    def _create_default_metadata(self, mongo_object):
        """
        Creates default metadata for a langchain document.

        Args:
            mongo_object (dict): The MongoDB document from which metadata is derived.

        Returns:
            dict: A dictionary containing default metadata.
        """
        metadata = {
            "source": "mongodb",  # Indicate the source of the document
            "database_name": self.db_name,  # The name of the mongo database
            "collection_name": self.collection.name,  # Collection from which the document originates
            "document_id": str(mongo_object.get("_id", "N/A")),  # Unique identifier of the document
            "page_content_field": self.page_content_field  # The field containing the page_content
        }
        return metadata

    def invoke(self, query, query_by_id=False, existing_metadata=None):
        # Note: bad metadata can affect processing and output significantly
        document = self._get_relevant_document(query, query_by_id, existing_metadata=existing_metadata)
        return document
