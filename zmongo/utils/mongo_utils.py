from bson.errors import InvalidId
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import pandas as pd
from zmongo import zconstants
from zmongo_support.utils.data_processing import DataProcessing


class MongoUtils:
    """
    A utility class for interacting with MongoDB.
    This class contains static methods for performing common MongoDB operations
    such as fetching documents, inserting documents, updating documents, and more.
    """

    client = MongoClient(zconstants.MONGO_URI)
    db = client[zconstants.MONGO_DATABASE_NAME]
    zcases_collection = db[zconstants.ZCASES_COLLECTION]
    stock_collection = db[zconstants.YAHOO_FINANCE_COLLECTION]

    @staticmethod
    def is_valid_objectid(oid):
        """
        Checks if a given value is a valid ObjectId.

        Args:
            oid (str): The value to check.

        Returns:
            bool: True if valid ObjectId, False otherwise.
        """
        try:
            ObjectId(oid)
            return True
        except (InvalidId, TypeError):
            return False

    @staticmethod
    def fetch_document(collection, identifier):
        """
        Fetches a single document from the specified collection by its ID or a custom query.

        Args:
            collection (str): The name of the collection.
            identifier (Union[str, ObjectId, dict]): The ID of the document to fetch, or a custom query.

        Returns:
            dict: The fetched document, or None if not found.
        """
        try:
            if isinstance(identifier, dict):
                query = identifier
            else:
                if MongoUtils.is_valid_objectid(identifier):
                    query = {'_id': ObjectId(identifier)}
                else:
                    query = {'username': identifier}  # Custom query by username if not ObjectId

            document = MongoUtils.db[collection].find_one(query)
            if document:
                return document
            else:
                print(f"Document not found in collection '{collection}' with query: {query}")
                return None
        except Exception as e:
            print(f"Error fetching document from collection '{collection}': {e}")
            return None

    @staticmethod
    def insert_document(collection, document):
        """
        Inserts a document into the specified collection.

        Args:
            collection (str): The name of the collection.
            document (dict): The document to insert.

        Returns:
            str: The ID of the inserted document, or None if error occurs.
        """
        try:
            result = MongoUtils.db[collection].insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error inserting document into collection '{collection}': {e}")
            return None

    @staticmethod
    def update_document(collection, document_id, update_data):
        """
        Updates a document in the specified collection by its ID.

        Args:
            collection (str): The name of the collection.
            document_id (str): The ID of the document to update.
            update_data (dict): The update data.

        Returns:
            pymongo.results.UpdateResult: The result of the update operation, or None if error occurs.
        """
        try:
            if MongoUtils.is_valid_objectid(document_id):
                result = MongoUtils.db[collection].update_one({'_id': ObjectId(document_id)}, {'$set': update_data})
                return result
            else:
                print(f"Invalid ObjectId: {document_id}")
                return None
        except Exception as e:
            print(f"Error updating document in collection '{collection}': {e}")
            return None

    @staticmethod
    def update_document_push(collection, document_id, update_data):
        """
        Pushes data to an array field in a document in the specified collection by its ID.

        Args:
            collection (str): The name of the collection.
            document_id (str): The ID of the document to update.
            update_data (dict): The data to push.

        Returns:
            pymongo.results.UpdateResult: The result of the update operation, or None if error occurs.
        """
        try:
            if MongoUtils.is_valid_objectid(document_id):
                return MongoUtils.db[collection].update_one({'_id': ObjectId(document_id)}, {'$push': update_data})
            else:
                print(f"Invalid ObjectId: {document_id}")
                return None
        except Exception as e:
            print(f"Error pushing data to document in collection '{collection}': {e}")
            return None

    @staticmethod
    def update_document_pull(collection, document_id, pull_data):
        """
        Pulls data from an array field in a document in the specified collection by its ID.

        Args:
            collection (str): The name of the collection.
            document_id (str): The ID of the document to update.
            pull_data (dict): The data to pull from the array.

        Returns:
            pymongo.results.UpdateResult: The result of the pull operation, or None if error occurs.
        """
        try:
            if MongoUtils.is_valid_objectid(document_id):
                return MongoUtils.db[collection].update_one({'_id': ObjectId(document_id)}, {'$pull': pull_data})
            else:
                print(f"Invalid ObjectId: {document_id}")
                return None
        except Exception as e:
            print(f"Error pulling data from document in collection '{collection}': {e}")
            return None

    @staticmethod
    def fetch_documents(collection, query=None):
        """
        Fetches multiple documents from the specified collection based on a query.

        Args:
            collection (str): The name of the collection.
            query (dict): The query to match documents.

        Returns:
            list: A list of matched documents.
        """
        try:
            return list(MongoUtils.db[collection].find(query or {}))
        except Exception as e:
            print(f"Error fetching documents from collection '{collection}': {e}")
            return []

    @staticmethod
    def delete_document(collection, document_id):
        """
        Deletes a document from the specified collection by its ID.

        Args:
            collection (str): The name of the collection.
            document_id (str): The ID of the document to delete.

        Returns:
            pymongo.results.DeleteResult: The result of the delete operation, or None if error occurs.
        """
        try:
            if MongoUtils.is_valid_objectid(document_id):
                return MongoUtils.db[collection].delete_one({'_id': ObjectId(document_id)})
            else:
                print(f"Invalid ObjectId: {document_id}")
                return None
        except Exception as e:
            print(f"Error deleting document from collection '{collection}': {e}")
            return None


    @staticmethod
    def fetch_stock_data(ticker):
        """
        Fetches stock data from MongoDB.
        :param ticker: Stock ticker symbol.
        :return: JSON data from MongoDB if found, None otherwise.
        """
        try:
            document = MongoUtils.stock_collection.find_one({"ticker": ticker})
            if document:
                return document.get("data")
            return None
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None

    @staticmethod
    def store_stock_data(ticker, data):
        """
        Stores or updates stock data in MongoDB.
        :param ticker: Stock ticker symbol.
        :param data: Stock data to be stored.
        """
        try:
            MongoUtils.stock_collection.update_one(
                {"ticker": ticker},
                {"$set": {"data": data, "last_updated": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            print(f"Error storing stock data for {ticker}: {e}")

    @staticmethod
    def get_zcase_by_id(zcase_id):
        """
        Fetches a zcase by its ID and converts it to JSON format.

        Args:
            zcase_id (str): The ID of the zcase.

        Returns:
            dict: The zcase document in JSON format, or an empty dict if not found.
        """
        record = MongoUtils.zcases_collection.find_one({'_id': ObjectId(zcase_id)})
        if record is None:
            return {}
        else:
            return DataProcessing.convert_object_to_json(record)

    @staticmethod
    def list_ai_review_results(exclude_no_opinion=True):
        """
        Lists all AI review results from the zcases collection, optionally excluding those without opinions.

        Args:
            exclude_no_opinion (bool): Whether to exclude results with "No opinion provided".

        Returns:
            list: A list of AI review results.
        """
        all_results = []

        query = {"ai_review_requests.ai_review_result": {"$exists": True}}
        for zcase in MongoUtils.zcases_collection.find(query):
            for request in zcase.get('ai_review_requests', []):
                if 'ai_review_result' in request and (
                        not exclude_no_opinion or "No opinion provided" not in request['ai_review_result']):
                    result = {
                        'zcase_id': str(zcase['_id']),
                        'name_abbreviation': zcase.get('name_abbreviation', 'N/A'),
                        'request_type': request.get('request_type', 'N/A'),
                        'ai_review_result': request['ai_review_result'],
                        'status': request.get('status', 'N/A')
                    }
                    all_results.append(result)

        return all_results

    @staticmethod
    def get_collection_as_dataframe(collection_name=None) -> pd.DataFrame:
        """
        Retrieves all documents from a MongoDB collection and returns them as a pandas DataFrame
        with each row representing a document and columns for the metadata with path-like keys.

        Args:
            collection_name (str, optional): The collection name to retrieve documents from.

        Returns:
            pd.DataFrame: A DataFrame containing the documents and their metadata.
        """
        this_collection = MongoUtils.db[collection_name] if collection_name else MongoUtils.zcases_collection

        documents = []
        for record in this_collection.find():
            metadata = DataProcessing.convert_mongo_to_metadata(record)
            documents.append(metadata)

        return pd.DataFrame(documents)

    @staticmethod
    def get_values_as_list(df: pd.DataFrame, prefix: str = 'embedding_') -> list:
        """
        Retrieves the values of all columns that start with a given prefix and returns them as a list.

        Args:
            df (pd.DataFrame): The DataFrame containing the embedding columns.
            prefix (str): The prefix used to identify the embedding columns.

        Returns:
            list: A list containing the values of the embedding columns.
        """
        return DataProcessing.get_values_as_list(df, prefix)
