# File: test_zmongo_retriever.py
import os
import unittest

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

# REM: put a .env file with OPENAI_API_KEY in tests directory
class TestJsonKeys(unittest.TestCase):
    def setUp(self):
        # Initialize MongoDB client and collection
        self.mongo_client = MongoClient(os.environ.get('MONGO_URI'))
        self.mongo_db = self.mongo_client[os.environ.get('MONGO_DB_NAME')]
        self.mongo_collection = self.mongo_db[os.environ.get('DEFAULT_COLLECTION_NAME')]

    def test_get_mongodb_metadata(self):
        # Invoke the method with an object _id value from the collection
        document = self.mongo_collection.find_one({'_id': ObjectId('65f1b6beae7cd4d4d1d3ae8d')})
        # document_metadata = convert_json_to_metadata(document)
        # If you retrieved an object from mongodb then it will have an ObjectId('_id')
        # this_id = document_metadata.get('_id')
        # self.assertIsInstance(ObjectId(this_id), ObjectId)
        # An example of a more complex object as shown in UNDERSTANDING_KEY_SELECTION.md for report.details.content:
        # report_details_content = document_metadata.get('report.details.content')



if __name__ == '__main__':
    unittest.main()
