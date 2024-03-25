# File: test_zmongo_retriever.py

import unittest

from pymongo import MongoClient

import zconstants
from zmongo_retriever import ZMongoRetriever, Document


class TestZMongoRetriever(unittest.TestCase):
    def setUp(self):
        # Initialize ZMongoRetriever with mock MongoDB client and collection
        self.mongo_db = MongoClient(zconstants.MONGO_URI)
        self.mongo_collection = self.mongo_db[zconstants.ZCASES_COLLECTION]
        self.zmongo_retriever = ZMongoRetriever()

    def test_get_relevant_document_by_id(self):
        # Invoke the method with an object _id value from the collection
        documents = self.zmongo_retriever.invoke('65f1b6beae7cd4d4d1d3ae8d')

        # Assert that Document objects are returned with correct content and metadata
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0][0], Document)
        self.assertEqual(documents[0][0].metadata['source'], 'mongodb')
        self.assertEqual(documents[0][0].metadata['collection_name'], 'zcases')


if __name__ == '__main__':
    unittest.main()
