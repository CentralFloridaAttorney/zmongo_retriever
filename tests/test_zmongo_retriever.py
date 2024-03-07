# File: test_zmongo_retriever.py

import unittest
from unittest.mock import MagicMock
from src.zmongo_retriever import ZMongoRetriever, Document


class TestZMongoRetriever(unittest.TestCase):
    def setUp(self):
        # Initialize ZMongoRetriever with mock MongoDB client and collection
        self.mock_collection = MagicMock()
        self.mock_db = MagicMock()
        self.mock_db.__getitem__.return_value = self.mock_collection
        self.mock_client = MagicMock()
        self.mock_client.__getitem__.return_value = self.mock_db

        self.retriever = ZMongoRetriever(
            mongo_uri='mongodb://localhost:49999',
            chunk_size=1024,
            collection_name='zcases',
            page_content_field='opinion'
        )
        self.retriever.client = self.mock_client

    def test_get_relevant_document_by_id(self):
        # Mock MongoDB find method to return a document
        mock_document = {'_id': '123', 'opinion': 'Test opinion content'}
        self.mock_collection.find.return_value = [mock_document]

        # Invoke the method with query_by_id=True
        documents = self.retriever.invoke('123', query_by_id=True)

        # Assert that Document objects are returned with correct content and metadata
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0][0], Document)
        self.assertEqual(documents[0][0].page_content, 'Test opinion content')
        self.assertEqual(documents[0][0].metadata['document_id'], '123')

    def test_get_relevant_document_by_search(self):
        # Mock MongoDB find method to return a document
        mock_document = {'_id': '456', 'opinion': 'Another test opinion content'}
        self.mock_collection.find.return_value = [mock_document]

        # Invoke the method with query_by_id=False
        documents = self.retriever.invoke('search query', query_by_id=False)

        # Assert that Document objects are returned with correct content and metadata
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0][0], Document)
        self.assertEqual(documents[0][0].page_content, 'Another test opinion content')
        self.assertEqual(documents[0][0].metadata['document_id'], '456')


if __name__ == '__main__':
    unittest.main()
