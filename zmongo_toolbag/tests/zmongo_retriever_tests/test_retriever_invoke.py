import unittest
from unittest.mock import AsyncMock, MagicMock
from langchain.schema import Document

from zmongo_toolbag import ZRetriever


class TestZRetrieverInvoke(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_returns_raw_documents_when_max_tokens_disabled(self):
        # Arrange
        mock_repo = MagicMock()
        retriever = ZRetriever(repository=mock_repo, max_tokens_per_set=0)  # < 1 triggers raw return
        dummy_doc = Document(page_content="test text", metadata={"x": 1})
        retriever.get_zdocuments = AsyncMock(return_value=[dummy_doc])

        # Act
        results = await retriever.invoke(collection="mock_collection", object_ids=["123"])

        # Assert
        self.assertEqual(results, [dummy_doc])
