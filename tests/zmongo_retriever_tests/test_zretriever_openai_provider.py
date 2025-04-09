import unittest
from zmongo_toolbag.zretriever import ZRetriever
from langchain_openai import OpenAIEmbeddings


class DummyRepo:
    db = mongo_client = None


class TestZRetrieverOpenAIProvider(unittest.TestCase):
    def test_default_embedding_provider_is_openai(self):
        retriever = ZRetriever(repository=DummyRepo(), embedding_provider='openai')
        self.assertIsInstance(retriever.embedding_model, OpenAIEmbeddings)

    def test_unspecified_embedding_provider_defaults_to_openai(self):
        retriever = ZRetriever(repository=DummyRepo())
        self.assertIsInstance(retriever.embedding_model, OpenAIEmbeddings)


if __name__ == "__main__":
    unittest.main()
