# zmongo_retriever/zmongo_toolbag/run_retriever.py
import logging
import asyncio
import os
from typing import List, Any

from bson import ObjectId
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from zmongo import ZMongo
from zmongo_embedder import ZMongoEmbedder
from unified_vector_search import LocalVectorSearch
from data_processing import DataProcessor
logger = logging.getLogger(__name__)


class ZMongoRetriever(BaseRetriever):
    """
    A LangChain-compliant retriever that uses a unified, in-memory vector
    search engine (`LocalVectorSearch`) to find relevant documents from a
    standard ZMongo repository.
    """
    repository: Any
    embedder: Any
    vector_searcher: Any
    collection_name: str
    embedding_field: str = "embeddings"
    content_field: str = "text"
    top_k: int = 10
    similarity_threshold: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        embeddings = await self.embedder.embed_text(query)
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        search_result = await self.vector_searcher.search(query_embedding, top_k=self.top_k)
        if not search_result.success:
            raise RuntimeError(f"Local vector search failed: {search_result.error}")
        return self._format_results(search_result.data)

    def _format_results(self, items: List[dict]) -> List[Document]:
        final_docs = []
        for item in items:
            score = item.get("retrieval_score", 0.0)
            if score >= self.similarity_threshold:
                doc_data = item.get("document", item)
                content = DataProcessor.get_value(doc_data, self.content_field)
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                metadata = {k: v for k, v in doc_data.items() if k != self.embedding_field}
                metadata["retrieval_score"] = score
                final_docs.append(Document(page_content=content, metadata=metadata))
        return final_docs


COLLECTION_NAME = "retriever_demo_knowledge_base"

async def populate_data(documents: List[dict], repository: ZMongo, collection_name: str):
    embedder = ZMongoEmbedder(collection=collection_name, repository=repository)
    texts_to_embed = [doc.get("text") for doc in documents if doc.get("text")]
    if texts_to_embed:
        embedding_results = await embedder.embed_texts_batched(texts_to_embed)
        for doc in documents:
            if doc.get("text") in embedding_results:
                doc["embeddings"] = embedding_results[doc["text"]]
    zmongo = ZMongo()
    await zmongo.insert_documents(collection=collection_name, documents=documents)
    await asyncio.sleep(1)
    zmongo.close()

async def main():
    repo = ZMongo()
    collection_name = COLLECTION_NAME
    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=collection_name,
        embedding_field="embeddings",
        chunked_embeddings=True,
        exact_rescore=True,
    )
    embedder = ZMongoEmbedder(repository=repo, collection=collection_name)

    retriever = ZMongoRetriever(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=collection_name,
        similarity_threshold=0.80,
        top_k=3,
    )

    print(f"--- Setting up the '{collection_name}' collection ---")
    await repo.delete_documents(collection=collection_name, query={})
    knowledge_base = [
        {
            "_id": ObjectId(),
            "topic": "Astronomy",
            "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined.",
        },
        {
            "_id": ObjectId(),
            "topic": "Biology",
            "text": "Mitochondria are organelles that create energy-rich molecules (ATP) and are often called the powerhouse of the cell.",
        },
        {
            "_id": ObjectId(),
            "topic": "History",
            "text": "The Roman Empire was one of the most influential civilizations in world history.",
        },
    ]
    await populate_data(knowledge_base, repository=repo, collection_name=collection_name)

    query = "What is the powerhouse of the cell?"
    results = await retriever.ainvoke(query)
    logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
    logger.info(f"Results: {results}")

    knowledge_base = [
        {"topic": "Astronomy", "text": "Jupiter is the fifth planet from the Sun and the largest. It is a gas giant."},
        {"topic": "Biology", "text": "Mitochondria generate most of the cell's ATP; they're the 'powerhouse' of the cell."},
        {"topic": "History", "text": "The Renaissance marked a transition from the Middle Ages to modernity."},
    ]
    for doc in knowledge_base:
        insert_res = await repo.insert_document(COLLECTION_NAME, doc)
        doc_id = insert_res.data["inserted_id"]
        await embedder.embed_and_store(doc_id, doc["text"])

    print(f"Successfully inserted and embedded {len(knowledge_base)} facts.\n")

    query = "What is the fifth planet from the sun?"
    print(f"--- Invoking retriever with query: '{query}' ---")
    results = await retriever.ainvoke(query)

    print(f"\nFound {len(results)} relevant document(s):")
    if not results:
        print("No documents met the similarity threshold.")
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"  Content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")

    await repo.db.drop_collection(collection_name)
    repo.close()

if __name__ == "__main__":
    if not all(os.getenv(k) for k in ["MONGO_URI", "GEMINI_API_KEY"]):
        print("ERROR: Please set MONGO_URI and GEMINI_API_KEY in your ~/resources/.env_local file.")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        asyncio.run(main())
