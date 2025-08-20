import logging
import asyncio
import os
from typing import List

from bson import ObjectId
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from zmongo_toolbag.unified_vector_search import LocalVectorSearch
# Adjust these imports to match your project's structure
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.data_processing import DataProcessor

logger = logging.getLogger(__name__)


class ZMongoRetriever(BaseRetriever):
    """
    A LangChain-compliant retriever that uses a unified, in-memory vector
    search engine (`LocalVectorSearch`) to find relevant documents from a
    standard ZMongo repository.
    """
    repository: ZMongo
    embedder: ZMongoEmbedder
    vector_searcher: LocalVectorSearch
    collection_name: str
    embedding_field: str = "embeddings"
    content_field: str = "text"
    top_k: int = 10
    similarity_threshold: float = 0.0

    class Config:
        """Pydantic config to allow custom class types."""
        arbitrary_types_allowed = True

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Synchronous implementation for the core retrieval logic.
        This is the required abstract method from the LangChain BaseRetriever.
        """
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronous implementation of the core retrieval logic. This method
        embeds the query and uses the LocalVectorSearch instance to find documents.
        """
        embeddings = await self.embedder.embed_text(query)
        if not embeddings:
            return []

        query_embedding = embeddings[0]

        # Delegate the search operation to the unified vector searcher
        search_result = await self.vector_searcher.search(query_embedding, top_k=self.top_k)

        if not search_result.success:
            raise RuntimeError(f"Local vector search failed: {search_result.error}")

        return self._format_results(search_result.data)

    # --- Helper Methods ---

    def _format_results(self, items: List[dict]) -> List[Document]:
        """
        Filters search results by the similarity_threshold and formats them into
        LangChain Document objects for standardized output.
        """
        final_docs = []
        for item in items:
            score = item.get("retrieval_score", 0.0)
            if score >= self.similarity_threshold:
                doc_data = item.get("document", item)
                content = DataProcessor.get_value(doc_data, self.content_field)
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""

                # Exclude the large embedding vector from the final metadata
                metadata = {k: v for k, v in doc_data.items() if k != self.embedding_field}
                metadata['retrieval_score'] = score
                final_docs.append(Document(page_content=content, metadata=metadata))
        return final_docs


# --- Example Usage ---
COLLECTION_NAME = "retriever_demo_knowledge_base"

# Helper Function to populate test data
async def populate_data(embedder: ZMongoEmbedder, documents: List[dict]):
    """Helper to insert and embed test documents."""
    texts_to_embed = [doc.get("text") for doc in documents if doc.get("text")]
    if texts_to_embed:
        embedding_results = await embedder.embed_texts_batched(texts_to_embed)
        for doc in documents:
            if doc.get("text") in embedding_results:
                doc["embeddings"] = embedding_results[doc["text"]]
    zmongo = ZMongo()
    await zmongo.insert_documents(COLLECTION_NAME, documents)
    await asyncio.sleep(1)
    zmongo.close()

async def main():
    """Demonstrates the full end-to-end workflow for the ZMongoRetriever."""

    # 1. Initialize all components
    repo = ZMongo()
    embedder = ZMongoEmbedder(collection=COLLECTION_NAME)
    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=COLLECTION_NAME,
        embedding_field="embeddings",
        chunked_embeddings=True,
        exact_rescore=True
    )
    retriever = ZMongoRetriever(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=COLLECTION_NAME,
        similarity_threshold=0.10,  # Set a reasonably high threshold for the demo
        top_k=3
    )

    # 2. Clean up and populate the database with distinct facts
    print(f"--- Setting up the '{COLLECTION_NAME}' collection ---")
    await repo.delete_documents(COLLECTION_NAME, {})
    knowledge_base = [
        {
            "_id": ObjectId(),
            "topic": "Astronomy",
            "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined."
        },
        {
            "_id": ObjectId(),
            "topic": "Biology",
            "text": "Mitochondria are organelles that act like a digestive system which takes in nutrients, breaks them down, and creates energy rich molecules for the cell. They are often referred to as the powerhouse of the cell."
        },
        {
            "_id": ObjectId(),
            "topic": "History",
            "text": "The Roman Empire was one of the most influential civilizations in world history, known for its contributions to law, architecture, and language, lasting for over a thousand years."
        },
    ]
    await populate_data(embedder, knowledge_base)

    # 2. Act: Ask a specific question related to only one of the facts.
    query = "What is the powerhouse of the cell?"
    results = await retriever.ainvoke(query)
    logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
    logger.info(f"Results: {results}")
    # 3. Assertions
    # assert len(results) == 1, "Should only retrieve the single most relevant document."
    #
    # top_result = results[0]
    # assert isinstance(top_result, Document)
    #
    # # Check that the content and metadata are correct
    # assert "Mitochondria" in top_result.page_content
    # assert top_result.metadata["topic"] == "Biology"
    # assert top_result.metadata["retrieval_score"] >= retriever.similarity_threshold

    knowledge_base = [
        {"topic": "Astronomy",
         "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant."},
        {"topic": "Biology",
         "text": "Mitochondria are often referred to as the powerhouse of the cell because they generate most of the cell's supply of adenosine triphosphate (ATP)."},
        {"topic": "History",
         "text": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries."}
    ]

    # Use the embedder to process and store the documents with their embeddings
    for doc in knowledge_base:
        insert_res = await repo.insert_document(COLLECTION_NAME, doc)
        doc_id = insert_res.data['inserted_id']
        await embedder.embed_and_store(doc_id, doc["text"])

    print(f"Successfully inserted and embedded {len(knowledge_base)} facts.\n")

    # 3. Define a specific query and invoke the retriever
    query = "What is the fifth planet from the sun?"
    print(f"--- Invoking retriever with query: '{query}' ---")

    # Use the LangChain standard `ainvoke` method, which is inherited from BaseRetriever
    results = await retriever.ainvoke(query)

    # 4. Print the results
    print(f"\nFound {len(results)} relevant document(s):")
    if not results:
        print("No documents met the similarity threshold.")
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"  Content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")

    # 5. Clean up
    await repo.db.drop_collection(COLLECTION_NAME)
    repo.close()


if __name__ == "__main__":
    # Ensure you have a .env_local file with MONGO_URI and GEMINI_API_KEY
    if not all(os.getenv(k) for k in ["MONGO_URI", "GEMINI_API_KEY"]):
        print("ERROR: Please set MONGO_URI and GEMINI_API_KEY in your ~/resources/.env_local file.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        asyncio.run(main())
