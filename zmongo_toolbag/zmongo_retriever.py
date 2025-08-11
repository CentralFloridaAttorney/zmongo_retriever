import logging
import asyncio
from typing import List

import numpy as np
from langchain.schema import BaseRetriever, Document
from langchain_core.runnables import RunnableConfig
from pymongo.errors import OperationFailure

# Assuming these are in the same directory or a package
from zmongo_toolbag.zmongo_atlas import ZMongoAtlas
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.data_processing import DataProcessor

logger = logging.getLogger(__name__)


class ZMongoRetriever(BaseRetriever):
    """
    A high-performance, LangChain-compliant retriever that uses a two-stage
    search-then-filter process for optimal relevance.
    """
    repository: ZMongoAtlas
    embedder: ZMongoEmbedder
    collection_name: str
    embedding_field: str = "embeddings"
    content_field: str = "text"
    top_k: int = 10
    vector_search_index_name: str = "vector_index"
    similarity_threshold: float = 0.60

    # --- FIX ---
    # This configuration tells the Pydantic model to allow custom class types
    # for the 'repository' and 'embedder' fields, resolving the validation error.
    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input: str, config: RunnableConfig | None = None) -> List[Document]:
        return asyncio.run(self.ainvoke(input, config=config))

    async def ainvoke(self, input: str, config: RunnableConfig | None = None) -> List[Document]:
        return await self._get_relevant_documents(input)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        vec1, vec2 = np.asarray(v1, dtype=np.float32), np.asarray(v2, dtype=np.float32)
        if vec1.shape != vec2.shape or vec1.size == 0: return 0.0
        norm_v1, norm_v2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    def _filter_and_format_results(self, items: List[dict]) -> List[Document]:
        """
        Filters results by the similarity_threshold and formats them into
        LangChain Document objects.
        """
        final_docs = []
        for item in items:
            score = item.get("retrieval_score", 0.0)
            if score >= self.similarity_threshold:
                doc_data = item.get("document", item)

                # Extract page content using the dot-separated key
                content = DataProcessor.get_value(doc_data, self.content_field)
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""

                # The rest of the document becomes the metadata
                metadata = {k: v for k, v in doc_data.items() if k != self.embedding_field}
                metadata['retrieval_score'] = score

                final_docs.append(Document(page_content=content, metadata=metadata))
        return final_docs

    async def _atlas_search(self, query_embedding: List[float]) -> List[Document]:
        """Performs a search using the optimized Atlas Vector Search."""
        res = await self.repository.vector_search(
            self.collection_name, query_embedding, self.vector_search_index_name, self.embedding_field, self.top_k
        )
        if not res.success:
            # Ensure the original exception is raised if it exists
            if res.original():
                raise res.original()
            raise RuntimeError(res.error)
        return self._filter_and_format_results(res.data)

    async def _manual_search(self, query_embedding: List[float]) -> List[Document]:
        """Performs a manual similarity search as a fallback."""
        logger.warning("Falling back to manual similarity search.")
        res = await self.repository.find_documents(self.collection_name, {self.embedding_field: {"$exists": True}})
        if not res.success or not res.data: return []

        scored_docs = []
        for doc in res.data:
            # Handle cases where the embedding field might not be a list of lists
            embeddings_list = doc.get(self.embedding_field, [])
            if not isinstance(embeddings_list, list) or not all(isinstance(e, list) for e in embeddings_list):
                continue

            max_sim = max(
                (self._cosine_similarity(query_embedding, chunk) for chunk in embeddings_list),
                default=-1.0
            )
            if max_sim >= self.similarity_threshold:
                scored_docs.append({"retrieval_score": max_sim, "document": doc})

        scored_docs.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return self._filter_and_format_results(scored_docs)

    async def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Orchestrates finding relevant documents, trying Atlas Search first and
        falling back to a manual search if the environment does not support it.
        """
        embeddings = await self.embedder.embed_text(query)
        if not embeddings: return []

        query_embedding = embeddings[0]
        try:
            return await self._atlas_search(query_embedding)
        except OperationFailure as e:
            if e.code == 31082 or 'SearchNotEnabled' in str(e):
                return await self._manual_search(query_embedding)
            else:
                logger.error(f"A MongoDB OperationFailure occurred: {e}")
                raise
