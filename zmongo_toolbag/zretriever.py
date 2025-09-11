from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# LangChain core imports
try:  # pragma: no cover
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
except Exception:  # pragma: no cover
    from langchain.schema import BaseRetriever, Document  # type: ignore
    from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun  # type: ignore

from pydantic import ConfigDict, Field

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    EMBEDDING_STYLE_RETRIEVAL_QUERY,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
    CHUNK_STYLE_FIXED,
)
from zmongo_toolbag.unified_vector_search import LocalVectorSearch

__all__ = ["ZRetriever", "__version__"]
__version__ = "2.1.0"  # Version bump for final polish

logger = logging.getLogger(__name__)

# --- Default configuration constants ---
_DEFAULT_COLLECTION = "zretriever_default_kb"
_DEFAULT_EMBED_FIELD = "embeddings"
_DEFAULT_CONTENT_FIELD = "text"

load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")


class ZRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that orchestrates ZMongo, ZEmbedder,
    and a vector searcher to perform robust, real-time retrieval.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Core components
    repository: ZMongo = Field(...)
    embedder: ZEmbedder = Field(...)
    vector_searcher: LocalVectorSearch = Field(...)

    # Retrieval settings
    collection_name: str = Field(...)
    embedding_field: str = Field(default=_DEFAULT_EMBED_FIELD)
    content_field: str = Field(default=_DEFAULT_CONTENT_FIELD)
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.8)
    query_embedding_style: str = Field(default=EMBEDDING_STYLE_RETRIEVAL_QUERY)

    def __repr__(self) -> str:
        return (
            f"ZRetriever(collection='{self.collection_name}', "
            f"top_k={self.top_k}, threshold={self.similarity_threshold})"
        )

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Core asynchronous retrieval logic for LangChain."""
        logger.debug(f"Received query: '{query}'")

        query_embeddings = await self.embedder.get_embedding(
            query, embedding_style=self.query_embedding_style, as_safe_result=False
        )
        if not query_embeddings or not query_embeddings[0]:
            logger.warning("Query embedding failed or returned no vectors.")
            return []
        query_embedding = query_embeddings[0]

        search_result = await self.vector_searcher.search(query_embedding, top_k=self.top_k)
        if not search_result.success:
            logger.error("Vector search failed: %s", search_result.error)
            return []

        raw_hits = search_result.data or []
        logger.info("Vector search returned %d raw hits before filtering.", len(raw_hits))

        return self._format_and_filter_results(raw_hits)

    def _format_and_filter_results(self, items: List[Dict[str, Any]]) -> List[Document]:
        """
        Transforms raw search hits into Document objects, filtering by threshold
        and cleaning up metadata.
        """
        docs: List[Document] = []
        for i, hit in enumerate(items):
            score = hit.get("retrieval_score")
            if not isinstance(score, (int, float)):
                logger.warning(f"Hit #{i + 1} is missing a valid score. Skipping.")
                continue

            if score >= self.similarity_threshold:
                logger.info(f"KEEP Hit #{i + 1} — score {score:.4f} >= threshold {self.similarity_threshold}")

                doc_data = hit.get("document", hit)
                content = doc_data.get(self.content_field, "")

                metadata = {
                    k: v for k, v in doc_data.items()
                    if k not in [self.content_field, self.embedding_field]
                    # Exclude content and the huge embedding vector
                }
                metadata["retrieval_score"] = score

                docs.append(Document(page_content=content, metadata=metadata))
            else:
                logger.info(f"DISCARD Hit #{i + 1} — score {score:.4f} < threshold {self.similarity_threshold}")

        logger.info("Finished processing hits. Returning %d documents after filtering.", len(docs))
        return docs

    # --- LangChain compatibility and convenience wrappers ---
    def _get_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def ainvoke(self, query: str, **kwargs: Any) -> List[Document]:
        return await self._aget_relevant_documents(query,
                                                   run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager())

    def invoke(self, query: str, **kwargs: Any) -> List[Document]:
        return asyncio.run(self.ainvoke(query))


async def _demo() -> None:
    """A self-contained demonstration of the perfected ZRetriever."""
    from bson import ObjectId

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting ZRetriever demo...")

    db_client = ZMongo()
    embedder = ZEmbedder(repository=db_client, n_ctx=2048)

    print(f"--- Setting up the '{_DEFAULT_COLLECTION}' collection ---")
    await db_client.db[_DEFAULT_COLLECTION].drop()

    knowledge_base = [
        {"_id": ObjectId(), "topic": "Biology",
         "text": "Mitochondria are organelles often called the powerhouse of the cell."},
        {"_id": ObjectId(), "topic": "Astronomy",
         "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System."},
        {"_id": ObjectId(), "topic": "History",
         "text": "The Roman Empire was one of the most influential civilizations in world history."},
    ]
    await db_client.insert_documents(_DEFAULT_COLLECTION, knowledge_base)

    for doc in knowledge_base:
        embed_result = await embedder.get_embedding(
            collection=_DEFAULT_COLLECTION,
            document_id=doc["_id"],
            text=doc["text"],
            embedding_field=_DEFAULT_EMBED_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            chunk_style=CHUNK_STYLE_FIXED
        )
        if not embed_result.success:
            logger.error(f"Failed to embed document {doc['_id']}: {embed_result.error}")
            return
    print(f"Successfully inserted and embedded {len(knowledge_base)} facts.\n")

    vector_searcher = LocalVectorSearch(
        repository=db_client,
        collection=_DEFAULT_COLLECTION,
        embedding_field=_DEFAULT_EMBED_FIELD,
        chunked_embeddings=True
    )
    retriever = ZRetriever(
        repository=db_client,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=_DEFAULT_COLLECTION,
        similarity_threshold=0.80,
        top_k=3,
    )

    query = "What is the powerhouse of the cell?"
    print(f"--- Invoking retriever with query: '{query}' ---")
    results = await retriever.ainvoke(query)

    print(f"\nFound {len(results)} relevant document(s):")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")  # This will now be clean

    embedder.close()


if __name__ == "__main__":
    asyncio.run(_demo())

