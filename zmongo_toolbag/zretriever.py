"""
ZRetriever – SafeResult-compatible retriever for non-async ZMongo repositories.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ConfigDict, Field

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import BaseRetriever, Document  # type: ignore

from zmongo_toolbag.safe_result import SafeResult
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import (
    ZEmbedder,
    EMBEDDING_STYLE_RETRIEVAL_QUERY,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
)
from zmongo_toolbag.unified_vector_search import LocalVectorSearch

__all__ = ["ZRetriever", "__version__"]
__version__ = "2.3.0"

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "zretriever_default_kb"
_DEFAULT_EMBED_FIELD = "embeddings"
_DEFAULT_CONTENT_FIELD = "text"

load_dotenv(Path.home() / ".resources" / ".env")
load_dotenv(Path.home() / ".resources" / ".secrets")


class ZRetriever(BaseRetriever):
    """
    LangChain-compatible retriever using SafeResult-based ZMongo + ZEmbedder.
    Works synchronously or asynchronously.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    repository: ZMongo = Field(...)
    embedder: ZEmbedder = Field(...)
    vector_searcher: LocalVectorSearch = Field(...)
    collection_name: str = Field(default=_DEFAULT_COLLECTION)
    embedding_field: str = Field(default=_DEFAULT_EMBED_FIELD)
    content_field: str = Field(default=_DEFAULT_CONTENT_FIELD)
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.8)
    query_embedding_style: str = Field(default=EMBEDDING_STYLE_RETRIEVAL_QUERY)

    # ------------------------------------------------------------------
    # Core retrieval logic
    # ------------------------------------------------------------------
    def get_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None,
            **kwargs: Any,
    ) -> List[Document]:

        """Main entrypoint for LangChain synchronous compatibility."""
        logger.debug("ZRetriever starting query: %s", query)

        # 1. Embed the query
        emb_res = self.embedder.get_embedding_sync(
            query,
            embedding_style=self.query_embedding_style,
            as_safe_result=True,
        )
        if not emb_res.success:
            logger.error("Query embedding failed: %s", emb_res.error)
            return []

        vectors = emb_res.data.get("vectors")
        if not vectors:
            logger.warning("Embedding returned no vectors for query.")
            return []

        qvec = np.array(vectors[0], dtype=float)

        # 2. Perform vector search
        search_res = self.vector_searcher.search(qvec, top_k=self.top_k)
        if hasattr(search_res, "success") and not search_res.success:
            logger.error("Vector search failed: %s", search_res.error)
            return []

        hits = search_res.data if isinstance(search_res, SafeResult) else search_res
        if not hits:
            logger.info("No hits found.")
            return []

        # 3. Format and filter
        return self._format_and_filter_results(hits)

    # ------------------------------------------------------------------
    # Async version for LangChain compatibility
    # ------------------------------------------------------------------
    async def aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None,
            **kwargs: Any,
    ) -> List[Document]:
        return self.get_relevant_documents(query)

    async def ainvoke(self, query: str, **kwargs: Any) -> List[Document]:
        return self.get_relevant_documents(query)

    def invoke(self, query: str, **kwargs: Any) -> List[Document]:
        return self.get_relevant_documents(query)

    # ------------------------------------------------------------------
    # Formatting and threshold filtering
    # ------------------------------------------------------------------
    def _format_and_filter_results(self, hits: List[Dict[str, Any]]) -> List[Document]:
        """Convert SafeResult hits to LangChain Document objects."""
        docs: List[Document] = []
        for i, hit in enumerate(hits):
            score = hit.get("retrieval_score")
            if not isinstance(score, (int, float)):
                logger.debug(f"Skipping hit {i} – invalid score {score}")
                continue
            if score < self.similarity_threshold:
                continue

            doc_data = hit.get("document", hit)
            content = doc_data.get(self.content_field, "")
            metadata = {
                k: v
                for k, v in doc_data.items()
                if k not in [self.content_field, self.embedding_field]
            }
            metadata["retrieval_score"] = float(score)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs


# ----------------------------------------------------------------------
# Self-test demo using real data
# ----------------------------------------------------------------------
def _demo():
    """Integration demo showing retrieval from real SafeResult-based ZMongo."""
    import asyncio
    from bson import ObjectId

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting ZRetriever real-data demo...")

    db_client = ZMongo()
    embedder = ZEmbedder(repository=db_client)
    collection = _DEFAULT_COLLECTION

    db_client.delete_all_documents(collection)
    knowledge = [
        {"_id": ObjectId(), "topic": "Biology", "text": "Mitochondria generate energy in the cell."},
        {"_id": ObjectId(), "topic": "Astronomy", "text": "Jupiter is the largest planet."},
        {"_id": ObjectId(), "topic": "History", "text": "The Roman Empire shaped Western civilization."},
    ]

    for doc in knowledge:
        db_client.insert_one(collection, doc)
        emb_res = embedder.get_embedding_sync(
            text=doc["text"],
            collection=collection,
            document_id=doc["_id"],
            embedding_field=_DEFAULT_EMBED_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        )
        if not emb_res.success:
            logger.error("Doc embedding failed: %s", emb_res.error)

    vector_search = LocalVectorSearch(
        repository=db_client,
        collection=collection,
        embedding_field=_DEFAULT_EMBED_FIELD,
    )

    retriever = ZRetriever(
        repository=db_client,
        embedder=embedder,
        vector_searcher=vector_search,
        collection_name=collection,
        similarity_threshold=0.75,
        top_k=2,
    )

    query = "Which organelle provides energy in the cell?"
    results = retriever.get_relevant_documents(query)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

    embedder.close()


if __name__ == "__main__":
    _demo()
