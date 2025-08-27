"""
ZRetriever — LangChain-Compatible Retriever over ZMongo + LocalVectorSearch
===========================================================================

`ZRetriever` implements LangChain's :class:`~langchain_core.retrievers.BaseRetriever`
interface and wires together:

* **ZMongo** — async repository helper for MongoDB
* **ZEmbedder** — deterministic, async embedder with chunking and styles
* **LocalVectorSearch** — in-memory cosine search over chunked embeddings

It produces :class:`langchain.schema.Document` (or `langchain_core.documents.Document`)
results with page content and normalized metadata.

Highlights
----------
- **Pydantic v2** compatible (`model_config = ConfigDict(arbitrary_types_allowed=True)`).
- **Style-aware querying**: choose `query_embedding_style` (typically
  ``EMBEDDING_STYLE_RETRIEVAL_QUERY``) to pair with documents embedded using
  ``EMBEDDING_STYLE_RETRIEVAL_DOCUMENT``.
- **Thresholding**: `similarity_threshold` filters weak matches.
- **Flexible hit-shape handling**: supports both legacy `{"document": {...}}`
  and new `{"text","metadata","doc_id","retrieval_score"}` shapes returned by
  `LocalVectorSearch`.
- **Developer helpers**: convenience constructors (`from_components`), sync/async
  wrappers (`retrieve`/`aretrieve`, `invoke`/`ainvoke`), and readable `__repr__`.

Quick Start
-----------
.. code-block:: python

    import asyncio
    from bson import ObjectId
    from zmongo_toolbag.zmongo import ZMongo
    from zmongo_toolbag.zembedder import (
        ZEmbedder,
        EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        EMBEDDING_STYLE_RETRIEVAL_QUERY,
    )
    from zmongo_toolbag.unified_vector_search import LocalVectorSearch
    from zmongo_toolbag.zretriever import ZRetriever

    COLL = "retriever_demo_knowledge_base"
    EMBED_FIELD = "embeddings"

    async def main():
        repo = ZMongo()
        embedder = ZEmbedder(repository=repo)

        # Insert & embed 3 docs (stored with RETRIEVAL_DOCUMENT)
        kb = [
            {"_id": ObjectId(), "topic": "Astronomy", "text": "Jupiter is the fifth planet from the Sun."},
            {"_id": ObjectId(), "topic": "Biology",   "text": "Mitochondria are called the powerhouse of the cell."},
            {"_id": ObjectId(), "topic": "History",   "text": "The Roman Empire was highly influential."},
        ]
        await repo.delete_documents(COLL, {})
        await repo.insert_documents(COLL, kb)
        for d in kb:
            await embedder.embed_and_store(
                collection=COLL,
                document_id=d["_id"],
                text=d["text"],
                embedding_field=EMBED_FIELD,
                embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            )

        # Build the local vector index (over chunked embeddings)
        lvs = LocalVectorSearch(
            repository=repo,
            collection=COLL,
            embedding_field=EMBED_FIELD,
            chunked_embeddings=True,
            exact_rescore=True,
        )

        # Create retriever (queries use RETRIEVAL_QUERY, compatible with stored RD)
        retriever = ZRetriever(
            repository=repo,
            embedder=embedder,
            vector_searcher=lvs,
            collection_name=COLL,
            embedding_field=EMBED_FIELD,
            similarity_threshold=0.10,
            top_k=3,
            query_embedding_style=EMBEDDING_STYLE_RETRIEVAL_QUERY,
        )

        docs = await retriever.aretrieve("What is the powerhouse of the cell?")
        for d in docs:
            print(d.page_content)
            print(d.metadata)

        repo.close()
        embedder.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId

# Prefer modern LangChain core types; fall back for older versions
try:  # pragma: no cover
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import BaseRetriever, Document  # type: ignore

from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from pydantic import ConfigDict, Field

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder import ZEmbedder, EMBEDDING_STYLE_RETRIEVAL_QUERY
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.data_processing import DataProcessor

__all__ = ["ZRetriever", "__version__"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Default knobs for the embedded demo
_DEFAULT_COLLECTION = "retriever_demo_knowledge_base"
_DEFAULT_EMBED_FIELD = "embeddings"
_DEFAULT_QUERY_STYLE = EMBEDDING_STYLE_RETRIEVAL_QUERY
_DEFAULT_CONTENT_FIELD = "text"


class ZRetriever(BaseRetriever):
    """LangChain-compatible retriever using `LocalVectorSearch` over `ZMongo` data.

    This class turns a natural-language **query** into embeddings using
    :class:`ZEmbedder` and runs an in-memory cosine search via
    :class:`LocalVectorSearch` against a MongoDB collection, returning a
    list of :class:`~langchain_core.documents.Document`.

    Parameters
    ----------
    repository : ZMongo
        Async MongoDB repository helper.
    embedder : ZEmbedder
        Text embedder used to create **query** embeddings.
    vector_searcher : LocalVectorSearch
        In-memory cosine searcher over chunked embeddings.
    collection_name : str
        MongoDB collection to search.
    embedding_field : str, optional
        Field within documents containing embeddings (default ``"embeddings"``).
    content_field : str, optional
        Field from which to pull human-readable content for `Document.page_content`.
    top_k : int, optional
        Maximum number of documents to return.
    similarity_threshold : float, optional
        Minimum retrieval score for a hit to be returned.
    query_embedding_style : str, optional
        Embedding style for **queries**. For compatibility with stored vectors,
        use ``EMBEDDING_STYLE_RETRIEVAL_QUERY`` when documents are stored with
        ``EMBEDDING_STYLE_RETRIEVAL_DOCUMENT``.

    Notes
    -----
    * **Pydantic v2**: `model_config = ConfigDict(arbitrary_types_allowed=True)` allows
      passing service objects without Pydantic trying to validate them.
    * **Hit shape compatibility**: :meth:`_format_results` accepts both the
      legacy `{ "document": {...} }` items and the newer
      `{ "text", "metadata", "doc_id", "retrieval_score" }` shape.
    """

    # Pydantic v2 model config — permit arbitrary service objects
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Use Any for service handles to avoid pydantic-core instance checks
    repository: Any = Field(...)
    embedder: Any = Field(...)
    vector_searcher: Any = Field(...)

    collection_name: str = Field(...)
    embedding_field: str = Field(default=_DEFAULT_EMBED_FIELD)
    content_field: str = Field(default=_DEFAULT_CONTENT_FIELD)
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.0)
    query_embedding_style: str = Field(default=_DEFAULT_QUERY_STYLE)

    # ---------------------------------------------------------------------
    # Lifecycle & representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ZRetriever(collection='{self.collection_name}', field='{self.embedding_field}', "
            f"top_k={self.top_k}, threshold={self.similarity_threshold})"
        )

    @property
    def is_configured(self) -> bool:
        """Return ``True`` if the retriever has all required components.

        This is a convenience check for app boot-time validation.
        """
        return all([self.repository, self.embedder, self.vector_searcher, self.collection_name])

    def validate_configuration(self) -> None:
        """Raise ``RuntimeError`` if mandatory components are missing.

        Useful for early failure in service startup.
        """
        if not self.is_configured:
            raise RuntimeError("ZRetriever is not fully configured (missing components)")

    # ---------------------------------------------------------------------
    # LangChain hooks — sync wrapper delegates to async implementation
    # ---------------------------------------------------------------------
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Sync wrapper that delegates to :meth:`_aget_relevant_documents`.

        In non-async contexts, LangChain may call the sync API.
        """
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Compute a query embedding and return the top-``k`` documents.

        Steps
        -----
        1. Build a **query** embedding (style: :pyattr:`query_embedding_style`).
        2. Run vector search (:pyattr:`top_k`) with :class:`LocalVectorSearch`.
        3. Format results into LangChain :class:`Document` objects with metadata.
        """
        embeddings = await self.embedder.get_embedding(query, embedding_style=self.query_embedding_style)
        if not embeddings:
            return []
        query_embedding = embeddings[0]

        search_result = await self.vector_searcher.search(query_embedding, top_k=self.top_k)
        if not search_result.success:
            raise RuntimeError(f"Local vector search failed: {search_result.error}")
        return self._format_results(search_result.data)

    # ---------------------------------------------------------------------
    # Convenience wrappers for app developers (outside LangChain)
    # ---------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Document]:
        """Synchronous retrieval helper.

        Equivalent to :meth:`invoke`. Useful in simple scripts and CLIs.
        """
        return self.invoke(query)

    async def aretrieve(self, query: str) -> List[Document]:
        """Asynchronous retrieval helper.

        Equivalent to :meth:`ainvoke`. Mirrors LangChain's async pipeline.
        """
        return await self.ainvoke(query)

    # Runnable-style helpers (keep signatures friendly for non-LC users)
    def invoke(self, query: str, **_: Any) -> List[Document]:
        """Run retrieval synchronously.

        Parameters
        ----------
        query : str
            Natural-language query.

        Returns
        -------
        list[Document]
            Top-``k`` results meeting :pyattr:`similarity_threshold`.
        """
        return asyncio.run(self.ainvoke(query))

    async def ainvoke(self, query: str, **_: Any) -> List[Document]:
        """Run retrieval asynchronously (preferred in async apps).

        This uses a **no-op callback manager** to satisfy the LangChain
        interface without firing any callbacks.
        """
        # Prefer a built-in no-op manager when available (langchain-core >= 0.2)
        run_manager: Any
        try:  # pragma: no cover — version-dependent
            run_manager = AsyncCallbackManagerForRetrieverRun.get_noop_manager()  # type: ignore[attr-defined]
        except Exception:  # Fallback for older versions
            class _NopManager:  # minimal shim; _aget_relevant_documents ignores it
                pass
            run_manager = _NopManager()
        these_documents = await self._aget_relevant_documents(query, run_manager=run_manager)
        return these_documents

    # ---------------------------------------------------------------------
    # Result shaping
    # ---------------------------------------------------------------------
    def _format_results(self, items: List[Dict[str, Any]]) -> List[Document]:
        """Normalize search hits into LangChain :class:`Document` objects.

        Accepts either:
        - **Old shape**: ``{"document": {...}, "retrieval_score": float}``
        - **New shape**: ``{"text": str, "metadata": {...}, "doc_id": str, "retrieval_score": float}``

        Returns
        -------
        list[Document]
            Documents with `page_content` and JSON-safe `metadata` (no embeddings).
        """
        final_docs: List[Document] = []

        for item in items:
            score = float(item.get("retrieval_score", 0.01))
            if score < self.similarity_threshold:
                continue

            # Newer shape from LocalVectorSearch
            if "text" in item or "metadata" in item:
                content = item.get("text", "")
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                metadata = dict(item.get("metadata") or {})
                if "_id" in metadata and isinstance(metadata["_id"], ObjectId):
                    metadata["_id"] = str(metadata["_id"])
                if "doc_id" in item and "doc_id" not in metadata:
                    metadata["doc_id"] = item["doc_id"]
                metadata["retrieval_score"] = score
                # ensure we don't leak embeddings if present
                if self.embedding_field in metadata:
                    metadata.pop(self.embedding_field, None)
                final_docs.append(Document(page_content=content, metadata=metadata))
                continue

            # Legacy shape
            if "document" in item and isinstance(item["document"], dict):
                doc_data: Dict[str, Any] = item["document"]
                content = DataProcessor.get_value(doc_data, self.content_field)
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                if "_id" in doc_data and isinstance(doc_data["_id"], ObjectId):
                    doc_data["_id"] = str(doc_data["_id"])
                metadata = {k: v for k, v in doc_data.items() if k != self.embedding_field}
                metadata["retrieval_score"] = score
                final_docs.append(Document(page_content=content, metadata=metadata))
                continue

            # Unknown/unsupported shape — be lenient
            content = str(item.get("text", "")) if item.get("text") is not None else ""
            metadata = dict(item.get("metadata") or {})
            metadata["retrieval_score"] = score
            final_docs.append(Document(page_content=content, metadata=metadata))

        return final_docs

    # ---------------------------------------------------------------------
    # Convenience constructors
    # ---------------------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        *,
        repository: Optional[ZMongo] = None,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_field: str = _DEFAULT_EMBED_FIELD,
        embedder: Optional[ZEmbedder] = None,
        vector_searcher: Optional[LocalVectorSearch] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        query_embedding_style: str = _DEFAULT_QUERY_STYLE,
        content_field: str = _DEFAULT_CONTENT_FIELD,
    ) -> "ZRetriever":
        """Instantiate a :class:`ZRetriever` with sensible defaults.

        Any missing component (repo/embedder/searcher) is created on the fly.
        This is a convenience factory for app wiring and tests.
        """
        repo = repository or ZMongo()
        emb = embedder or ZEmbedder(repository=repo)
        lvs = vector_searcher or LocalVectorSearch(
            repository=repo,
            collection=collection_name,
            embedding_field=embedding_field,
            chunked_embeddings=True,
            exact_rescore=True,
        )
        return cls(
            repository=repo,
            embedder=emb,
            vector_searcher=lvs,
            collection_name=collection_name,
            embedding_field=embedding_field,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            query_embedding_style=query_embedding_style,
            content_field=content_field,
        )


# ---------------------------------------------------------------------
# Demo entrypoint (manual testing)
# ---------------------------------------------------------------------
async def _demo() -> None:  # pragma: no cover — demo only
    """Run a small end‑to‑end demonstration of the retriever."""
    from bson import ObjectId
    from zmongo_toolbag.zembedder import EMBEDDING_STYLE_RETRIEVAL_DOCUMENT

    repo = ZMongo()
    embedder = ZEmbedder(repository=repo)
    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=_DEFAULT_COLLECTION,
        embedding_field=_DEFAULT_EMBED_FIELD,
        chunked_embeddings=True,
        exact_rescore=True,
    )

    retriever = ZRetriever(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=_DEFAULT_COLLECTION,
        embedding_field=_DEFAULT_EMBED_FIELD,
        similarity_threshold=0.10,
        top_k=3,
        query_embedding_style=_DEFAULT_QUERY_STYLE,
    )

    print(f"--- Setting up the '{_DEFAULT_COLLECTION}' collection ---")
    await repo.delete_documents(_DEFAULT_COLLECTION, {})
    knowledge_base = [
        {
            "_id": ObjectId(),
            "topic": "Astronomy",
            "text": "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
        },
        {
            "_id": ObjectId(),
            "topic": "Biology",
            "text": "Mitochondria are organelles often called the powerhouse of the cell.",
            },
        {
            "_id": ObjectId(),
            "topic": "History",
            "text": "The Roman Empire was one of the most influential civilizations in world history.",
        },
    ]

    ins = await repo.insert_documents(_DEFAULT_COLLECTION, knowledge_base)
    assert ins.success, ins.error
    # Persist embeddings (RD) for each doc
    for doc in knowledge_base:
        e = await embedder.embed_and_store(
            collection=_DEFAULT_COLLECTION,
            document_id=doc["_id"],
            text=doc["text"],
            embedding_field=_DEFAULT_EMBED_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        )
        assert e.success, e.error

    print("Successfully inserted and embedded", len(knowledge_base), "facts.\n")

    # Give the index a moment for TTL refresh paths in tests
    await asyncio.sleep(0.5)

    query = "What is the powerhouse of the cell?"
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

    repo.close()
    embedder.close()


if __name__ == "__main__":  # pragma: no cover — demo only
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(_demo())
