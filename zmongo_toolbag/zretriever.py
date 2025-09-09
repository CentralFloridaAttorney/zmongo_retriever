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
- **Pydantic v2** compatible (via `model_config = ConfigDict(arbitrary_types_allowed=True)`).
- **Style-aware querying**: choose `query_embedding_style` (typically
  ``EMBEDDING_STYLE_RETRIEVAL_QUERY``) to pair with documents embedded using
  ``EMBEDDING_STYLE_RETRIEVAL_DOCUMENT``.
- **Thresholding**: `similarity_threshold` filters weak matches based on **cosine similarity**.
- **Flexible hit-shape handling**: supports both legacy `{"document": {...}}`
  and new `{"text","metadata","doc_id","retrieval_score"}` shapes returned by
  `LocalVectorSearch`.
- **Developer helpers**: convenience constructors (`from_components`), sync/async
  wrappers (`retrieve`/`aretrieve`, `invoke`/`ainvoke`), and readable `__repr__`.

Scoring Semantics (Important)
-----------------------------
`LocalVectorSearch` is expected to return `retrieval_score` as **raw cosine similarity**
in the range **[-1.0, 1.0]**, where **higher is better**. The retriever applies the
`similarity_threshold` as `score >= threshold`.

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
            similarity_threshold=0.80,
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from bson import ObjectId
from dotenv import load_dotenv

# Prefer modern LangChain core types; fall back for older versions
try:  # pragma: no cover
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import BaseRetriever, Document  # type: ignore

from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from pydantic import ConfigDict, Field

from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zembedder_gemini import (
    ZEmbedder,
    EMBEDDING_STYLE_RETRIEVAL_QUERY,
    EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
)
from zmongo_toolbag.unified_vector_search import LocalVectorSearch
from zmongo_toolbag.data_processing import SafeResult

__all__ = ["ZRetriever", "__version__"]
__version__ = "1.0.0"

# Setup detailed logging
logger = logging.getLogger(__name__)

# Default knobs for the embedded demo
_DEFAULT_COLLECTION = "test"
_DEFAULT_EMBED_FIELD = "embeddings"
_DEFAULT_QUERY_STYLE = EMBEDDING_STYLE_RETRIEVAL_QUERY
_DEFAULT_EMBEDDING_STYLE = EMBEDDING_STYLE_RETRIEVAL_DOCUMENT
_DEFAULT_CONTENT_FIELD = "text"

# Load optional local env files (no-op if absent)
load_dotenv(Path.home() / ".resources" / ".env_zmongo_retriever")
load_dotenv(Path.home() / ".resources" / ".secrets")


class ZRetriever(BaseRetriever):
    """
    LangChain-compatible retriever using `LocalVectorSearch` over `ZMongo` data.

    This class turns a natural-language **query** into embeddings using
    :class:`ZEmbedder` and runs an in-memory cosine search via
    :class:`LocalVectorSearch` against a MongoDB collection, returning a
    list of :class:`~langchain_core.documents.Document`.

    Parameters
    ----------
    repository : Any
        A `ZMongo` repository instance (or repository-like object) providing
        async CRUD operations over a MongoDB collection.
    embedder : Any
        A `ZEmbedder` instance capable of producing embeddings for text,
        with support for multiple `embedding_style` values.
    vector_searcher : Any
        A `LocalVectorSearch` instance tied to the same collection/embeddings.
        Expected to return `retrieval_score` as **cosine similarity** in [-1, 1].
    collection_name : str
        Name of the MongoDB collection that stores documents and their embeddings.
    embedding_field : str, default "embeddings"
        Field name under which embeddings are stored on each document.
    content_field : str, default "text"
        Field name on the document to use as the `page_content` of the LangChain
        `Document`. Falls back to `"text"` if the specific field is absent.
    top_k : int, default 10
        Number of top results to retrieve from the vector index before filtering.
    similarity_threshold : float, default 0.8
        Minimum cosine similarity required for a hit to be kept. Using raw cosine
        in [-1, 1], **higher is better**.
    query_embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_QUERY
        Embedding style for queries. Pair `RETRIEVAL_QUERY` for queries with
        `RETRIEVAL_DOCUMENT` for indexed docs.

    Notes
    -----
    - `ZRetriever` assumes your `LocalVectorSearch.search()` returns items that
      include a `retrieval_score` key representing cosine similarity. If your
      searcher returns a different scale, ensure you map it back to cosine.
    - The class exposes both LangChain retriever APIs and simple `invoke/ainvoke`
      helpers for "Runnable"-style usage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Core components
    repository: Any = Field(...)
    embedder: Any = Field(...)
    vector_searcher: Any = Field(...)

    # Retrieval settings
    collection_name: str = Field(...)
    embedding_field: str = Field(default=_DEFAULT_EMBED_FIELD)
    content_field: str = Field(default=_DEFAULT_CONTENT_FIELD)
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.8)
    query_embedding_style: str = Field(default=_DEFAULT_QUERY_STYLE)

    def __repr__(self) -> str:
        """Return a concise, developer-friendly representation of this retriever."""
        return (
            f"ZRetriever(collection='{self.collection_name}', field='{self.embedding_field}', "
            f"top_k={self.top_k}, threshold={self.similarity_threshold})"
        )

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        This helper is currently used only for debugging/explanations; the
        actual scoring comes from `LocalVectorSearch`. Kept for completeness.

        Parameters
        ----------
        v1 : numpy.ndarray | Sequence[float]
            First vector.
        v2 : numpy.ndarray | Sequence[float]
            Second vector.

        Returns
        -------
        float
            Cosine similarity in [-1.0, 1.0]. Returns 0.0 if any vector has zero norm.
        """
        if not isinstance(v1, np.ndarray):
            v1 = np.array(v1)
        if not isinstance(v2, np.ndarray):
            v2 = np.array(v2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Implementation of LangChain's async retrieval method.

        Steps
        -----
        1. **Embed query** using the configured `query_embedding_style`.
        2. **Search** with `LocalVectorSearch.search(query_embedding, top_k=self.top_k)`.
        3. **Filter & format** results whose `retrieval_score >= similarity_threshold`.
        4. **Return** a list of LangChain `Document` objects.

        Parameters
        ----------
        query : str
            Natural-language query.
        run_manager : AsyncCallbackManagerForRetrieverRun
            LangChain callback manager provided by the framework.

        Returns
        -------
        List[Document]
            Zero or more `Document` instances meeting the threshold.

        Raises
        ------
        RuntimeError
            If vector search fails (e.g., returns an error SafeResult).
        """
        logger.debug(f"Received query for retrieval: '{query}'")

        # 1) Embed the query
        logger.debug(f"Embedding query with style: '{self.query_embedding_style}'")
        embeddings = await self.embedder.get_embedding(
            query, embedding_style=self.query_embedding_style
        )
        if not embeddings:
            logger.warning("Query embedding returned no results. Cannot perform search.")
            return []
        query_embedding = embeddings[0]
        logger.debug(
            "Successfully generated query embedding. Vector dim: %d, start: %s",
            len(query_embedding),
            np.array(query_embedding[:5]).round(4),
        )

        # 2) Run vector search
        logger.debug(
            "Performing vector search in '%s' for top %d results.",
            self.collection_name,
            self.top_k,
        )
        search_result = await self.vector_searcher.search(query_embedding, top_k=self.top_k)

        if not search_result.success:
            logger.error("Vector search failed: %s", search_result.error)
            raise RuntimeError(f"Local vector search failed: {search_result.error}")

        raw_hits = search_result.data or []
        logger.info("Vector search returned %d raw hits before filtering.", len(raw_hits))

        # 3) Format & filter
        return self._format_and_filter_results(raw_hits, query_embedding)

    @staticmethod
    def _docs_to_payload(docs: List["Document"]) -> List[Dict[str, Any]]:
        """
        Convert a list of LangChain `Document` objects into a JSON-serializable list.

        Parameters
        ----------
        docs : List[Document]
            Documents to serialize.

        Returns
        -------
        List[Dict[str, Any]]
            Each item contains `page_content` and `metadata` keys.
        """
        out: List[Dict[str, Any]] = []
        for d in docs:
            out.append({"page_content": d.page_content, "metadata": dict(d.metadata or {})})
        return out

    def invoke_sr(self, query: str, **_: Any) -> SafeResult:
        """
        Synchronous, SafeResult-wrapped entry point.

        This is a convenience for systems that prefer `SafeResult` error handling
        and cannot or do not want to `await` the async API.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        SafeResult
            `SafeResult.ok([...])` with serialized docs on success, or
            `SafeResult.fail(...)` on error. The `.data` field is a list of
            dicts as produced by `_docs_to_payload`.
        """
        try:
            return asyncio.run(self.ainvoke_sr(query))
        except Exception as e:
            logger.exception("Error during synchronous invoke_sr execution.")
            return SafeResult.fail(str(e), exc=e)

    async def ainvoke_sr(self, query: str, **_: Any) -> SafeResult:
        """
        Async, SafeResult-wrapped entry point.

        Useful when you want structured error handling plus extra metadata
        (e.g., `page_content_field`, `source_collection`) added to the payload.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        SafeResult
            `SafeResult.ok([...])` with enriched metadata, or `SafeResult.fail(...)`.
        """
        try:
            docs = await self.ainvoke(query)  # Re-use the main ainvoke logic
            payload = self._docs_to_payload(docs)

            # Add extra metadata to the payload
            for item in payload:
                item["metadata"] = dict(item.get("metadata") or {})
                item["metadata"].setdefault("page_content_field", self.content_field)
                item["metadata"].setdefault("source_collection", self.collection_name)

            return SafeResult.ok(payload)
        except Exception as e:
            logger.exception("Error during async ainvoke_sr for query: '%s'", query)
            return SafeResult.fail(str(e), exc=e)

    def retrieve(self, query: str) -> List[Document]:
        """
        Legacy sync helper for LangChain-style naming.

        Equivalent to :meth:`invoke`, provided for migration convenience.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        List[Document]
            Retrieved `Document` objects.
        """
        return self.invoke(query)

    async def aretrieve(self, query: str) -> List[Document]:
        """
        Legacy async helper for LangChain-style naming.

        Equivalent to :meth:`ainvoke`, provided for migration convenience.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        List[Document]
            Retrieved `Document` objects.
        """
        return await self.ainvoke(query)

    @property
    def is_configured(self) -> bool:
        """
        Whether all required components are present.

        Returns
        -------
        bool
            True if `repository`, `embedder`, `vector_searcher`, and
            `collection_name` are set; False otherwise.
        """
        return all([self.repository, self.embedder, self.vector_searcher, self.collection_name])

    def validate_configuration(self) -> None:
        """
        Validate that the retriever has the minimum required components.

        Raises
        ------
        RuntimeError
            If any core component is missing.
        """
        if not self.is_configured:
            raise RuntimeError("ZRetriever is not fully configured (missing components)")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Sync wrapper used by LangChain when async is not available.

        Parameters
        ----------
        query : str
            Query text.
        run_manager : AsyncCallbackManagerForRetrieverRun
            LangChain callback manager.

        Returns
        -------
        List[Document]
            Retrieved `Document` objects.
        """
        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    def _format_and_filter_results(
        self, items: List[Dict[str, Any]], query_embedding: List[float]
    ) -> List[Document]:
        """
        Transform raw search hits into LangChain `Document`s and filter by threshold.

        This method supports two hit "shapes":

        1. **New shape (preferred)**:
           `{"doc_id": str, "text": str, "metadata": dict, "retrieval_score": float, ...}`

        2. **Legacy shape**:
           `{"document": { <original doc dict> }, "retrieval_score": float, ...}`

        Any hit missing a numeric `retrieval_score` is skipped.

        Parameters
        ----------
        items : List[Dict[str, Any]]
            Raw items returned by `LocalVectorSearch.search(...)`.
        query_embedding : List[float]
            The query embedding (logged for traceability only).

        Returns
        -------
        List[Document]
            Filtered and formatted documents suitable for LangChain pipelines.
        """
        logger.debug(
            "Formatting and filtering %d raw hits with threshold >= %s",
            len(items),
            self.similarity_threshold,
        )
        docs: List[Document] = []
        for i, hit in enumerate(items or []):
            logger.debug("--- Processing Hit #%d: %s", i + 1, hit)
            score = hit.get("retrieval_score")

            if score is None:
                logger.warning("Hit #%d missing 'retrieval_score'. Skipping. Data: %s", i + 1, hit)
                continue

            # Explain how the score relates to the query vector for traceability
            logger.debug(
                "Hit #%d score = %.4f (cosine). Query head=%s",
                i + 1,
                score,
                np.array(query_embedding[:5]).round(4),
            )

            if score < self.similarity_threshold:
                logger.info(
                    "DISCARD Hit #%d — score %.4f < threshold %.2f",
                    i + 1,
                    score,
                    self.similarity_threshold,
                )
                continue

            logger.info(
                "KEEP Hit #%d — score %.4f >= threshold %.2f",
                i + 1,
                score,
                self.similarity_threshold,
            )

            # New shape (preferred)
            if "doc_id" in hit or "metadata" in hit or "text" in hit:
                content = hit.get("page_content") or hit.get("text") or ""
                meta = dict(hit.get("metadata") or {})
                meta["retrieval_score"] = score
                docs.append(Document(page_content=content, metadata=meta))

            # Legacy shape (document dict embedded under "document")
            elif "document" in hit and isinstance(hit["document"], dict):
                doc_data = hit["document"]
                content = doc_data.get(self.content_field) or doc_data.get("text") or ""
                meta = {
                    k: v
                    for k, v in doc_data.items()
                    if k not in (self.embedding_field, self.content_field)
                }
                meta["retrieval_score"] = score
                docs.append(Document(page_content=content, metadata=meta))
            else:
                logger.warning("Unrecognized hit format at #%d. Skipping. Data: %s", i + 1, hit)

        logger.info("Finished processing hits. Returning %d documents after filtering.", len(docs))
        return docs

    def invoke(self, query: str, **_: Any) -> List[Document]:
        """
        Runnable-style **synchronous** interface.

        Equivalent to calling :meth:`aretrieve` via `asyncio.run` internally.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        List[Document]
            Retrieved `Document` objects.
        """
        return asyncio.run(self.ainvoke(query))

    async def ainvoke(self, query: str, **_: Any) -> List[Document]:
        """
        Runnable-style **asynchronous** interface.

        This is the primary entry point for non-LangChain callers.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        List[Document]
            Retrieved `Document` objects.
        """
        # Provide a no-op run manager if LangChain's helper is not available
        try:
            run_manager: Any = AsyncCallbackManagerForRetrieverRun.get_noop_manager()
        except AttributeError:  # older LC versions without this helper
            class _NopManager:
                pass

            run_manager = _NopManager()

        return await self._aget_relevant_documents(query, run_manager=run_manager)

    @classmethod
    def from_components(
        cls,
        *,
        repository: Optional[ZMongo] = None,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_field: str = _DEFAULT_EMBED_FIELD,
        embedder: Optional[ZEmbedder] = None,
        vector_searcher: Optional[LocalVectorSearch] = None,
        top_k: int = 1,
        similarity_threshold: float = 0.5,
        query_embedding_style: str = _DEFAULT_QUERY_STYLE,
        content_field: str = _DEFAULT_CONTENT_FIELD,
    ) -> "ZRetriever":
        """
        Factory constructor for quick, consistent wiring.

        If any component is not supplied, a reasonable default will be created:

        - `repository` -> `ZMongo()`
        - `embedder` -> `ZEmbedder(repository=repo)`
        - `vector_searcher` -> `LocalVectorSearch` bound to the given collection

        Parameters
        ----------
        repository : ZMongo, optional
            Existing repository instance to use.
        collection_name : str, default "test"
            Collection name containing documents and embeddings.
        embedding_field : str, default "embeddings"
            Field name that stores embeddings in the collection.
        embedder : ZEmbedder, optional
            Existing embedder instance to use.
        vector_searcher : LocalVectorSearch, optional
            Existing vector searcher instance to use.
        top_k : int, default 1
            Number of hits to request from the vector searcher before threshold filtering.
        similarity_threshold : float, default 0.5
            Minimum cosine required to keep a hit.
        query_embedding_style : str, default EMBEDDING_STYLE_RETRIEVAL_QUERY
            Embedding style for queries.
        content_field : str, default "text"
            Field used as document page content.

        Returns
        -------
        ZRetriever
            A fully wired retriever ready to call `.ainvoke()`.

        Examples
        --------
        .. code-block:: python

            retriever = ZRetriever.from_components(
                collection_name="kb",
                embedding_field="embeddings",
                top_k=5,
                similarity_threshold=0.8,
            )
            results = await retriever.ainvoke("Your question here")
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


async def _demo() -> None:  # pragma: no cover — demo only
    """
    Run a small end-to-end demonstration of the retriever.

    The demo:
      1) Clears a test collection.
      2) Inserts a few short facts.
      3) Embeds them with `RETRIEVAL_DOCUMENT`.
      4) Runs a query with a higher threshold (0.80) to keep only strong matches.

    This function is intended for manual/local testing and is excluded from coverage.
    """
    from bson import ObjectId
    from zmongo_toolbag.zembedder_gemini import EMBEDDING_STYLE_RETRIEVAL_DOCUMENT

    repo = ZMongo()
    embedder = ZEmbedder(repository=repo)
    vector_searcher = LocalVectorSearch(
        repository=repo,
        collection=_DEFAULT_COLLECTION,
        embedding_field=_DEFAULT_EMBED_FIELD,
        chunked_embeddings=True,
        exact_rescore=True,
    )

    retriever = ZRetriever.from_components(
        repository=repo,
        embedder=embedder,
        vector_searcher=vector_searcher,
        collection_name=_DEFAULT_COLLECTION,
        similarity_threshold=0.80,
        top_k=3,
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
    if not ins.success:
        logger.error("Failed to insert documents: %s", ins.error)
        return

    for doc in knowledge_base:
        e = await embedder.embed_and_store(
            collection=_DEFAULT_COLLECTION,
            document_id=doc["_id"],
            text=doc["text"],
            embedding_field=_DEFAULT_EMBED_FIELD,
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        )
        if not e.success:
            logger.error("Failed to embed document %s: %s", doc["_id"], e.error)

    print("Successfully inserted and embedded", len(knowledge_base), "facts.\n")
    await asyncio.sleep(0.5)

    query = "What is the powerhouse of the cell?"
    print(f"--- Invoking retriever with query: '{query}' ---")
    results = await retriever.ainvoke(query)

    print(f"\nFound {len(results)} relevant document(s):")
    if not results:
        print(
            "No documents met the similarity threshold of "
            f"{retriever.similarity_threshold}."
        )
    else:
        for i, doc in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"  Content: {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")

    repo.close()
    embedder.close()


if __name__ == "__main__":
    # To see the detailed debug logs, set the level to DEBUG
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting ZRetriever demo...")
    asyncio.run(_demo())
