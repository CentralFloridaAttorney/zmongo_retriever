from typing import List, Union

import numpy as np
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import logging
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.utils.data_processing import DataProcessor
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

logger = logging.getLogger(__name__)

class ZRetriever:
    def __init__(
        self,
        collection: str,
        overlap_prior_chunks=0,
        max_tokens_per_set=4096,
        chunk_size=512,
        embedding_length=4096,
        encoding_name='cl100k_base',
        use_embedding=True,
        repository: ZMongo = None,
        model_path=None,
    ):
        self.repo = repository or ZMongo()
        self.collection = collection
        self.chunk_size = chunk_size
        self.max_tokens_per_set = max_tokens_per_set
        self.encoding_name = encoding_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(overlap_prior_chunks, chunk_size - 1)
        )
        self.overlap_prior_chunks = overlap_prior_chunks
        self.use_embedding = use_embedding
        # Use ZMongoEmbedder for embeddings (with cache)
        self.embedder = ZMongoEmbedder(
            collection, repository=self.repo, model_path=model_path
        )

    def get_chunk_sets(self, chunks: List[Document]) -> List[List[Document]]:
        max_tokens = self.max_tokens_per_set
        sized_chunks = []
        current_chunks = []
        current_tokens = 0
        for chunk in chunks:
            chunk_tokens = self.num_tokens_from_string(chunk.page_content)
            if current_tokens + chunk_tokens <= max_tokens:
                current_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                overlap_start = max(0, len(current_chunks) - self.overlap_prior_chunks)
                sized_chunks.append(current_chunks[:])
                current_chunks = current_chunks[overlap_start:]
                current_tokens = sum(self.num_tokens_from_string(c.page_content) for c in current_chunks)
                current_chunks.append(chunk)
                current_tokens += chunk_tokens
        if current_chunks:
            sized_chunks.append(current_chunks)
        return sized_chunks

    def _create_default_metadata(self, mongo_object):
        return {
            "source": "mongodb",
            "database_name": mongo_object.get("database_name", "unknown"),
            "collection_name": mongo_object.get("collection_name", "unknown"),
            "document_id": str(mongo_object.get("_id", "N/A")),
        }

    def num_tokens_from_string(self, page_content: str) -> int:
        encoding = tiktoken.get_encoding(self.encoding_name)
        return len(encoding.encode(page_content))

    async def get_zdocuments(
        self,
        object_ids: Union[str, List[str]],
        page_content_key='casebody.data.opinions.0.text',
        existing_metadata=None,
    ) -> List[Document]:
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        documents = []
        for object_id in object_ids:
            safe_id = ObjectId(object_id) if not isinstance(object_id, ObjectId) else object_id
            res = await self.repo.find_document(self.collection, {"_id": safe_id})
            if not res.success or not res.data:
                logger.warning(f"No document found for ID: {object_id}")
                continue
            doc = res.data
            page_content = DataProcessor.get_value(doc, page_content_key)
            if not isinstance(page_content, str):
                logger.warning(f"Invalid page content for ID: {object_id}")
                continue
            chunks = self.splitter.split_text(page_content)
            metadata = self._create_default_metadata(doc)
            combined_metadata = dict(existing_metadata) if existing_metadata else {}
            combined_metadata.update(metadata)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=combined_metadata))
        return documents

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        out = []
        for doc in documents:
            emb = await self.embedder.embed_text(doc.page_content)
            # embed_text should return [vector] for a short chunk
            if not emb or not isinstance(emb[0], list):
                raise ValueError(f"embed_text returned invalid: {emb}")
            out.append(emb[0])
        return out

    async def invoke(
        self,
        object_ids: Union[str, List[str]],
        page_content_key='casebody.data.opinions.0.text',
        existing_metadata=None,
    ) -> Union[List[Document], List[List[float]]]:
        documents = await self.get_zdocuments(object_ids, page_content_key, existing_metadata)
        if self.max_tokens_per_set < 1:
            return documents
        chunk_sets = self.get_chunk_sets(documents)
        if self.use_embedding:
            # Embed all document chunks for retrieval
            return [await self.embed_documents(chunk_set) for chunk_set in chunk_sets]
        return chunk_sets



    async def retrieve_by_embedding(self, query_embedding, top_k=3, embedding_field="_embedding", content_field="content"):
        """
        Retrieve top_k documents most similar to the query_embedding.
        Returns: List[Document] (langchain.schema.Document)
        """
        # Step 1: Get all docs with embeddings
        all_docs_result = await self.repo.find_documents(self.collection, {embedding_field: {"$exists": True}})
        docs = all_docs_result.data if hasattr(all_docs_result, "data") else all_docs_result

        # Step 2: Score by max similarity (across chunks)
        def cosine_similarity(a, b):
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
            if a.shape != b.shape or a.size == 0:
                return -1.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        scored = []
        for doc in docs:
            embeddings = doc.get(embedding_field, [])
            if not embeddings:
                continue
            valid_scores = [
                cosine_similarity(query_embedding, emb)
                for emb in embeddings
                if isinstance(emb, (list, np.ndarray)) and len(emb) == len(query_embedding)
            ]
            if not valid_scores:
                continue
            max_score = max(valid_scores)
            scored.append((max_score, doc))

        # Step 3: Sort, select top_k
        scored.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for score, doc in scored[:top_k]]

        # Step 4: Convert to langchain Document objects
        results = []
        for doc in top_docs:
            page_content = doc.get(content_field, "")
            metadata = dict(doc)
            metadata.pop(embedding_field, None)
            metadata.pop(content_field, None)
            results.append(Document(page_content=page_content, metadata=metadata))
        return results
