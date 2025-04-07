from typing import List, Union
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import logging

from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings

from zmongo_toolbag.utils.data_processing import DataProcessing

logger = logging.getLogger(__name__)

class ZRetriever:
    def __init__(self, repository, overlap_prior_chunks=0, max_tokens_per_set=4096, chunk_size=512,
                 embedding_length=1536, encoding_name='cl100k_base', use_embedding=False):
        self.repo = repository
        self.db = repository.db
        self.client = repository.mongo_client
        self.chunk_size = chunk_size
        self.max_tokens_per_set = max_tokens_per_set
        self.encoding_name = encoding_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(overlap_prior_chunks, chunk_size - 1)
        )
        self.overlap_prior_chunks = overlap_prior_chunks
        self.use_embedding = use_embedding
        self.ollama_embedding_model = OllamaEmbeddings(model="mistral")
        self.openai_embedding_model = OpenAIEmbeddings()
        self.embedding_model = self.openai_embedding_model

    def get_chunk_sets(self, chunks):
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

    def num_tokens_from_string(self, page_content) -> int:
        encoding = tiktoken.get_encoding(self.encoding_name)
        return len(encoding.encode(page_content))

    async def get_zdocuments(self, collection: str, object_ids: Union[str, List[str]],
                             page_content_key='casebody.data.opinions.0.text', existing_metadata=None):
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        documents = []
        for object_id in object_ids:
            doc = await self.repo.find_document(collection, {"_id": ObjectId(object_id)})
            if not doc:
                logger.warning(f"No document found for ID: {object_id}")
                continue
            page_content = DataProcessing.get_value(doc, page_content_key)
            if not isinstance(page_content, str):
                logger.warning(f"Invalid page content for ID: {object_id}")
                continue
            chunks = self.splitter.split_text(page_content)
            metadata = self._create_default_metadata(doc)
            combined_metadata = existing_metadata or {}
            combined_metadata.update(metadata)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=combined_metadata))
        return documents

    async def invoke(self, collection: str, object_ids: Union[str, List[str]],
                     page_content_key='casebody.data.opinions.0.text', existing_metadata=None):
        documents = await self.get_zdocuments(collection, object_ids, page_content_key, existing_metadata)
        return documents if self.max_tokens_per_set < 1 else self.get_chunk_sets(documents)
