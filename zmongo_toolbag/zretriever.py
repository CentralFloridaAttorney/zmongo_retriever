from typing import List, Union
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import logging

from langchain_openai import OpenAIEmbeddings

# Use the official OpenAIEmbeddings from langchain
# If an updated OllamaEmbeddings becomes available in langchain,
# you could import it here. For now, we default to OpenAIEmbeddings.
# from langchain.embeddings.ollama import OllamaEmbeddings

from .data_processing import DataProcessing

logger = logging.getLogger(__name__)


class ZRetriever:
    def __init__(self, repository, overlap_prior_chunks=0, max_tokens_per_set=4096, chunk_size=512,
                 embedding_length=1536, encoding_name='cl100k_base', use_embedding=False,
                 embedding_provider: str = 'openai'):
        """
        Initializes the retriever.

        Parameters:
          repository: An object with attributes `db` and `mongo_client` and a method `find_document`.
          overlap_prior_chunks (int): How many previous chunks to overlap.
          max_tokens_per_set (int): Maximum tokens allowed per chunk set.
          chunk_size (int): Token length of each chunk.
          embedding_length (int): Embedding vector length.
          encoding_name (str): Encoding name for tiktoken.
          use_embedding (bool): If True, embeddings will be used.
          embedding_provider (str): Either 'openai' or 'ollama'; defaults to 'openai'.
        """
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

        # Select embedding model based on provider choice.
        if embedding_provider.lower() == 'ollama':
            logger.warning("OllamaEmbeddings not yet supported; falling back to OpenAIEmbeddings.")
            self.embedding_model = OpenAIEmbeddings()
        else:
            self.embedding_model = OpenAIEmbeddings()

    def get_chunk_sets(self, chunks: List[Document]) -> List[List[Document]]:
        """
        Combines Document chunks into sets such that each set has fewer tokens than max_tokens_per_set.

        Parameters:
          chunks (List[Document]): List of Document objects.

        Returns:
          List[List[Document]]: List of chunk sets.
        """
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
        """
        Creates default metadata from a mongo document.
        """
        return {
            "source": "mongodb",
            "database_name": mongo_object.get("database_name", "unknown"),
            "collection_name": mongo_object.get("collection_name", "unknown"),
            "document_id": str(mongo_object.get("_id", "N/A")),
        }

    def num_tokens_from_string(self, page_content: str) -> int:
        """
        Computes number of tokens for a given string using tiktoken.
        """
        encoding = tiktoken.get_encoding(self.encoding_name)
        return len(encoding.encode(page_content))

    async def get_zdocuments(self, collection: str, object_ids: Union[str, List[str]],
                             page_content_key='casebody.data.opinions.0.text', existing_metadata=None) -> List[
        Document]:
        """
        Retrieves documents from MongoDB, splits their text into chunks, and attaches metadata.

        Parameters:
          collection (str): The MongoDB collection name.
          object_ids (Union[str, List[str]]): One or more document IDs.
          page_content_key (str): Key path to extract text content.
          existing_metadata: Additional metadata to merge.

        Returns:
          List[Document]: List of Document objects.
        """
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
                     page_content_key='casebody.data.opinions.0.text', existing_metadata=None) -> Union[
        List[Document], List[List[Document]]]:
        """
        Retrieves and chunks documents. If max_tokens_per_set is less than 1, returns the raw documents.

        Parameters:
          collection (str): MongoDB collection name.
          object_ids (Union[str, List[str]]): Document IDs.
          page_content_key (str): Key path for text extraction.
          existing_metadata: Additional metadata.

        Returns:
          Either a list of Document objects or a list of chunk sets.
        """
        documents = await self.get_zdocuments(collection, object_ids, page_content_key, existing_metadata)
        if self.max_tokens_per_set < 1:
            return documents
        return self.get_chunk_sets(documents)
