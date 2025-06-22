import hashlib
import os
import logging
from typing import List, Optional, Sequence
from bson.objectid import ObjectId
from dotenv import load_dotenv

from zmongo_toolbag.zmongo import ZMongo

logger = logging.getLogger(__name__)
load_dotenv()

# --- You may need to pip install llama-cpp-python tiktoken ---
from llama_cpp import Llama
import tiktoken

CHUNK_SIZE = 1024  # tokens per chunk, adjust for best practice
CHUNK_OVERLAP = 100  # overlap in tokens between chunks

class ZMongoEmbedder:
    def __init__(self, collection: str, repository: Optional[ZMongo] = None, model_path: Optional[str] = None) -> None:
        self.repository = repository or ZMongo()
        self.collection = collection
        # Use local GGUF embedding model (Mistral-7B in this case)
        self.model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH",
            "C:/Users/iriye/resources/models/mistral-7b-instruct-v0.1.Q4_0.gguf"
        )
        self._llama = None
        self.encoding_name = os.getenv("EMBEDDING_ENCODING", "cl100k_base")
        self.max_tokens = CHUNK_SIZE

    @property
    def llama(self):
        if self._llama is None:
            self._llama = Llama(
            model_path=self.model_path,
            embedding=True,  # enable embedding mode
            n_ctx=CHUNK_SIZE,  # use as context window
            n_threads=os.cpu_count() or 4
        )
        return self._llama

    def _tokenize(self, text: str) -> List[int]:
        encoding = tiktoken.get_encoding(self.encoding_name)
        return encoding.encode(text)

    def _detokenize(self, tokens: Sequence[int]) -> str:
        encoding = tiktoken.get_encoding(self.encoding_name)
        return encoding.decode(tokens)

    def _split_chunks(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        tokens = self._tokenize(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i : i + chunk_size]
            chunks.append(self._detokenize(chunk))
            if i + chunk_size >= len(tokens):
                break
            i += chunk_size - overlap
        return chunks

    async def embed_text(self, text: str) -> List[List[float]]:
        """
        Generate embeddings for a given text, splitting long docs and returning a list of embeddings (one per chunk).
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        chunks = self._split_chunks(text)
        embeddings = []
        for chunk in chunks:
            # Use hash of chunk for cache
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            cached_doc = await self.repository.find_document("_embedding_cache", {"text_hash": chunk_hash})
            if cached_doc and cached_doc.success and cached_doc.data and "embedding" in cached_doc.data:
                logger.info(f"🔁 Reusing cached embedding for chunk hash: {chunk_hash}")
                embeddings.append(cached_doc.data["embedding"])
                continue

            # Get embedding from model
            result = self.llama.create_embedding(chunk)
            if isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
                embedding = result["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected embedding result format: {result}")
            await self.repository.insert_document("_embedding_cache", {
                "text_hash": chunk_hash,
                "embedding": embedding,
                "source_text": chunk
            })
            embeddings.append(embedding)
        return embeddings

    async def embed_and_store(self, document_id: ObjectId, text: str, embedding_field: str = "embeddings") -> None:
        """
        Embed the text (possibly split into chunks), store all chunk embeddings in a list field in Mongo.
        """
        if not isinstance(document_id, ObjectId):
            raise ValueError("document_id must be an instance of ObjectId")
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")
        embeddings = await self.embed_text(text)
        # Store the embeddings list into the document, creating if necessary (upsert)
        update_result = await self.repository.update_document(
            self.collection,
            {"_id": document_id},
            {embedding_field: embeddings},
            upsert=True,
        )
        if not update_result.success:
            raise RuntimeError(f"Failed to store embeddings: {update_result.error}")
