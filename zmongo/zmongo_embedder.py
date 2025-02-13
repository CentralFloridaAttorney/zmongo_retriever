import asyncio
import logging
import os
import hashlib
from typing import List

import numpy as np
import openai
import tenacity
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from zmongo.zmongo_chunker import ZMongoChunker
from zmongo.zmongo_repository import ZMongoRepository

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


class ZMongoEmbedder:
    def __init__(
        self,
        page_content_keys: List[str],
        collection_name: str,
        embedding_model: str = os.getenv('EMBEDDING_MODEL_OPENAI'),
        max_tokens_per_chunk: int = 4096,
        overlap_prior_chunks: int = 0,
        encoding_name: str = os.getenv('EMBEDDING_ENCODING'),
        openai_api_key: str = os.getenv("OPENAI_API_KEY"),
    ):
        """
        Initialize the embedder with all necessary parameters.
        """
        self.page_content_keys = page_content_keys
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Instantiate repository and chunker.
        self.zmongo_repository = ZMongoRepository()
        self.chunker = ZMongoChunker(
            page_content_keys=page_content_keys,
            overlap_prior_chunks=overlap_prior_chunks,
            max_tokens_per_set=max_tokens_per_chunk,
            encoding_name=encoding_name,
            db_name=self.zmongo_repository.db_name,
            collection_name=collection_name,
        )

        openai.api_key = openai_api_key

        # In–process cache for chunk embeddings: {chunk_hash -> embedding vector}
        self.chunk_cache = {}

    async def embed_collection(self) -> None:
        """
        Process the entire collection in batches. For each document, this function
        will obtain embeddings for each specified content key.
        """
        try:
            batch_size = 1000
            skip = 0
            total_processed = 0

            while True:
                # Fetch only document _id fields.
                documents = await self.zmongo_repository.find_documents(
                    collection=self.collection_name,
                    query={},
                    projection={"_id": 1},
                    limit=batch_size,
                    skip=skip
                )

                if not documents:
                    break

                logger.info(f"Processing batch of {len(documents)} documents (skip={skip})")
                object_ids = [str(doc["_id"]) for doc in documents]
                await self.process_documents(object_ids)

                total_processed += len(object_ids)
                skip += batch_size

            logger.info(f"Finished processing {total_processed} documents in collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error while embedding collection '{self.collection_name}': {e}")
            raise

    async def _get_embedding_from_api(self, text: str) -> List[float]:
        """
        Call the external API (OpenAI) to generate an embedding for the provided text.
        """
        logger.debug(f"Calling API for text (first 30 chars): {text[:30]}...")
        response = openai.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        embedding = response.data[0].embedding
        logger.info("Generated new embedding from API.")
        return embedding

    async def get_embedding(self, doc_id: ObjectId, content_key: str, text: str) -> List[float]:
        """
        Retrieve an embedding for a given document field.
        First, check MongoDB for an existing embedding (stored under a field
        name based on the content key). If found, return it.
        Otherwise, call the API to generate the embedding, save it, and return it.
        """
        embedding_field = f"embeddings.{content_key.replace('.', '_')}"
        logger.debug(f"Attempting to retrieve embedding from MongoDB for doc {doc_id}, field '{embedding_field}'.")
        existing = await self.zmongo_repository.fetch_embedding(
            collection=self.collection_name,
            document_id=doc_id,
            embedding_field=embedding_field
        )
        if existing is not None:
            logger.info(f"Using embedding from MongoDB for doc {doc_id} and key '{content_key}'.")
            return existing
        else:
            logger.info(f"No embedding found in MongoDB for doc {doc_id} and key '{content_key}'. Fetching from API.")
            new_embedding = await self._get_embedding_from_api(text)
            logger.debug(f"Saving new embedding to MongoDB for doc {doc_id}, field '{embedding_field}'.")
            await self.zmongo_repository.save_embedding(
                collection=self.collection_name,
                document_id=doc_id,
                embedding=new_embedding,
                embedding_field=embedding_field
            )
            return new_embedding

    def _hash_text(self, text: str) -> str:
        """
        Return a SHA-256 hash of the given text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @tenacity.retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=10),
        retry=tenacity.retry_if_exception_type(OpenAIError)
    )
    async def process_documents(self, object_ids: List[str]) -> None:
        """
        Process a list of document IDs: for each document and for each specified content key,
        obtain chunked text, check for existing embeddings (both at document–level and chunk–level)
        and if necessary call get_embedding() to generate and save embeddings. Finally,
        compute an average embedding for the document and save it.
        """
        # Convert string IDs to ObjectId instances.
        valid_object_ids = []
        for oid in object_ids:
            try:
                valid_object_ids.append(ObjectId(oid))
            except InvalidId:
                logger.error(f"Invalid ObjectId: {oid}")

        if not valid_object_ids:
            logger.warning("No valid ObjectIds after parsing.")
            return

        # Invoke the chunker to get document chunks.
        documents = await self.chunker.invoke(
            object_ids=[str(vid) for vid in valid_object_ids],
            existing_metadata=None,
        )
        if not documents:
            logger.warning("No documents found or all documents are empty.")
            return

        # Group chunks by document ID and content key.
        docs_by_id_and_key = {}
        for doc in documents:
            document_id = doc.metadata.get("document_id")
            content_key = doc.metadata.get("page_content_key")
            if not document_id or not content_key:
                logger.warning(f"Missing document_id or content_key in metadata: {doc.metadata}")
                continue
            docs_by_id_and_key.setdefault(document_id, {}).setdefault(content_key, []).append(doc)

        # Process each document and key group.
        for doc_id_str, content_map in docs_by_id_and_key.items():
            try:
                doc_id = ObjectId(doc_id_str)
            except InvalidId:
                logger.error(f"Invalid ObjectId in metadata: {doc_id_str}")
                continue

            for content_key, doc_chunks in content_map.items():
                embedding_field = f"embeddings.{content_key.replace('.', '_')}"
                logger.debug(
                    f"Checking for existing document-level embedding for doc {doc_id}, field '{embedding_field}'.")
                existing_embedding = await self.zmongo_repository.fetch_embedding(
                    collection=self.collection_name,
                    document_id=doc_id,
                    embedding_field=embedding_field
                )
                if existing_embedding:
                    logger.info(
                        f"[SKIP] Existing document-level embedding found for doc {doc_id} and key '{content_key}'.")
                    continue

                chunk_embeddings = []
                for chunk_doc in doc_chunks:
                    # -----------------------------------------------------------------------------------
                    # NEW: Convert all non-string values to string before embedding.
                    # -----------------------------------------------------------------------------------
                    if chunk_doc.page_content is None:
                        logger.warning(
                            f"Page content for key '{content_key}' in doc {doc_id} is None. Skipping."
                        )
                        continue

                    # If it's not a string, make it one (e.g., ObjectId -> 'ObjectId("...")')
                    # or just str(...) if you prefer simpler output.
                    if not isinstance(chunk_doc.page_content, str):
                        chunk_doc.page_content = str(chunk_doc.page_content)

                    chunk_text = chunk_doc.page_content.strip()
                    if not chunk_text:
                        logger.warning(f"Empty chunk text for doc {doc_id}, key '{content_key}'. Skipping.")
                        continue

                    chash = self._hash_text(chunk_text)
                    if chash in self.chunk_cache:
                        logger.debug(f"Using cached chunk embedding for doc {doc_id}, key '{content_key}'.")
                        chunk_embeddings.append(self.chunk_cache[chash])
                        continue

                    # Check for a stored chunk embedding in MongoDB.
                    chunk_embedding_field = f"chunk_embeddings.{content_key.replace('.', '_')}.{chash}"
                    logger.debug(
                        f"Checking for stored chunk embedding for doc {doc_id}, field '{chunk_embedding_field}'.")
                    chunk_embedding_in_db = await self.zmongo_repository.fetch_embedding(
                        collection=self.collection_name,
                        document_id=doc_id,
                        embedding_field=chunk_embedding_field
                    )
                    if chunk_embedding_in_db:
                        logger.info(f"Using stored chunk embedding for doc {doc_id}, key '{content_key}'.")
                        chunk_embeddings.append(chunk_embedding_in_db)
                        self.chunk_cache[chash] = chunk_embedding_in_db
                        continue

                    # Otherwise, generate a new embedding for this chunk.
                    logger.info(
                        f"No stored chunk embedding for doc {doc_id}, key '{content_key}'. Calling API for chunk."
                    )
                    try:
                        new_embedding = await self.get_embedding(doc_id, content_key, chunk_text)
                        if new_embedding:
                            chunk_embeddings.append(new_embedding)
                            self.chunk_cache[chash] = new_embedding
                        else:
                            logger.warning(
                                f"API returned no embedding for doc {doc_id}, key '{content_key}'."
                            )
                    except OpenAIError as e:
                        logger.error(f"Error generating embedding for doc {doc_id}, key '{content_key}': {e}")
                        continue

                # Compute and save the average embedding for the document if any chunk embeddings exist.
                if chunk_embeddings:
                    embeddings_arr = np.array(chunk_embeddings, dtype=float)
                    avg_embedding = np.mean(embeddings_arr, axis=0)
                    if np.any(np.isnan(avg_embedding)) or np.any(np.isinf(avg_embedding)):
                        logger.error(f"Invalid average embedding for doc {doc_id}, key '{content_key}'. Skipping.")
                        continue

                    final_embedding = [float(x) for x in avg_embedding.tolist()]
                    logger.debug(
                        f"Saving document-level averaged embedding for doc {doc_id}, field '{embedding_field}'."
                    )
                    await self.zmongo_repository.save_embedding(
                        collection=self.collection_name,
                        document_id=doc_id,
                        embedding=final_embedding,
                        embedding_field=embedding_field
                    )
                    logger.info(f"Saved document-level averaged embedding for doc {doc_id}, key '{content_key}'.")
                else:
                    logger.warning(f"No valid chunk embeddings for doc {doc_id}, key '{content_key}'.")

    async def main(self):
        """
        Example entry point for the embedder.
        """
        # Example fields you want to process.
        page_content_keys = [
            "username",
            # You can add more keys if needed, e.g. "email", "roles", etc.
        ]

        embedder = ZMongoEmbedder(
            page_content_keys=page_content_keys,
            collection_name="user",
            embedding_model=os.getenv('EMBEDDING_MODEL_OPENAI'),
            max_tokens_per_chunk=128,
            overlap_prior_chunks=50,
            encoding_name=os.getenv('EMBEDDING_ENCODING'),
            openai_api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),
        )

        await embedder.embed_collection()


if __name__ == "__main__":
    asyncio.run(ZMongoEmbedder([], "").main())
