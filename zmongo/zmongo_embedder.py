import asyncio
import logging
import os
import hashlib
from typing import List

import numpy as np
import openai
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from zmongo import zconstants
from zmongo.zmongo_chunker import ZMongoChunker
from zmongo.zmongo_repository import ZMongoRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ZMongoEmbedder:
    def __init__(
        self,
        page_content_keys: List[str],
        collection_name: str,
        embedding_model: str = zconstants.EMBEDDING_MODEL,
        max_tokens_per_chunk: int = 4096,
        overlap_prior_chunks: int = 0,
        encoding_name: str = zconstants.EMBEDDING_ENCODING,
        openai_api_key: str = os.getenv("OPENAI_API_KEY"),
    ):
        """
        Initialize the ZMongoEmbedder with the necessary parameters.
        """
        self.page_content_keys = page_content_keys
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Repository handles reading/writing from MongoDB
        self.zmongo_repository = ZMongoRepository()

        # The chunker is responsible for splitting text into smaller pieces
        self.chunker = ZMongoChunker(
            page_content_keys=page_content_keys,
            overlap_prior_chunks=overlap_prior_chunks,
            max_tokens_per_set=max_tokens_per_chunk,
            encoding_name=encoding_name,
            db_name=self.zmongo_repository.db_name,
            collection_name=collection_name,
        )

        # Set the API key for OpenAI
        openai.api_key = openai_api_key

        # In-memory chunk cache for this run:  {chunk_hash -> embedding vector}
        # This avoids repeated calls within the same run if multiple documents
        # share identical text chunks.
        self.chunk_cache = {}

    async def embed_collection(self) -> None:
        """
        Generate and save embeddings for all documents in the specified collection.
        Processes documents in batches to handle large datasets efficiently.
        """
        try:
            batch_size = 1000  # Adjust based on memory and performance requirements
            skip = 0
            total_processed = 0

            while True:
                # Fetch a batch of documents (just _id)
                documents = await self.zmongo_repository.find_documents(
                    collection=self.collection_name,
                    query={},
                    projection={"_id": 1},
                    limit=batch_size,
                    skip=skip
                )

                if not documents:
                    break  # No more documents to process

                logger.info(f"Processing batch of {len(documents)} documents (skip={skip})")

                # Collect object IDs
                object_ids = [str(doc["_id"]) for doc in documents]
                await self.process_documents(object_ids)

                total_processed += len(object_ids)
                skip += batch_size

            logger.info(
                f"Finished processing {total_processed} documents in collection '{self.collection_name}'."
            )

        except Exception as e:
            logger.error(f"Error while embedding collection '{self.collection_name}': {e}")
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using OpenAI's embeddings API.
        Retries up to 5 times with exponential backoff on transient errors.
        """
        try:
            response = openai.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except OpenAIError as e:
            logger.error(f"OpenAIError during get_embedding: {e}")
            raise

    def _hash_text(self, text: str) -> str:
        """
        Returns a SHA-256 hash of the text, used for chunk-level caching.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    async def process_documents(self, object_ids: List[str]) -> None:
        """
        Processes a list of document ObjectIds to generate and save their embeddings.
        Steps:
          1) For each doc ID, retrieve chunked text.
          2) Check if doc-level embedding for each content_key is already present.
          3) If not, embed the chunks that do not already have embeddings,
             compute a doc-level average, and store it in doc['embeddings'][content_key].
        """
        # 1. Convert IDs to valid ObjectId
        valid_object_ids = []
        for oid in object_ids:
            try:
                valid_object_ids.append(ObjectId(oid))
            except InvalidId:
                logger.error(f"Invalid ObjectId: {oid}")

        if not valid_object_ids:
            logger.warning("No valid ObjectIds after parsing.")
            return

        # 2. Chunk the documents
        documents = await self.chunker.invoke(
            object_ids=[str(vid) for vid in valid_object_ids],
            existing_metadata=None,
        )
        if not documents:
            logger.warning("No documents found or all documents are empty.")
            return

        # Group by (doc_id, content_key)
        docs_by_id_and_key = {}
        for doc in documents:
            document_id = doc.metadata.get("document_id")
            content_key = doc.metadata.get("page_content_key")

            if not document_id or not content_key:
                logger.warning(
                    f"Missing document_id or content_key in doc.metadata: {doc.metadata}"
                )
                continue

            docs_by_id_and_key.setdefault(document_id, {}).setdefault(content_key, []).append(doc)

        # 3. For each (doc_id, content_key) group, check doc-level embedding and embed chunks if needed
        for doc_id_str, content_map in docs_by_id_and_key.items():
            try:
                doc_id = ObjectId(doc_id_str)
            except InvalidId:
                logger.error(f"Invalid ObjectId in document metadata: {doc_id_str}")
                continue

            for content_key, doc_chunks in content_map.items():
                embedding_field = f"embeddings.{content_key.replace('.', '_')}"

                # 3a. Check doc-level embedding in MongoDB
                existing_embedding = await self.zmongo_repository.fetch_embedding(
                    collection=self.collection_name,
                    document_id=doc_id,
                    embedding_field=embedding_field
                )

                if existing_embedding:
                    # If doc-level embedding already exists, skip it
                    logger.info(
                        f"[SKIP] Found existing doc-level embedding for doc {doc_id} / {content_key}. "
                        f"No re-embedding needed."
                    )
                    continue

                # 3b. Embed the chunks if doc-level is missing
                chunk_embeddings = []
                for chunk_doc in doc_chunks:
                    chunk_text = chunk_doc.page_content.strip()
                    if not chunk_text:
                        logger.warning(
                            f"Empty chunk text for doc {doc_id}, content key '{content_key}'. Skipping."
                        )
                        continue

                    # Step 1: Check in-process cache
                    chash = self._hash_text(chunk_text)
                    if chash in self.chunk_cache:
                        chunk_embeddings.append(self.chunk_cache[chash])
                        continue

                    # Step 2: Check chunk-level embedding in MongoDB
                    chunk_embedding_field = (
                        f"chunk_embeddings.{content_key.replace('.', '_')}.{chash}"
                    )
                    chunk_embedding_in_db = await self.zmongo_repository.fetch_embedding(
                        collection=self.collection_name,
                        document_id=doc_id,
                        embedding_field=chunk_embedding_field
                    )
                    if chunk_embedding_in_db:
                        chunk_embeddings.append(chunk_embedding_in_db)
                        self.chunk_cache[chash] = chunk_embedding_in_db
                        continue

                    # Step 3: If not found, call OpenAI
                    try:
                        new_embedding = await self.get_embedding(chunk_text)
                        # Save chunk-level embedding to DB
                        await self.zmongo_repository.save_embedding(
                            collection=self.collection_name,
                            document_id=doc_id,
                            embedding=new_embedding,
                            embedding_field=chunk_embedding_field
                        )
                        # Update in-process cache
                        self.chunk_cache[chash] = new_embedding
                        chunk_embeddings.append(new_embedding)

                    except OpenAIError as e:
                        logger.error(
                            f"Error generating embedding for chunk in doc {doc_id}, content key '{content_key}': {e}"
                        )
                        # skip adding None embeddings

                # 3c. Compute average embedding if we got any chunk embeddings
                if chunk_embeddings:
                    embeddings_arr = np.array(chunk_embeddings, dtype=float)
                    avg_embedding = np.mean(embeddings_arr, axis=0)

                    if np.any(np.isnan(avg_embedding)) or np.any(np.isinf(avg_embedding)):
                        logger.error(
                            f"NaN/Inf found in average embedding for doc {doc_id}, key '{content_key}'. Skipping doc."
                        )
                        continue

                    final_embedding = avg_embedding.tolist()
                    # Convert each element to float
                    final_embedding = [float(x) for x in final_embedding]

                    # 3d. Save doc-level average embedding
                    await self.zmongo_repository.save_embedding(
                        collection=self.collection_name,
                        document_id=doc_id,
                        embedding=final_embedding,
                        embedding_field=embedding_field
                    )
                    logger.info(
                        f"Saved doc-level average embedding for doc ID {doc_id}, content key '{content_key}'."
                    )
                else:
                    logger.warning(
                        f"No valid chunk embeddings for doc {doc_id}, content key '{content_key}'."
                    )


async def main():
    """
    Example entry point.
    """
    page_content_keys = [
        "meaning_upright",
        "meaning_reversed",
    ]

    embedder = ZMongoEmbedder(
        page_content_keys=page_content_keys,
        collection_name="tarot_cards",   # Use your actual collection name
        embedding_model=zconstants.EMBEDDING_MODEL,
        max_tokens_per_chunk=128,
        overlap_prior_chunks=50,
        encoding_name=zconstants.EMBEDDING_ENCODING,
        openai_api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),
    )

    await embedder.embed_collection()


if __name__ == "__main__":
    asyncio.run(main())
