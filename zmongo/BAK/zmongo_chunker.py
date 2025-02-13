import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
import tiktoken
from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
import openai  # Corrected import

from zmongo.utils.data_processing import DataProcessing

# Load environment variables
load_dotenv('.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve and validate environment variables
ENCODING_NAME = os.getenv("DEFAULT_ENCODING_NAME")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_ENCODING = os.getenv("EMBEDDING_ENCODING")
MONGO_URI = os.getenv("MONGO_URI")  # Ensure this is set

# Validation
# required_env_vars = {
#     "DEFAULT_ENCODING_NAME": ENCODING_NAME,
#     "MONGO_DATABASE_NAME": MONGO_DATABASE_NAME,
#     "DEFAULT_COLLECTION_NAME": DEFAULT_COLLECTION_NAME,
#     "OPENAI_API_KEY": OPENAI_API_KEY,
#     "EMBEDDING_ENCODING": EMBEDDING_ENCODING,
#     "MONGO_URI": MONGO_URI,
# }

# missing_vars = [var for var, value in required_env_vars.items() if not value]
# if missing_vars:
#     logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
#     raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class Document:
    def __init__(self, page_content: str, this_metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = this_metadata if this_metadata else {}

class ZMongoChunker:
    """
    ZMongoChunker retrieves and processes documents from a MongoDB collection,
    splitting them into chunks up to the maximum token count.
    Each chunk will also have metadata showing the number of tokens it contains.
    """

    def __init__(
        self,
        page_content_keys: List[str],
        overlap_prior_chunks: int = 0,
        max_tokens_per_set: int = 4096,
        encoding_name: str = ENCODING_NAME,
        db_name: str = MONGO_DATABASE_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        use_embedding: bool = False,
        openai_api_key: Optional[str] = OPENAI_API_KEY,
    ):
        from zmongo.BAK.zmongo_repository import ZMongoRepository
        self.mongo_repository = ZMongoRepository()
        self.db_name = db_name
        self.collection_name = collection_name
        self.page_content_fields = page_content_keys
        self.encoding_name = encoding_name
        self.max_tokens_per_set = max_tokens_per_set
        self.overlap_prior_chunks = overlap_prior_chunks

        # Initialize OpenAI client if needed
        if use_embedding:
            if not openai_api_key:
                logger.error("OpenAI API key is required for embeddings.")
                raise ValueError("OpenAI API key is missing.")
            openai.api_key = openai_api_key
        else:
            openai.api_key = None

        # Initialize the tokenizer encoding
        self.encoding = tiktoken.get_encoding(self.encoding_name)

    def _create_default_metadata(self, mongo_object: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a default metadata dictionary for a given MongoDB document object.
        """
        return {
            "source": "mongodb",
            "database_name": self.db_name,
            "collection_name": self.collection_name,
            "document_id": str(mongo_object.get("_id", "N/A")),
        }

    def num_tokens_from_string(self, page_content: str) -> int:
        """Returns the number of tokens in a text string."""
        return len(self.encoding.encode(page_content))

    def split_text(self, text: str, max_tokens: int, overlap: int = 0) -> List[str]:
        """Splits text into chunks of maximum token size with optional overlap."""
        tokens = self.encoding.encode(text)
        chunks = []
        stride = max_tokens - overlap if max_tokens > overlap else max_tokens
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            if i + max_tokens >= len(tokens):
                break
        return chunks

    async def get_zdocuments(
        self,
        object_ids: List[str],
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieves and processes documents from a MongoDB collection by their ObjectIDs,
        splitting them into chunks up to the maximum token limit.
        Adds token count metadata to each chunk.
        """
        if not object_ids:
            logger.warning("No object IDs provided.")
            return []

        # Validate and convert ObjectIds
        valid_object_ids = []
        for oid in object_ids:
            try:
                valid_object_ids.append(ObjectId(oid))
            except InvalidId:
                logger.error(f"Invalid ObjectId: {oid}")

        if not valid_object_ids:
            logger.warning("No valid ObjectIds after parsing.")
            return []

        # Use the initialized collection
        target_collection = self.collection_name

        # Build projection to include all content keys
        projection = {"_id": 1}
        for key in self.page_content_fields:
            projection[key] = 1

        # Fetch all documents in one query
        documents = await self.mongo_repository.find_documents(
            collection=target_collection,
            query={'_id': {'$in': valid_object_ids}},
            projection=projection,
            limit=len(valid_object_ids)
        )

        if not documents:
            logger.warning("No documents found for the provided ObjectIds.")
            return []

        # Process documents
        these_zdocuments = []

        for doc in documents:
            try:
                # Convert document to JSON-compatible format
                this_mongo_record = DataProcessing.convert_object_to_json(doc)

                # For each page_content_key, extract content and process
                for content_key in self.page_content_fields:
                    page_content = DataProcessing.get_value(
                        json_data=this_mongo_record, key=content_key
                    )

                    if not isinstance(page_content, str):
                        logger.warning(
                            f"Page content for key '{content_key}' in document ID {doc.get('_id')} is not a string or does not exist. Skipping this content."
                        )
                        continue

                    # Split the page_content into chunks
                    chunks = self.split_text(
                        page_content,
                        self.max_tokens_per_set,
                        overlap=self.overlap_prior_chunks
                    )
                    for chunk in chunks:
                        token_count = self.num_tokens_from_string(chunk)
                        # Create metadata for this chunk
                        metadata = existing_metadata.copy() if existing_metadata else {}
                        metadata.update(self._create_default_metadata(mongo_object=this_mongo_record))
                        metadata["token_count"] = token_count
                        metadata["page_content_key"] = content_key  # Include which key this content came from
                        these_zdocuments.append(
                            Document(page_content=chunk.strip(), this_metadata=metadata)
                        )
            except Exception as e:
                logger.error(f"Error processing document ID {doc.get('_id')}: {e}")

        return these_zdocuments

    async def invoke(
        self,
        object_ids: List[str],
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieves and processes a set of documents identified by their MongoDB object IDs,
        splitting them into chunks up to the maximum token limit.
        """
        return await self.get_zdocuments(
            object_ids=object_ids,
            existing_metadata=existing_metadata,
        )

async def main():
    """
    Main function to test the ZMongoChunker class.
    Initializes the chunker, retrieves documents, and prints the processed chunks.
    """
    # Test data: replace these with actual Object IDs from your database
    test_object_ids = [
        "66eda2f1b0b518a2e79e001d",  # Replace with valid ObjectId strings
        "66eda2f1b0b518a2e79e0027",
    ]

    # Initialize ZMongoChunker
    chunker = ZMongoChunker(
        page_content_keys=["meaning_upright", "meaning_reversed"],  # Replace with actual keys in your documents
        overlap_prior_chunks=10,
        max_tokens_per_set=100,  # Use a smaller max tokens for testing
        encoding_name=EMBEDDING_ENCODING,
        db_name=MONGO_DATABASE_NAME,
        collection_name="tarot_cards",  # Replace with your collection name
        use_embedding=False,  # Set to True if embedding is required
        openai_api_key=OPENAI_API_KEY,
    )

    try:
        # Invoke the chunker
        logger.info("Fetching and processing documents...")
        processed_documents = await chunker.invoke(
            object_ids=test_object_ids,
            existing_metadata={"source": "test_run"}  # Add any additional metadata
        )

        # Print the processed chunks
        if not processed_documents:
            logger.warning("No documents were processed.")
        else:
            for doc in processed_documents:
                logger.info(f"Chunk: {doc.page_content}")
                logger.info(f"Metadata: {doc.metadata}")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
    finally:
        # Clean up if needed
        logger.info("Test completed.")

if __name__ == "__main__":
    asyncio.run(main())
