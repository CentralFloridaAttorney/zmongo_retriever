import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Any

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv()

# Set default values with validation
DEFAULT_QUERY_LIMIT = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
TEST_COLLECTION_NAME = os.getenv("TEST_COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not TEST_COLLECTION_NAME:
    raise ValueError("TEST_COLLECTION_NAME must be set in environment variables.")
if not MONGO_URI:
    raise ValueError("MONGO_URI must be set in environment variables.")
if not MONGO_DB_NAME:
    raise ValueError("MONGO_DB_NAME must be set in environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZMongoRepository:
    """
    Asynchronous MongoDB repository with an in-memory cache and support for various MongoDB operations.
    """
    def __init__(self) -> None:
        self.mongo_client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.cache = defaultdict(dict)

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """Fetch a single document from MongoDB or cache."""
        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key(query)

        if cache_key in self.cache[normalized]:
            return self.cache[normalized][cache_key]

        document = await self.db[collection].find_one(filter=query)
        if document:
            serialized = self.serialize_document(document)
            self.cache[normalized][cache_key] = serialized
            return serialized
        return None

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        """Fetch multiple documents from MongoDB with optional parameters."""
        return await self.db[collection].find(filter=query, **kwargs).to_list(length=kwargs.get('limit', DEFAULT_QUERY_LIMIT))

    async def insert_document(self, collection: str, document: dict) -> Any:
        """Insert a document into MongoDB and cache it."""
        result = await self.db[collection].insert_one(document)
        document["_id"] = result.inserted_id

        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key({"_id": str(result.inserted_id)})
        self.cache[normalized][cache_key] = self.serialize_document(document)

        return result

    async def delete_document(self, collection: str, query: dict) -> Any:
        """Delete a document from MongoDB and remove it from cache."""
        result = await self.db[collection].delete_one(query)
        if result.deleted_count:
            self.cache[self._normalize_collection_name(collection)].pop(self._generate_cache_key(query), None)
        return result

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False) -> bool:
        """Update a document and update the cache."""
        result = await self.db[collection].update_one(filter=query, update=update_data, upsert=upsert)
        return result.modified_count > 0 or result.upserted_id is not None

    async def fetch_embedding(self, collection: str, document_id: ObjectId, embedding_field: str = "embedding") -> Optional[List[float]]:
        """Retrieve an embedding from a document by its ObjectId."""
        document = await self.db[collection].find_one({"_id": document_id}, {embedding_field: 1})
        return document.get(embedding_field) if document else None

    async def save_embedding(self, collection: str, document_id: ObjectId, embedding: List[float], embedding_field: str = "embedding") -> None:
        """Save or update an embedding vector in MongoDB."""
        await self.db[collection].update_one({"_id": document_id}, {"$set": {embedding_field: embedding}}, upsert=True)

    async def aggregate_documents(self, collection: str, pipeline: list, limit: int = DEFAULT_QUERY_LIMIT) -> List[dict]:
        """Run an aggregation pipeline in MongoDB."""
        return await self.db[collection].aggregate(pipeline).to_list(length=limit)

    async def bulk_write(self, collection: str, operations: list) -> None:
        """Perform bulk write operations in MongoDB."""
        await self.db[collection].bulk_write(operations)

    async def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self.cache.clear()
        logger.info("Cache cleared.")

    async def close(self) -> None:
        """Close the MongoDB client connection."""
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")

    @staticmethod
    def serialize_document(document: dict) -> dict:
        """Serialize a MongoDB document into JSON-compatible format."""
        return json.loads(json_util.dumps(document)) if document else {}


if __name__ == "__main__":
    async def main():
        repo = ZMongoRepository()
        test_collection = "test_collection"

        try:
            doc = {"name": "Alice", "role": "Engineer", "skills": ["Python", "MongoDB"], "score": 10}
            insert_result = await repo.insert_document(test_collection, doc)
            print(f"Inserted ID: {insert_result.inserted_id}")

            found = await repo.find_document(test_collection, {"_id": insert_result.inserted_id})
            print("Found document:", found)

            await repo.update_document(test_collection, {"_id": insert_result.inserted_id}, {"$set": {"score": 20}})
            updated = await repo.find_document(test_collection, {"_id": insert_result.inserted_id})
            print("Updated document:", updated)

            await repo.delete_document(test_collection, {"_id": insert_result.inserted_id})
            print("Deleted document.")
        finally:
            await repo.close()

    asyncio.run(main())
