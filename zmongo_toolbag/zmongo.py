import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import Optional, List, Any, Union

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, InsertOne, DeleteOne, ReplaceOne  # Bulk operations

# Environment variables
load_dotenv()
DEFAULT_QUERY_LIMIT = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZMongo:
    """
    A repository class that interacts with MongoDB.
    Includes methods for CRUD operations and additional utilities.
    """

    def __init__(self) -> None:
        MONGO_URI = os.getenv("MONGO_URI")
        MONGO_DB_NAME = os.getenv("MONGO_DATABASE_NAME")

        if not MONGO_URI or not MONGO_DB_NAME:
            raise ValueError("MONGO_URI and MONGO_DATABASE_NAME must be set in environment variables.")

        self.mongo_client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.cache = defaultdict(dict)

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        """
        Normalize the collection name for consistency.
        """
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        """
        Generate a cache key for a given query.
        """
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def serialize_document(document: dict) -> dict:
        """
        Serialize a MongoDB document to JSON-compatible format.
        """
        return json.loads(json_util.dumps(document)) if document else {}

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """
        Find a single document in a collection by query.
        """
        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key(query)

        # Check if result is cached
        if cache_key in self.cache[normalized]:
            return self.cache[normalized][cache_key]

        document = await self.db[collection].find_one(filter=query)
        if document:
            serialized = self.serialize_document(document)
            self.cache[normalized][cache_key] = serialized
            return serialized
        return None

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        """
        Find multiple documents in a collection by query.
        """
        return await self.db[collection].find(filter=query, **kwargs).to_list(
            length=kwargs.get('limit', DEFAULT_QUERY_LIMIT)
        )

    async def insert_document(self, collection: str, document: dict) -> Any:
        """
        Insert a document into a collection.
        """
        result = await self.db[collection].insert_one(document)
        document["_id"] = result.inserted_id
        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key({"_id": str(result.inserted_id)})
        self.cache[normalized][cache_key] = self.serialize_document(document)
        return result

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False,
                              array_filters: Optional[List[dict]] = None) -> dict:
        """
        Update a document in a collection with optional `arrayFilters`, and update cache if applicable.
        """
        try:
            result = await self.db[collection].update_one(
                filter=query, update=update_data, upsert=upsert, array_filters=array_filters
            )

            # Update the cache if something was matched or upserted
            if result.matched_count > 0 or result.upserted_id:
                updated_doc = await self.db[collection].find_one(filter=query)
                if updated_doc:
                    normalized = self._normalize_collection_name(collection)
                    cache_key = self._generate_cache_key(query)
                    self.cache[normalized][cache_key] = self.serialize_document(updated_doc)

            return {
                "matchedCount": result.matched_count,
                "modifiedCount": result.modified_count,
                "upsertedId": result.upserted_id
            }
        except Exception as e:
            logger.error(f"Error updating document in {collection}: {e}")
            return {}


    async def delete_document(self, collection: str, query: dict) -> Any:
        """
        Delete a document from a collection.
        """
        result = await self.db[collection].delete_one(filter=query)
        if result.deleted_count:
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            self.cache[normalized].pop(cache_key, None)
        return result

    async def get_simulation_steps(self, collection: str, simulation_id: Union[str, ObjectId]) -> List[dict]:
        """
        Retrieves all simulation steps for a given simulation ID, sorted by `step`.
        """
        if isinstance(simulation_id, str):
            try:
                simulation_id = ObjectId(simulation_id)
            except Exception:
                logger.error(f"Invalid simulation_id: {simulation_id}")
                return []

        # Query for simulation steps
        query = {"simulation_id": simulation_id}
        steps = await self.db[collection].find(query).sort("step", 1).to_list(length=None)
        return [self.serialize_document(step) for step in steps]

    async def save_embedding(self, collection: str, document_id: ObjectId, embedding: List[float],
                             embedding_field: str = "embedding") -> None:
        """
        Save a generated embedding into a MongoDB document.
        """
        await self.db[collection].update_one(
            {"_id": document_id}, {"$set": {embedding_field: embedding}}, upsert=True
        )

    async def clear_cache(self) -> None:
        """
        Clears the local query cache.
        """
        self.cache.clear()
        logger.info("Cache cleared.")

    async def bulk_write(self, collection: str,
                         operations: List[Union[UpdateOne, InsertOne, DeleteOne, ReplaceOne]]) -> None:
        """
        Perform a bulk write operation on the collection.
        """
        if not operations:
            return
        await self.db[collection].bulk_write(operations)

    async def close(self) -> None:
        """
        Close the MongoDB client connection.
        """
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")


