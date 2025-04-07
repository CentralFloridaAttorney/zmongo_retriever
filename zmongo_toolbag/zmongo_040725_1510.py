import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import Optional, List, Any, Union, Dict

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, InsertOne, DeleteOne, ReplaceOne, MongoClient
from pymongo.results import InsertOneResult

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
        self.MONGO_URI = os.getenv("MONGO_URI")
        self.MONGO_DB_NAME = os.getenv("MONGO_DATABASE_NAME")

        if not self.MONGO_URI:
            logger.warning("⚠️ MONGO_URI is not set in .env. Defaulting to 'mongodb://127.0.0.1:27017'")
            self.MONGO_URI = "mongodb://127.0.0.1:27017"

        if not self.MONGO_DB_NAME:
            logger.warning("❌ MONGO_DATABASE_NAME is not set in .env. Defaulting to 'documents'")
            self.MONGO_DB_NAME = "documents"

        self.mongo_client = AsyncIOMotorClient(self.MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[self.MONGO_DB_NAME]
        self.cache = defaultdict(dict)

        self.sync_client = MongoClient(self.MONGO_URI, maxPoolSize=200)
        self.sync_db = self.sync_client[self.MONGO_DB_NAME]

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def serialize_document(document: dict) -> dict:
        return json.loads(json_util.dumps(document)) if document else {}

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
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

    # Alias for convenience
    find_one = find_document

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        return await self.db[collection].find(filter=query, **kwargs).to_list(
            length=kwargs.get('limit', DEFAULT_QUERY_LIMIT)
        )


    async def insert_document(self, collection: str, document: dict, use_cache: bool = True) -> Optional[
        InsertOneResult]:
        """
        Inserts a single document into the specified collection.

        :param collection: The collection name.
        :param document: The document to insert.
        :param use_cache: Whether to cache the inserted document.
        :return: The InsertOneResult if successful, else None.
        """
        try:
            result = await self.db[collection].insert_one(document)
            document["_id"] = result.inserted_id

            if use_cache:
                normalized = self._normalize_collection_name(collection)
                cache_key = self._generate_cache_key({"_id": str(result.inserted_id)})
                self.cache[normalized][cache_key] = self.serialize_document(document)

            # Returning the InsertOneResult object
            return result

        except Exception as e:
            logger.error(f"Error inserting document into '{collection}': {e}")
            return None

    async def insert_documents(
            self, collection: str, documents: List[dict],
            batch_size: int = 1000, use_cache: bool = True, use_sync: bool = False
    ) -> Dict[str, Union[int, List[str]]]:

        if use_sync:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self.insert_documents_sync,
                collection,
                documents,
                batch_size
            )

        if not documents:
            return {"inserted_count": 0}

        total_inserted = 0
        errors = []
        normalized = self._normalize_collection_name(collection)

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = await self.db[collection].insert_many(batch, ordered=False)
                if use_cache:
                    for doc, _id in zip(batch, result.inserted_ids):
                        doc["_id"] = _id
                        cache_key = self._generate_cache_key({"_id": str(_id)})
                        self.cache[normalized][cache_key] = self.serialize_document(doc)

                total_inserted += len(result.inserted_ids)
            except Exception as e:
                error_msg = f"Batch insert failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        response = {"inserted_count": total_inserted}
        if errors:
            response["errors"] = errors
        return response

    def insert_documents_sync(self, collection: str, documents: List[dict], batch_size: int = 1000) -> dict:
        if not documents:
            return {"inserted_count": 0}

        total_inserted = 0
        errors = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = self.sync_db[collection].insert_many(batch, ordered=False)
                total_inserted += len(result.inserted_ids)
            except Exception as e:
                errors.append(str(e))

        response = {"inserted_count": total_inserted}
        if errors:
            response["errors"] = errors
        return response

    def log_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Logs training metrics to the 'training_metrics' collection in MongoDB.
        Adds a UTC timestamp to the metrics document and then inserts it synchronously.

        Parameters:
          metrics (Dict[str, Any]): A dictionary containing training metrics,
                                    e.g., {"loss_D": 0.1234, "loss_G": 0.5678}.
        """
        try:
            from datetime import datetime
            metrics_doc = {
                "timestamp": datetime.utcnow(),
                **metrics
            }
            # Using the synchronous client for logging
            self.sync_db["training_metrics"].insert_one(metrics_doc)
            logger.info(f"Logged training metrics: {metrics_doc}")
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False,
                              array_filters: Optional[List[dict]] = None) -> dict:
        try:
            result = await self.db[collection].update_one(
                filter=query, update=update_data, upsert=upsert, array_filters=array_filters
            )

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

    async def delete_all_documents(self, collection: str) -> int:
        result = await self.db[collection].delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents from '{collection}'")
        return result.deleted_count

    async def delete_document(self, collection: str, query: dict) -> Any:
        result = await self.db[collection].delete_one(filter=query)
        if result.deleted_count:
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            self.cache[normalized].pop(cache_key, None)
        return result

    async def get_simulation_steps(self, collection: str, simulation_id: Union[str, ObjectId]) -> List[dict]:
        if isinstance(simulation_id, str):
            try:
                simulation_id = ObjectId(simulation_id)
            except Exception:
                logger.error(f"Invalid simulation_id: {simulation_id}")
                return []

        query = {"simulation_id": simulation_id}
        steps = await self.db[collection].find(query).sort("step", 1).to_list(length=None)
        return [self.serialize_document(step) for step in steps]

    async def save_embedding(self, collection: str, document_id: ObjectId, embedding: List[float],
                             embedding_field: str = "embedding") -> None:
        try:
            query = {"_id": document_id}
            update_data = {"$set": {embedding_field: embedding}}

            await self.db[collection].update_one(query, update_data, upsert=True)

            updated_doc = await self.db[collection].find_one(query)
            if updated_doc:
                normalized = self._normalize_collection_name(collection)
                cache_key = self._generate_cache_key(query)
                self.cache[normalized][cache_key] = self.serialize_document(updated_doc)
        except Exception as e:
            # logger.error(f"Error saving embedding to {collection}: {e}")
            print(f"Error saving embedding to {collection}: {e}")
    async def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Cache cleared.")

    async def bulk_write(self, collection: str,
                         operations: List[Union[UpdateOne, InsertOne, DeleteOne, ReplaceOne]]) -> None:
        if not operations:
            return
        await self.db[collection].bulk_write(operations)

    async def close(self) -> None:
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")
