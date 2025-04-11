import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Any, Union, Dict

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, InsertOne, DeleteOne, ReplaceOne, MongoClient
from pymongo.errors import BulkWriteError, PyMongoError
from pymongo.results import InsertOneResult

load_dotenv()
DEFAULT_QUERY_LIMIT: int = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZMongo:
    def __init__(self) -> None:
        self.MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.MONGO_DB_NAME: str = os.getenv("MONGO_DATABASE_NAME", "documents")

        if not os.getenv("MONGO_URI"):
            logger.warning("\u26a0\ufe0f MONGO_URI is not set in .env. Defaulting to 'mongodb://127.0.0.1:27017'")
        if not os.getenv("MONGO_DATABASE_NAME"):
            logger.warning("\u274c MONGO_DATABASE_NAME is not set in .env. Defaulting to 'documents'")

        self.mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(self.MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[self.MONGO_DB_NAME]
        self.cache: Dict[str, Dict[str, dict]] = defaultdict(dict)
        self.sync_client: MongoClient = MongoClient(self.MONGO_URI, maxPoolSize=200)
        self.sync_db = self.sync_client[self.MONGO_DB_NAME]

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        query_json = json.dumps(query, sort_keys=True, default=str)
        return hashlib.sha256(query_json.encode("utf-8")).hexdigest()

    @staticmethod
    def serialize_document(document: dict) -> dict:
        return json.loads(json_util.dumps(document)) if document else {}

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        limit = kwargs.get("limit", DEFAULT_QUERY_LIMIT)
        cursor = self.db[collection].find(filter=query, **kwargs)
        documents = await cursor.to_list(length=limit)
        return [self.serialize_document(doc) for doc in documents]

    async def get_field_names(self, collection: str, sample_size: int = 10) -> List[str]:
        try:
            cursor = self.db[collection].find({}, projection={"_id": 0}).limit(sample_size)
            documents = await cursor.to_list(length=sample_size)
            fields = set()
            for doc in documents:
                fields.update(doc.keys())
            return list(fields)
        except Exception as e:
            logger.error(f"Failed to extract fields from collection '{collection}': {e}")
            return []

    async def sample_documents(self, collection: str, sample_size: int = 5) -> List[dict]:
        try:
            cursor = self.db[collection].find({}).limit(sample_size)
            documents = await cursor.to_list(length=sample_size)
            return [self.serialize_document(doc) for doc in documents]
        except Exception as e:
            logger.error(f"Failed to get sample documents from '{collection}': {e}")
            return []

    async def text_search(self, collection: str, search_text: str, limit: int = 10) -> List[dict]:
        try:
            cursor = self.db[collection].find({"$text": {"$search": search_text}}).limit(limit)
            results = await cursor.to_list(length=limit)
            return [self.serialize_document(doc) for doc in results]
        except Exception as e:
            logger.error(f"Text search failed in '{collection}': {e}")
            return []

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        """Find and return a single document from a collection.

        Args:
            collection: The collection name.
            query: The filter query.

        Returns:
            The serialized document if found, else None.
        """
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

    async def insert_document(
        self, collection: str, document: dict, use_cache: bool = True
    ) -> Optional[InsertOneResult]:
        """Insert a single document into a collection.

        Args:
            collection: The collection name.
            document: The document to insert.
            use_cache: Whether to cache the inserted document.

        Returns:
            An InsertOneResult object if insertion is successful; otherwise, None.
        """
        try:
            result: InsertOneResult = await self.db[collection].insert_one(document)
            document["_id"] = result.inserted_id

            if use_cache:
                normalized = self._normalize_collection_name(collection)
                cache_key = self._generate_cache_key({"_id": str(result.inserted_id)})
                self.cache[normalized][cache_key] = self.serialize_document(document)

            return result
        except Exception as e:
            logger.error(f"Error inserting document into '{collection}': {e}")
            return None

    async def insert_documents(
        self,
        collection: str,
        documents: List[dict],
        batch_size: int = 1000,
        use_cache: bool = True,
        use_sync: bool = False,
    ) -> Dict[str, Union[int, List[str]]]:
        """Insert multiple documents into a collection in batches.

        Args:
            collection: The collection name.
            documents: List of documents to insert.
            batch_size: Number of documents per batch.
            use_cache: Whether to cache inserted documents.
            use_sync: Whether to use the synchronous insertion method.

        Returns:
            A dict with the total inserted count and any errors.
        """
        if use_sync:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.insert_documents_sync, collection, documents, batch_size
            )

        if not documents:
            return {"inserted_count": 0}

        total_inserted = 0
        errors: List[str] = []
        normalized = self._normalize_collection_name(collection)

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
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

        response: Dict[str, Union[int, List[str]]] = {"inserted_count": total_inserted}
        if errors:
            response["errors"] = errors
        return response

    def insert_documents_sync(self, collection: str, documents: List[dict], batch_size: int = 1000) -> dict:
        """Synchronously insert multiple documents into a collection.

        Args:
            collection: The collection name.
            documents: List of documents to insert.
            batch_size: Number of documents per batch.

        Returns:
            A dict with the total inserted count and any errors.
        """
        if not documents:
            return {"inserted_count": 0}

        total_inserted = 0
        errors: List[str] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
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
        """Log training metrics to the 'training_metrics' collection.

        Adds a UTC timestamp to the metrics document and inserts it synchronously.

        Args:
            metrics: A dictionary of training metrics.
        """
        try:
            metrics_doc = {"timestamp": datetime.utcnow(), **metrics}
            self.sync_db["training_metrics"].insert_one(metrics_doc)
            logger.info(f"Logged training metrics: {metrics_doc}")
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")

    async def update_document(
        self,
        collection: str,
        query: dict,
        update_data: dict,
        upsert: bool = False,
        array_filters: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Update a document in the specified collection.

        Args:
            collection: The collection name.
            query: The filter query.
            update_data: The update operations.
            upsert: Whether to upsert if no document matches.
            array_filters: Optional filters for array updates.

        Returns:
            A dictionary with keys: matched_count, modified_count, and upserted_id.
        """
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
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "upserted_id": result.upserted_id,
            }
        except Exception as e:
            logger.error(f"Error updating document in {collection}: {e}")
            return {}

    async def delete_all_documents(self, collection: str) -> int:
        """Delete all documents from a collection.

        Args:
            collection: The collection name.

        Returns:
            The number of documents deleted.
        """
        result = await self.db[collection].delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents from '{collection}'")
        return result.deleted_count

    async def delete_document(self, collection: str, query: dict) -> Any:
        """Delete a single document from a collection.

        Args:
            collection: The collection name.
            query: The filter query.

        Returns:
            The result of the delete operation.
        """
        result = await self.db[collection].delete_one(filter=query)
        if result.deleted_count:
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            self.cache[normalized].pop(cache_key, None)
        return result

    async def get_simulation_steps(self, collection: str, simulation_id: Union[str, ObjectId]) -> List[dict]:
        """Retrieve simulation steps for a given simulation ID.

        Args:
            collection: The collection name.
            simulation_id: The simulation identifier (str or ObjectId).

        Returns:
            A list of serialized simulation step documents.
        """
        if isinstance(simulation_id, str):
            try:
                simulation_id = ObjectId(simulation_id)
            except Exception:
                logger.error(f"Invalid simulation_id: {simulation_id}")
                return []

        query = {"simulation_id": simulation_id}
        steps = await self.db[collection].find(query).sort("step", 1).to_list(length=None)
        return [self.serialize_document(step) for step in steps]

    async def save_embedding(
        self,
        collection: str,
        document_id: ObjectId,
        embedding: List[float],
        embedding_field: str = "embedding",
    ) -> None:
        """Save an embedding to a document.

        Args:
            collection: The collection name.
            document_id: The ObjectId of the document.
            embedding: A list of floats representing the embedding.
            embedding_field: The field name to store the embedding.
        """
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
            print(f"Error saving embedding to {collection}: {e}")

    async def clear_cache(self) -> None:
        """Clear the internal cache."""
        self.cache.clear()
        logger.info("Cache cleared.")

    # zmongo.py
    # ... (rest of your code remains unchanged up to the end of the class)

    async def bulk_write(
            self, collection: str, operations: List[Union[UpdateOne, InsertOne, DeleteOne, ReplaceOne]]
    ) -> Union[Dict[str, Any], None]:
        """Perform a bulk write operation on a collection.

        Args:
            collection: The collection name.
            operations: A list of bulk write operations.

        Returns:
            A dictionary of operation results or an error.
        """
        if not operations:
            return {
                "inserted_count": 0,
                "matched_count": 0,
                "modified_count": 0,
                "deleted_count": 0,
                "upserted_count": 0,
                "acknowledged": True,
            }

        try:
            result = await self.db[collection].bulk_write(operations)
            return {
                "inserted_count": getattr(result, "inserted_count", 0),
                "matched_count": getattr(result, "matched_count", 0),
                "modified_count": getattr(result, "modified_count", 0),
                "deleted_count": getattr(result, "deleted_count", 0),
                "upserted_count": getattr(result, "upserted_count", 0),
                "acknowledged": getattr(result, "acknowledged", True),
            }
        except BulkWriteError as e:
            logger.error(f"BulkWriteError during bulk_write: {e.details}")
            return {"error": e.details}
        except PyMongoError as e:
            logger.error(f"PyMongoError during bulk_write: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during bulk_write: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the MongoDB connections."""
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")

    async def list_collections(self) -> List[str]:
        """List all collection names in the database."""
        try:
            return await self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def count_documents(self, collection: str) -> int:
        """Return estimated number of documents in a collection."""
        try:
            return await self.db[collection].estimated_document_count()
        except Exception as e:
            logger.error(f"Error counting documents in '{collection}': {e}")
            return 0

    async def get_document_by_id(self, collection: str, document_id: Union[str, ObjectId]) -> Optional[dict]:
        """Retrieve a document by its ObjectId."""
        try:
            if isinstance(document_id, str):
                document_id = ObjectId(document_id)
            doc = await self.db[collection].find_one({"_id": document_id})
            return self.serialize_document(doc) if doc else None
        except Exception as e:
            logger.error(f"Failed to retrieve document by ID from '{collection}': {e}")
            return None