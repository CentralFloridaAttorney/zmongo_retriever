import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import Optional, List, Any, Union, Dict

from bson.objectid import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
DEFAULT_QUERY_LIMIT: int = int(os.getenv("DEFAULT_QUERY_LIMIT", "100"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZMongo:

    def __init__(self) -> None:
        self.MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
        self.MONGO_DB_NAME: str = os.getenv("MONGO_DATABASE_NAME", "documents")

        if not os.getenv("MONGO_URI"):
            logger.warning("⚠️ MONGO_URI is not set in .env. Defaulting to 'mongodb://127.0.0.1:27017'")
        if not os.getenv("MONGO_DATABASE_NAME"):
            logger.warning("❌ MONGO_DATABASE_NAME is not set in .env. Defaulting to 'documents'")

        self.mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(self.MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[self.MONGO_DB_NAME]
        self.cache: Dict[str, Dict[str, dict]] = defaultdict(dict)

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def _generate_cache_key(query: dict) -> str:
        query_json = json.dumps(query, sort_keys=True, default=str)
        return hashlib.sha256(query_json.encode("utf-8")).hexdigest()

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        limit = kwargs.get("limit", DEFAULT_QUERY_LIMIT)
        cursor = self.db[collection].find(filter=query, **kwargs)
        return await cursor.to_list(length=limit)

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        normalized = self._normalize_collection_name(collection)
        cache_key = self._generate_cache_key(query)

        if cache_key in self.cache[normalized]:
            return self.cache[normalized][cache_key]

        document = await self.db[collection].find_one(filter=query)
        if document:
            self.cache[normalized][cache_key] = document
            return document
        return None

    async def insert_document(self, collection: str, document: dict, use_cache: bool = True) -> Optional[dict]:
        try:
            result = await self.db[collection].insert_one(document)
            document["_id"] = result.inserted_id

            if use_cache:
                normalized = self._normalize_collection_name(collection)
                cache_key = self._generate_cache_key({"_id": str(result.inserted_id)})
                self.cache[normalized][cache_key] = document

            return document
        except Exception as e:
            logger.error(f"Error inserting document into '{collection}': {e}")
            return None

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False, array_filters: Optional[List[dict]] = None):
        try:
            result = await self.db[collection].update_one(
                filter=query, update=update_data, upsert=upsert, array_filters=array_filters
            )
            if result.matched_count > 0 or result.upserted_id:
                updated_doc = await self.db[collection].find_one(filter=query)
                if updated_doc:
                    normalized = self._normalize_collection_name(collection)
                    cache_key = self._generate_cache_key(query)
                    self.cache[normalized][cache_key] = updated_doc
            return result
        except Exception as e:
            logger.error(f"Error updating document in {collection}: {e}")
            raise

    async def delete_document(self, collection: str, query: dict) -> Any:
        result = await self.db[collection].delete_one(filter=query)
        if result.deleted_count:
            normalized = self._normalize_collection_name(collection)
            cache_key = self._generate_cache_key(query)
            self.cache[normalized].pop(cache_key, None)
        return result

    async def delete_all_documents(self, collection: str) -> int:
        result = await self.db[collection].delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents from '{collection}'")
        return result.deleted_count

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
            return await cursor.to_list(length=sample_size)
        except Exception as e:
            logger.error(f"Failed to get sample documents from '{collection}': {e}")
            return []

    async def text_search(self, collection: str, search_text: str, limit: int = 10) -> List[dict]:
        try:
            cursor = self.db[collection].find({"$text": {"$search": search_text}}).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Text search failed in '{collection}': {e}")
            return []

    async def save_embedding(
        self,
        collection: str,
        document_id: ObjectId,
        embedding: List[float],
        embedding_field: str = "embedding",
    ) -> None:
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

    async def list_collections(self) -> List[str]:
        try:
            return await self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def count_documents(self, collection: str) -> int:
        try:
            return await self.db[collection].estimated_document_count()
        except Exception as e:
            logger.error(f"Error counting documents in '{collection}': {e}")
            return 0

    async def get_document_by_id(self, collection: str, document_id: Union[str, ObjectId]) -> Optional[dict]:
        try:
            if isinstance(document_id, str):
                document_id = ObjectId(document_id)
            doc = await self.db[collection].find_one({"_id": document_id})
            return self.serialize_document(doc) if doc else None
        except Exception as e:
            logger.error(f"Failed to retrieve document by ID from '{collection}': {e}")
            return None

    @staticmethod
    def serialize_document(document: dict) -> dict:
        return document

    async def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Cache cleared.")

    async def bulk_write(self, collection: str, operations: List[dict]) -> Dict[str, Any]:
        if not operations:
            return {
                "inserted_count": 0,
                "matched_count": 0,
                "modified_count": 0,
                "deleted_count": 0,
                "upserted_count": 0,
                "acknowledged": True,
            }

        inserted_count = 0
        matched_count = 0
        modified_count = 0
        deleted_count = 0
        upserted_count = 0

        for op in operations:
            try:
                action = op.get("operation")
                if action == "insert":
                    result = await self.db[collection].insert_one(op["document"])
                    if result.inserted_id:
                        inserted_count += 1

                elif action == "update":
                    result = await self.db[collection].update_one(
                        op["filter"], op["update"], upsert=op.get("upsert", False)
                    )
                    matched_count += result.matched_count
                    modified_count += result.modified_count
                    if result.upserted_id:
                        upserted_count += 1

                elif action == "delete":
                    result = await self.db[collection].delete_one(op["filter"])
                    deleted_count += result.deleted_count

                else:
                    logger.warning(f"Unknown operation type: {action}")
            except Exception as e:
                logger.error(f"Failed {action} operation in bulk_write: {e}")

        return {
            "inserted_count": inserted_count,
            "matched_count": matched_count,
            "modified_count": modified_count,
            "deleted_count": deleted_count,
            "upserted_count": upserted_count,
            "acknowledged": True,
        }

    async def close(self) -> None:
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")
