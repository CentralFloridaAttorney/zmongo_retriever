import asyncio
import json
import logging
import os
from typing import Optional, List, Any, Union

from bson import ObjectId, json_util
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, InsertOne, DeleteOne, ReplaceOne

# Environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DATABASE_NAME")
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
        if not MONGO_URI or not MONGO_DB_NAME:
            raise ValueError("MONGO_URI and MONGO_DATABASE_NAME must be set in environment variables.")

        self.mongo_client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=200)
        self.db = self.mongo_client[MONGO_DB_NAME]

    @staticmethod
    def _normalize_collection_name(collection: str) -> str:
        return collection.strip().lower()

    @staticmethod
    def serialize_document(document: dict) -> dict:
        return json.loads(json_util.dumps(document)) if document else {}

    async def find_document(self, collection: str, query: dict) -> Optional[dict]:
        document = await self.db[collection].find_one(filter=query)
        return self.serialize_document(document) if document else None

    async def find_documents(self, collection: str, query: dict, **kwargs) -> List[dict]:
        return await self.db[collection].find(filter=query, **kwargs).to_list(
            length=kwargs.get('limit', DEFAULT_QUERY_LIMIT)
        )

    async def insert_document(self, collection: str, document: dict) -> Any:
        result = await self.db[collection].insert_one(document)
        document["_id"] = result.inserted_id
        return result

    async def update_document(self, collection: str, query: dict, update_data: dict, upsert: bool = False,
                              array_filters: Optional[List[dict]] = None) -> dict:
        try:
            result = await self.db[collection].update_one(
                filter=query, update=update_data, upsert=upsert, array_filters=array_filters
            )

            return {
                "matchedCount": result.matched_count,
                "modifiedCount": result.modified_count,
                "upsertedId": result.upserted_id
            }

        except Exception as e:
            logger.error(f"Error updating document in {collection}: {e}")
            return {}

    async def delete_document(self, collection: str, query: dict) -> Any:
        return await self.db[collection].delete_one(filter=query)

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
        await self.db[collection].update_one(
            {"_id": document_id}, {"$set": {embedding_field: embedding}}, upsert=True
        )

    async def bulk_write(self, collection: str,
                         operations: List[Union[UpdateOne, InsertOne, DeleteOne, ReplaceOne]]) -> None:
        if operations:
            await self.db[collection].bulk_write(operations)

    async def close(self) -> None:
        self.mongo_client.close()
        logger.info("MongoDB connection closed.")


# Example usage or testing entry point
if __name__ == "__main__":
    async def main():
        mongo_repo = ZMongo()

        try:
            document = {
                "name": "John Doe",
                "role": "Developer",
                "skills": ["Python", "MongoDB"],
                "score": 12
            }
            insert_result = await mongo_repo.insert_document("test_collection", document)
            print(f"Inserted Document ID: {insert_result.inserted_id}")

            fetched_document = await mongo_repo.find_document("test_collection", {"_id": insert_result.inserted_id})
            print("Fetched Document:", fetched_document)

            await mongo_repo.delete_document("test_collection", {"_id": insert_result.inserted_id})
            print("Deleted the test document.")

        finally:
            await mongo_repo.close()

    asyncio.run(main())