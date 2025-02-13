
# mongodb_with_mysql_logging.py

from motor.motor_asyncio import AsyncIOMotorClient
from zmongo_support.utils.async_mysql_integration import AsyncMySQLIntegration
from zmongo import zconstants


class MongoDBWithMySQLLogging:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.mysql_integration = AsyncMySQLIntegration(
            host=zconstants.DB_HOST,
            port=zconstants.DB_PORT,
            user=zconstants.DB_USER,
            password=zconstants.DB_PASSWORD,
            db=zconstants.DEFAULT_DATABASE
        )

    async def insert_document(self, document):
        try:
            result = await self.collection.insert_one(document)
            self.mysql_integration.log_to_mysql('insert', document)
            return result
        except Exception as e:
            print(f"Insert document error: {e}")

    async def fetch_document(self, query):
        try:
            result = await self.collection.find_one(query)
            return result
        except Exception as e:
            print(f"Fetch document error: {e}")

    async def update_document(self, query, update):
        try:
            result = await self.collection.update_one(query, {'$set': update})
            self.mysql_integration.log_to_mysql('update', update)
            return result
        except Exception as e:
            print(f"Update document error: {e}")

    async def delete_document(self, query):
        try:
            result = await self.collection.delete_one(query)
            self.mysql_integration.log_to_mysql('delete', query)
            return result
        except Exception as e:
            print(f"Delete document error: {e}")
