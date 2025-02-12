# repository_singleton.py

import asyncio
from zmongo_hyper_speed import ZMongoHyperSpeed

class RepositorySingleton:
    _instance = None

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = ZMongoHyperSpeed()
            await cls._instance.initialize()
        return cls._instance

# Usage in main.py

async def run_all_tests():
    repository = await RepositorySingleton.get_instance()
    # Proceed with operations using 'repository'
