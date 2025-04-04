# Example usage or testing entry point
import asyncio

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

if __name__ == "__main__":
    async def main():
        mongo_repo = ZMongo()

        try:
            # Example: Insert a document
            document = {
                "name": "John Doe",
                "role": "Developer",
                "skills": ["Python", "MongoDB"],
                "score": 12
            }
            insert_result = await mongo_repo.insert_document("test_collection", document)
            print(f"Inserted Document ID: {insert_result.inserted_id}")

            # Example: Fetch the inserted document
            fetched_document = await mongo_repo.find_document("test_collection", {"_id": insert_result.inserted_id})
            print("Fetched Document:", fetched_document)

            # Example: Clean up after testing
            await mongo_repo.delete_document("test_collection", {"_id": insert_result.inserted_id})
            print("Deleted the test document.")

        finally:
            await mongo_repo.close()


    asyncio.run(main())