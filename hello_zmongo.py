from zmongo_toolbag import ZMongo
import nest_asyncio
import asyncio

# Apply the nest_asyncio patch for running nested event loops in Jupyter
nest_asyncio.apply()

mongo = ZMongo()  # Initialize the ZMongo instance


async def main():
    # Insert a document asynchronously
    await mongo.insert_document("users", {"name": "Alice"})
    # Retrieve the inserted document
    doc = await mongo.find_document("users", {"name": "Alice"})
    return doc


# Retrieve and print the document using asyncio.run()
doc = asyncio.run(main())
print(doc)