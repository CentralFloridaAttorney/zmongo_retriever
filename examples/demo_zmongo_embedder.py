import asyncio
import logging
from bson import ObjectId
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zmongo_embedder import ZMongoEmbedder

logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the ZMongo repository
    zmongo = ZMongo()

    # Define the target collection
    collection_name = "documents"

    # Example document ID and text (you can replace these)
    document_id = ObjectId("67e5ba645f74ae46ad39929d")
    text_to_embed = "ZMongoRetriever enables fast, async retrieval with OpenAI embedding support."

    # Initialize the embedder
    embedder = ZMongoEmbedder(repository=zmongo, collection=collection_name)

    # Call embed and store
    try:
        await embedder.embed_and_store(document_id, text_to_embed)
        print(f"✅ Successfully embedded and stored text for document: {document_id}")
    except Exception as e:
        print(f"❌ Failed to embed and store: {e}")

    await zmongo.close()

if __name__ == "__main__":
    asyncio.run(main())
