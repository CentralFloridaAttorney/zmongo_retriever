import asyncio
import logging
from bson import ObjectId
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.data_processing import DataProcessing

logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the ZMongo repository
    zmongo = ZMongo()

    # Define the target collection
    collection_name = "documents"
    the_document_oid = ObjectId("67e5ba645f74ae46ad39929d")
    page_content_key = "text"
    text_object = await zmongo.find_document(collection=collection_name, query={'_id': the_document_oid})
    text_to_embed = DataProcessing.get_value(json_data=text_object, key=page_content_key)

    # Initialize the embedder
    embedder = ZMongoEmbedder(repository=zmongo, collection=collection_name)

    # Call embed and store
    try:
        await embedder.embed_and_store(the_document_oid, text_to_embed)
        print(f"✅ Successfully embedded and stored text for document: {the_document_oid}")
    except Exception as e:
        print(f"❌ Failed to embed and store: {e}")

    await zmongo.close()

if __name__ == "__main__":
    asyncio.run(main())
