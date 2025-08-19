import asyncio

from zmongo import ZMongo
from zmongo_retriever import ZMongoRetriever


async def main():
    # Initialize MongoDB Repository (replace with your actual connection)
    repo = ZMongo()

    # Instantiate ZMongoRetriever
    retriever = ZMongoRetriever()

    # Define collection and document IDs (replace with actual IDs from your MongoDB)
    collection_name = 'documents'
    document_ids = ['67e5ba645f74ae46ad39929d', '67ef0bd71a349c7c108331a6']

    # Retrieve and process documents
    documents = retriever.invoke(collection=collection_name, object_ids=document_ids, page_content_key='text')

    # Display retrieved documents and their metadata
    for idx, doc_set in enumerate(documents):
        print(f"\nDocument Set {idx + 1}:")
        for doc in doc_set:
            print(f"Metadata: {doc.metadata}")
            print(f"Content: {doc.page_content[:200]}...\n")  # Display first 200 characters of each chunk

if __name__ == '__main__':
    asyncio.run(main())
