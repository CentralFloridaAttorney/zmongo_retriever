import time
import asyncio
from zmongo.BAK.zmongo_retriever import ZMongoRetriever
from zmongo.BAK.zmongo_embedder import ZMongoEmbedder
from dotenv import load_dotenv
import os

load_dotenv()

async def benchmark():
    retriever = ZMongoRetriever(
        overlap_prior_chunks=3,
        max_tokens_per_set=-1,
        chunk_size=512,
        use_embedding=True
    )

    sample_docs = retriever.collection.find({}, {"_id": 1}).limit(5)
    sample_object_ids = [str(doc["_id"]) for doc in sample_docs]

    if not sample_object_ids:
        print("No documents found in the collection.")
        return

    start_time = time.perf_counter()
    documents = retriever.invoke(
        object_ids=sample_object_ids,
        page_content_key=os.getenv("PAGE_CONTENT_KEY", "default_key")
    )
    end_time = time.perf_counter()

    retrieval_time = end_time - start_time
    num_documents = len(documents) if documents else 0

    print(f"Retrieved {num_documents} document chunks in {retrieval_time:.4f} seconds")

    embedder = ZMongoEmbedder(
        collection_name=os.getenv('DEFAULT_COLLECTION_NAME', 'default_collection'),
        page_content_keys=[os.getenv('PAGE_CONTENT_KEY', 'content_key')]
    )

    sample_text = "This is a sample text to demonstrate embedding."

    start_time = time.perf_counter()
    for document in documents:
        embedding_vector = await embedder.get_embedding(
            doc_id=document.metadata.get('document_id'),
            content_key=document.metadata.get('page_content_field'),
            text=sample_text
        )
    end_time = time.perf_counter()

    embedding_time = end_time - start_time

    print(f"Generated embeddings for {num_documents} chunks in {embedding_time:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(benchmark())
