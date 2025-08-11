import asyncio
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zmongo_embedder import ZMongoEmbedder
from zmongo_toolbag.zmongo_retriever import ZRetriever
from bson import ObjectId
from pydantic import BaseModel, Field
from llama_cpp import Llama

MODEL_PATH = r"C:\Users\iriye\resources\models\mistral-7b-instruct-v0.1.Q4_0.gguf"
CHUNK_SIZE = 512
COLLECTION = "rag_docs"

zm = ZMongo()
embedder = ZMongoEmbedder(collection=COLLECTION, repository=zm, model_path=MODEL_PATH)
retriever = ZRetriever(
    collection=COLLECTION,
    repository=zm,
    model_path=MODEL_PATH,
    chunk_size=CHUNK_SIZE,
    use_embedding=True,
)

class RAGDoc(BaseModel):
    doc_id: str = Field(..., alias="_id")
    title: str
    content: str
    meta: dict = Field(default_factory=dict)
    embedding: list = Field(default=None, alias="_embedding")

    model_config = {
        "populate_by_name": True
    }

# Usage
doc = RAGDoc(
    _id=str(ObjectId()),
    title="My First Doc",
    content="ZMongo is a fast, cache-first MongoDB library.",
    meta={"source": "manual"}
)


async def ingest_document(doc: RAGDoc):
    res = await zm.insert_document(COLLECTION, doc.model_dump(by_alias=True))
    assert res.success
    await embedder.embed_and_store(ObjectId(doc.doc_id), doc.content, embedding_field="_embedding")
    print(f"Document {doc.doc_id} ingested and embedded.")

async def retrieve_relevant(query: str, top_k: int = 3):
    # 1. Embed the query (always use embedder for query!)
    query_embeds = await embedder.embed_text(query)
    query_embedding = query_embeds[0]
    # 2. Use retriever to get top-k Documents
    # We'll assume retriever has a method to get docs by query embedding (adjust if your method signature differs)
    results = await retriever.retrieve_by_embedding(query_embedding, top_k=top_k)
    # Each item is a langchain.schema.Document with .page_content and .metadata
    return results

async def generate_rag_answer(user_query: str):
    docs = await retrieve_relevant(user_query, top_k=3)
    # Use the text from Documents, not from embeddings
    context = "\n\n".join(doc.page_content for doc in docs)
    llama = Llama(model_path=MODEL_PATH)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response = llama(prompt, max_tokens=128)
    return response['choices'][0]['text']

# --- Example usage ---
async def main():
    this_doc = RAGDoc(
        _id=str(ObjectId()),
        title="My First Doc",
        content="ZMongo is a fast, cache-first MongoDB library.",
        meta={"source": "manual"}
    )
    await ingest_document(this_doc)
    answer = await generate_rag_answer("What makes ZMongo unique?")
    print("RAG Answer:", answer)

if __name__ == "__main__":
    asyncio.run(main())
