import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from bson.objectid import ObjectId
from google.generativeai.types.text_types import EmbeddingDict
from pymongo.errors import OperationFailure, DuplicateKeyError
from dotenv import load_dotenv
from langchain.schema import Document
import motor.motor_asyncio
import google.generativeai as genai
import numpy as np

# User's provided path for environment variables
load_dotenv(Path.home() / "resources" / ".env_local")


# --- Prerequisite Classes ---

class SafeResult:
    """A predictable, serializable wrapper for all MongoDB operation results."""

    def __init__(self, data: Any = None, *, success: bool, error: Optional[str] = None,
                 original_exc: Optional[Exception] = None):
        self.success = success
        self.error = error
        self.data = self._convert_bson(data)
        self._original_exc = original_exc

    @staticmethod
    def _convert_bson(obj: Any) -> Any:
        if isinstance(obj, ObjectId): return str(obj)
        if isinstance(obj, dict): return {k: SafeResult._convert_bson(v) for k, v in obj.items()}
        if isinstance(obj, list): return [SafeResult._convert_bson(x) for x in obj]
        return obj

    @classmethod
    def ok(cls, data: Any = None) -> 'SafeResult':
        return cls(data=data, success=True)

    @classmethod
    def fail(cls, error: str, data: Any = None, exc: Optional[Exception] = None) -> 'SafeResult':
        return cls(data=data, success=False, error=error, original_exc=exc)

    def original(self) -> Union[Exception, None]:
        return self._original_exc

    def __repr__(self):
        return f"SafeResult(success={self.success}, error='{self.error}', data_preview='{str(self.data)[:100]}...')"


class ZMongo:
    """Base class for asynchronous MongoDB operations."""

    def __init__(self, db: motor.motor_asyncio.AsyncIOMotorDatabase):
        self.db = db

    async def close(self):
        if self.db is not None and self.db.client is not None:
            self.db.client.close()
            logging.info("MongoDB connection closed.")

    async def find_document(self, collection: str, query: Dict) -> SafeResult:
        try:
            doc = await self.db[collection].find_one(query)
            return SafeResult.ok(doc)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def insert_document(self, collection: str, document: Dict) -> SafeResult:
        """Inserts a document and returns a SafeResult containing the raw ObjectId."""
        try:
            res = await self.db[collection].insert_one(document)
            return SafeResult.ok({"inserted_id": res.inserted_id})
        except DuplicateKeyError as e:
            logging.warning(f"Attempted to insert a document with a duplicate key: {e.details}")
            raise e
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def update_document(self, collection: str, query: Dict, update_data: Dict, *,
                              upsert: bool = False) -> SafeResult:
        try:
            update_dict = update_data
            if not any(k.startswith("$") for k in update_dict):
                update_dict = {"$set": update_dict}
            res = await self.db[collection].update_one(query, update_dict, upsert=upsert)
            return SafeResult.ok({"modified_count": res.modified_count, "upserted_id": res.upserted_id})
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def aggregate(self, collection: str, pipeline: List[Dict], *, limit: int = 1000) -> SafeResult:
        try:
            cursor = self.db[collection].aggregate(pipeline)
            return SafeResult.ok(await cursor.to_list(length=limit))
        except OperationFailure as e:
            raise e
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def find_documents(self, collection: str, query: Dict, *, limit: int = 10000,
                             sort: Optional[List[tuple[str, int]]] = None) -> SafeResult:
        try:
            cursor = self.db[collection].find(query)
            if sort:
                cursor = cursor.sort(sort)
            docs = await cursor.to_list(length=limit)
            return SafeResult.ok(docs)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def delete_document(self, collection: str, query: Dict) -> SafeResult:
        try:
            res = await self.db[collection].delete_one(query)
            return SafeResult.ok({"deleted_count": res.deleted_count})
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)


class ZMongoAtlas(ZMongo):
    """Enhanced client for MongoDB Atlas, adding vector search capabilities."""

    # --- FINAL FIX ---
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors correctly."""
        vec1, vec2 = np.asarray(v1, dtype=np.float32), np.asarray(v2, dtype=np.float32)
        if vec1.shape != vec2.shape or vec1.size == 0: return 0.0
        norm_v1, norm_v2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        # The incorrect `int()` cast that caused the divide-by-zero error is now removed.
        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    async def vector_search(self, collection_name: str, query_vector: List[float], index_name: str,
                            embedding_field: str, top_k: int, num_candidates: Optional[int] = None) -> SafeResult:
        pipeline = [
            {"$vectorSearch": {
                "index": index_name, "path": embedding_field, "queryVector": query_vector,
                "numCandidates": num_candidates or top_k * 15, "limit": top_k,
            }},
            {"$project": {
                "retrieval_score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"
            }}
        ]
        try:
            return await self.aggregate(collection_name, pipeline, limit=top_k)
        except OperationFailure as e:
            if e.code == 31082 or 'SearchNotEnabled' in str(e):
                logging.warning("Falling back to manual similarity search for local testing.")
                find_res = await self.find_documents(collection_name, {embedding_field: {"$exists": True}})
                if not find_res.success: return find_res
                scored_docs = []
                for doc in find_res.data:
                    max_sim = max(
                        (self._cosine_similarity(query_vector, chunk) for chunk in doc.get(embedding_field, [])),
                        default=-1.0
                    )
                    logging.info(f"Manual search score for doc ID {doc.get('_id')}: {max_sim:.4f}")
                    scored_docs.append({"retrieval_score": max_sim, "document": doc})
                scored_docs.sort(key=lambda x: x["retrieval_score"], reverse=True)
                return SafeResult.ok(scored_docs[:top_k])
            else:
                return SafeResult.fail(str(e), data=e.details, exc=e)


class ZMongoEmbedder:
    """Generates and caches text embeddings using Google Gemini."""

    def __init__(self, repository: ZMongo, gemini_api_key: str):
        self.repository = repository
        self.embedding_model_name = "models/embedding-001"
        genai.configure(api_key=gemini_api_key)

    def _split_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
        if not text: return []
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def _embed_content_sync(self, content: str) -> EmbeddingDict:
        return genai.embed_content(
            model=self.embedding_model_name,
            content=content,
            task_type="RETRIEVAL_DOCUMENT"
        )

    async def _get_embedding_from_api(self, chunk: str) -> List[float]:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._embed_content_sync, chunk)
            return result['embedding']
        except Exception as e:
            logging.error(f"Error generating embedding for chunk: {e}")
            raise

    async def embed_text(self, text: str) -> List[List[float]]:
        if not text: return []
        chunks = self._split_chunks(text)
        embeddings = [await self._get_embedding_from_api(chunk) for chunk in chunks]
        return embeddings


class ZMongoRetriever:
    """A LangChain-style retriever for performing semantic search."""

    def __init__(self, repository: ZMongoAtlas, embedder: ZMongoEmbedder, collection_name: str,
                 embedding_field: str = "embeddings", content_field: str = "text", top_k: int = 10,
                 vector_search_index_name: str = "vector_index", similarity_threshold: float = 0.70):
        self.repository = repository
        self.embedder = embedder
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.content_field = content_field
        self.top_k = top_k
        self.vector_search_index_name = vector_search_index_name
        self.similarity_threshold = similarity_threshold

    def _filter_and_format_results(self, items: List[dict]) -> List[Document]:
        final_docs = []
        for item in items:
            score = item.get("retrieval_score", 0.0)
            if score >= self.similarity_threshold:
                doc = item.get("document", item)
                content = doc.get(self.content_field, "")
                metadata = {k: v for k, v in doc.items() if k not in [self.embedding_field, self.content_field, '_id']}
                metadata.update({'source_document_id': str(doc.get('_id')), 'retrieval_score': score})
                final_docs.append(Document(page_content=content, metadata=metadata))
        return final_docs

    async def get_relevant_documents(self, query: str) -> List[Document]:
        embeddings = await self.embedder.embed_text(query)
        if not embeddings: return []
        query_embedding = embeddings[0]

        res = await self.repository.vector_search(
            self.collection_name, query_embedding, self.vector_search_index_name, self.embedding_field, self.top_k
        )
        if not res.success:
            logging.error(f"Vector search failed: {res.error}")
            if res.original():
                raise res.original()
            return []

        return self._filter_and_format_results(res.data)


# --- Main Service Class ---

class ZMongoService:
    """
    A high-level service class that provides a simplified interface for
    interacting with MongoDB, handling text embeddings, and performing
    semantic searches.
    """

    def __init__(self, mongo_uri: str, db_name: str, gemini_api_key: str):
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.db = self.client[db_name]
            self.repository = ZMongoAtlas(self.db)
            self.embedder = ZMongoEmbedder(self.repository, gemini_api_key)
            logging.info(f"ZMongoService initialized for database '{db_name}'.")
        except Exception as e:
            logging.error(f"Failed to initialize ZMongoService: {e}")
            raise

    def get_retriever(self, collection_name: str, **kwargs) -> ZMongoRetriever:
        return ZMongoRetriever(
            repository=self.repository,
            embedder=self.embedder,
            collection_name=collection_name,
            **kwargs
        )

    async def add_and_embed(
            self,
            collection_name: str,
            document: Dict[str, Any],
            text_field: str,
            embedding_field: str = "embeddings"
    ) -> SafeResult:
        """
        Checks for duplicate content before inserting. If the content is unique,
        it inserts the document and adds embeddings.
        """
        text_to_embed = document.get(text_field)
        if not text_to_embed or not isinstance(text_to_embed, str):
            return SafeResult.fail(f"Document must contain a non-empty string in the '{text_field}' field.")

        # First, check if a document with this content already exists.
        existing_doc_res = await self.repository.find_document(collection_name, {text_field: text_to_embed})
        if existing_doc_res.success and existing_doc_res.data:
            existing_id = existing_doc_res.data.get('_id')
            logging.warning(f"Document with the same content already exists with ID {existing_id}. Skipping.")
            return SafeResult.ok(
                {"message": "Document already exists.", "inserted_id": str(existing_id), "existed": True})

        # If no duplicate is found, proceed with insertion.
        try:
            insert_res = await self.repository.insert_document(collection_name, document)
        except DuplicateKeyError:
            # This is a race condition fallback, the check above is the primary path.
            return await self.add_and_embed(collection_name, document, text_field, embedding_field)

        if not insert_res.success:
            return insert_res

        doc_id_obj = insert_res.data.get("inserted_id")
        if not doc_id_obj:
            return SafeResult.fail("Failed to retrieve document ID after insertion.")

        try:
            embeddings = await self.embedder.embed_text(text_to_embed)
            if not embeddings:
                await self.repository.delete_document(collection_name, {"_id": doc_id_obj})
                return SafeResult.fail("Embedding generation returned no vectors.")
        except Exception as e:
            logging.error(f"Embedding generation failed for doc_id {doc_id_obj}: {e}")
            await self.repository.delete_document(collection_name, {"_id": doc_id_obj})
            return SafeResult.fail(f"An exception occurred during embedding generation: {e}", exc=e)

        update_res = await self.repository.update_document(
            collection_name,
            {"_id": doc_id_obj},
            {"$set": {embedding_field: embeddings}}
        )

        if not update_res.success or update_res.data.get("modified_count", 0) == 0:
            logging.error(f"Failed to embed document with ID {doc_id_obj}.")
            return SafeResult.fail(f"Failed to save embeddings for document ID {doc_id_obj}.")

        logging.info(f"Successfully inserted and embedded document {doc_id_obj}.")
        return SafeResult.ok({"inserted_id": str(doc_id_obj), "existed": False})

    async def search(self, collection_name: str, query_text: str, **kwargs) -> List[Document]:
        retriever = self.get_retriever(collection_name, **kwargs)
        return await retriever.get_relevant_documents(query_text)

    async def close_connection(self):
        await self.repository.close()


async def main():
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not all([MONGO_URI, MONGO_DATABASE_NAME, GEMINI_API_KEY]):
        print("Please set MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY in your .env file.")
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    service = ZMongoService(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DATABASE_NAME,
        gemini_api_key=GEMINI_API_KEY
    )

    collection = "my_knowledge_base"
    content_field = "content"

    try:
        print("\n--- Attempting to add a document ---")
        doc_to_add = {"title": "Python Basics",
                      "content": "Python is a versatile and popular programming language known for its simplicity."}
        res = await service.add_and_embed(collection, doc_to_add, text_field=content_field)
        if res.success:
            if res.data.get("existed"):
                print(f"Document already existed with ID: {res.data['inserted_id']}")
            else:
                print(f"Successfully added new document with ID: {res.data['inserted_id']}")
        else:
            print(f"Failed to add document: {res.error}")

        print("\n--- Attempting to add the same document again ---")
        res_again = await service.add_and_embed(collection, doc_to_add, text_field=content_field)
        if res_again.success:
            if res_again.data.get("existed"):
                print(f"Document already existed with ID: {res_again.data['inserted_id']}")
            else:
                print(f"Successfully added new document with ID: {res_again.data['inserted_id']}")
        else:
            print(f"Failed to add document: {res_again.error}")

        print("\n--- Performing Search ---")
        query = "What is a good language for beginners?"

        search_results = await service.search(
            collection,
            query,
            content_field=content_field
        )

        print(f"\nFound {len(search_results)} results for the query: '{query}'")
        for doc in search_results:
            print(f"\n  Score: {doc.metadata.get('retrieval_score'):.4f}")
            print(f"  Content: {doc.page_content}")
            print(f"  Source Title: {doc.metadata.get('title')}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n--- Closing Connection ---")
        await service.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
