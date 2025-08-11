import logging
from typing import List, Optional

import numpy as np
from pymongo.errors import OperationFailure

from zmongo_toolbag.zmongo import ZMongo, SafeResult

logger = logging.getLogger(__name__)


class ZMongoAtlas(ZMongo):
    """
    An enhanced MongoDB client for interacting with MongoDB Atlas clusters.
    Includes a fallback mechanism for local testing when Atlas Search is not enabled.
    """

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors."""
        vec1, vec2 = np.asarray(v1, dtype=np.float32), np.asarray(v2, dtype=np.float32)
        if vec1.shape != vec2.shape or vec1.size == 0: return 0.0
        norm_v1, norm_v2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    async def create_vector_search_index(
            self,
            collection_name: str,
            index_name: str,
            embedding_field: str,
            num_dimensions: int,
            similarity: str = "cosine"
    ) -> SafeResult:
        """Creates a MongoDB Atlas Vector Search index on a collection."""
        logger.info(f"Attempting to create vector search index '{index_name}' on '{collection_name}'...")
        try:
            collection = self.db[collection_name]
            index_definition = {
                "mappings": {"dynamic": True, "fields": {
                    embedding_field: {
                        "type": "vector", "dimension": num_dimensions, "similarity": similarity
                    }
                }}
            }
            await collection.create_search_index({"name": index_name, "definition": index_definition})
            msg = f"Successfully created vector search index '{index_name}'."
            return SafeResult.ok({"message": msg})
        except OperationFailure as e:
            if e.code == 31082 or 'SearchNotEnabled' in str(e):
                msg = "Skipped index creation: Vector Search is not enabled in this environment."
                logger.warning(msg)
                return SafeResult.ok({"message": msg})
            if "already exists" in str(e):
                msg = f"Vector search index '{index_name}' already exists."
                logger.warning(msg)
                return SafeResult.ok({"message": msg})
            return SafeResult.fail(str(e), data=e.details, exc=e)
        except Exception as e:
            return SafeResult.fail(str(e), exc=e)

    async def vector_search(
            self,
            collection_name: str,
            query_vector: List[float],
            index_name: str,
            embedding_field: str,
            top_k: int,
            num_candidates: Optional[int] = None
    ) -> SafeResult:
        """
        Performs a vector search using a specified Atlas Search index, with a
        manual fallback for local environments that do not support vector search.

        Args:
            collection_name (str): The collection to search within.
            query_vector (List[float]): The vector representation of the query.
            index_name (str): The name of the vector search index to use.
            embedding_field (str): The document field that the index is built on.
            top_k (int): The number of top matching documents to return.
            num_candidates (Optional[int]): The number of candidate documents to
                                            consider during the search for higher
                                            accuracy. If None, defaults to top_k * 15.

        Returns:
            SafeResult: A result object containing a list of matching documents,
                        each including a 'retrieval_score'.
        """
        pipeline = [
            {"$vectorSearch": {
                "index": index_name, "path": embedding_field, "queryVector": query_vector,
                "numCandidates": num_candidates or top_k * 15, "limit": top_k,
            }},
            {"$project": {
                "retrieval_score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"
            }}
        ]

        # --- FIX: The try...except block must wrap the database call ---
        try:
            # This is the line that actually executes the query and may fail.
            return await self.aggregate(collection_name, pipeline, limit=top_k)
        except OperationFailure as e:
            # Check for the specific error indicating Vector Search is not enabled.
            if e.code == 31082 or 'SearchNotEnabled' in str(e):
                logger.warning("Falling back to manual similarity search for local testing.")

                # Execute the manual fallback logic.
                find_res = await self.find_documents(collection_name, {embedding_field: {"$exists": True}})
                if not find_res.success: return find_res

                scored_docs = []
                for doc in find_res.data:
                    max_sim = max(
                        (self._cosine_similarity(query_vector, chunk) for chunk in doc.get(embedding_field, [])),
                        default=-1.0
                    )
                    scored_docs.append({"retrieval_score": max_sim, "document": doc})

                scored_docs.sort(key=lambda x: x["retrieval_score"], reverse=True)
                return SafeResult.ok(scored_docs[:top_k])
            else:
                # Re-raise any other type of OperationFailure.
                return SafeResult.fail(str(e), data=e.details, exc=e)