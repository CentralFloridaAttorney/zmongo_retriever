from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# llama-cpp (optional import)
try:  # pragma: no cover
    from llama_cpp import Llama  # type: ignore
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore

try:  # pragma: no cover
    from bson import ObjectId  # type: ignore
except ImportError:  # pragma: no cover
    ObjectId = None

# --- Direct imports assuming a package structure ---
from zmongo_toolbag.data_processing import SafeResult
from zmongo_toolbag.zmongo import ZMongo

# Optional local env files
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------
EMBEDDING_STYLE_RETRIEVAL_QUERY = "retrieval_query"
EMBEDDING_STYLE_RETRIEVAL_DOCUMENT = "retrieval_document"
CHUNK_STYLE_PARAGRAPH = "paragraph"
CHUNK_STYLE_SENTENCE = "sentence"
CHUNK_STYLE_FIXED = "fixed"

__all__ = [
    "ZEmbedder", "EMBEDDING_STYLE_RETRIEVAL_QUERY", "EMBEDDING_STYLE_RETRIEVAL_DOCUMENT",
    "CHUNK_STYLE_PARAGRAPH", "CHUNK_STYLE_SENTENCE", "CHUNK_STYLE_FIXED",
]


# ---------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------
def _paragraph_split(text: str) -> List[str]:
    if not text: return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    if not text: return []
    if size <= overlap: raise ValueError("Chunk size must be greater than overlap.")

    doc = text.split()  # Split by whitespace for token-like units
    chunks = []
    start = 0
    while start < len(doc):
        end = start + size
        chunk_words = doc[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(doc):
            break
        start += (size - overlap)
    return chunks


def _chunk_text(text: str, *, chunk_style: str, chunk_size: int, overlap: int) -> List[str]:
    """Dispatches to the correct chunking function based on style."""
    if chunk_style == CHUNK_STYLE_FIXED:
        return _sliding_window(text, size=chunk_size, overlap=overlap)
    if chunk_style == CHUNK_STYLE_PARAGRAPH:
        return _paragraph_split(text)
    # Defaulting to paragraph style if unspecified
    return _paragraph_split(text)


# ---------------------------------------------------------------------
# ZEmbedder Class
# ---------------------------------------------------------------------
class ZEmbedder:
    def __init__(
            self, *, repository: Optional[ZMongo] = None,
            model_path: Optional[str] = None, n_ctx: int = 2048
    ) -> None:
        self.repo = repository or ZMongo()
        self._owns_repo = repository is None

        if Llama is None:
            raise ImportError("`pip install llama-cpp-python` is required to use ZEmbedder.")

        env_path = None
        env_var_path = os.getenv("EMBEDDING_MODEL_PATH")
        if env_var_path:
            env_path = os.path.join(Path.home(), env_var_path)

        raw_path = model_path or env_path
        if not raw_path:
            raise FileNotFoundError("model_path not provided and EMBEDDING_MODEL_PATH environment variable is not set.")

        self.model_path = str(Path(raw_path).expanduser().resolve())
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Embedding model not found at: {self.model_path}")

        logger.info("Loading Llama embedding model from: %s", self.model_path)
        logger.info("Using context size (n_ctx): %d", n_ctx)
        self.model = Llama(
            model_path=self.model_path, embedding=True,
            verbose=False, n_ctx=n_ctx, n_gpu_layers=-1
        )
        logger.info("Llama model loaded.")

    def close(self) -> None:
        if self._owns_repo:
            self.repo.close()

    async def _embed_batch(self, texts: List[str]) -> SafeResult:
        if not texts:
            return SafeResult.ok([])
        try:
            loop = asyncio.get_running_loop()
            vectors = await loop.run_in_executor(None, self.model.embed, texts)
            if not vectors or not all(isinstance(v, list) for v in vectors):
                return SafeResult.fail("Embedding process returned malformed data.")
            return SafeResult.ok(vectors)
        except Exception as e:
            logger.error("Embedding call failed in llama-cpp: %s", e)
            return SafeResult.fail("Embedding call failed in llama-cpp", exc=e)

    def _build_payload(self, *, text: str, vectors: List[List[float]], style: str, from_cache: bool,
                       **kwargs: Any) -> dict:
        return {
            "embedding_style": style, "text": text, "vectors": vectors,
            "vectors_count": len(vectors),
            "dimensionality": len(vectors[0]) if vectors and vectors[0] else 0,
            "from_cache": from_cache, "skipped_compute": from_cache, **kwargs
        }

    async def _process_document_embedding(self, text: Optional[str], collection: str, doc_id: Any, field: str,
                                          text_field: str, chunk_style: str, chunk_size: int, overlap: int,
                                          skip: bool) -> SafeResult:
        meta = {"collection": collection, "document_id": str(doc_id), "embedding_field": field,
                "chunk_style": chunk_style}

        if skip:
            find_res = await self.repo.find_document(collection, {"_id": doc_id})
            if find_res.success and find_res.data:
                cached_doc = find_res.data
                existing_vectors = cached_doc.get(field)
                if isinstance(existing_vectors, list) and existing_vectors:
                    final_text = text if text is not None else cached_doc.get(text_field, "")
                    payload = self._build_payload(text=final_text, vectors=existing_vectors,
                                                  style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, from_cache=True, **meta)
                    return SafeResult.ok(payload)
                if text is None:
                    text = cached_doc.get(text_field)
            elif not find_res.success:
                return SafeResult.fail(f"Failed to check for existing document: {find_res.error}",
                                       exc=find_res.original())

        if not text:
            return SafeResult.fail("Document text not provided and could not be found in source document.")

        chunks = _chunk_text(text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            payload = self._build_payload(text=text, vectors=[], style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                                          from_cache=False, **meta)
            return SafeResult.ok(payload)

        embed_res = await self._embed_batch(chunks)
        if not embed_res.success:
            return embed_res
        vectors = embed_res.data

        save_res = await self.repo.update_document(collection, {"_id": doc_id}, {"$set": {field: vectors}})
        if not save_res.success:
            logger.error("Failed to save embeddings to %s/%s: %s", collection, doc_id, save_res.error)
            meta["save_error"] = save_res.error

        final_payload = self._build_payload(text=text, vectors=vectors, style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
                                            from_cache=False, **meta)
        return SafeResult.ok(final_payload)

    async def get_embedding(
            self, text: Optional[str] = None, *,
            embedding_style: str = EMBEDDING_STYLE_RETRIEVAL_QUERY,
            collection: Optional[str] = None, document_id: Any = None,
            embedding_field: Optional[str] = None, text_field: str = "text",
            chunk_style: str = CHUNK_STYLE_PARAGRAPH, chunk_size: int = 512, overlap: int = 50,
            skip_if_present: bool = True, as_safe_result: Optional[bool] = None,
    ) -> Any:
        style = (embedding_style or EMBEDDING_STYLE_RETRIEVAL_QUERY).lower()
        if as_safe_result is None:
            as_safe_result = (style == EMBEDDING_STYLE_RETRIEVAL_DOCUMENT)

        if style == EMBEDDING_STYLE_RETRIEVAL_QUERY:
            if not text:
                return SafeResult.fail("Query text cannot be empty.") if as_safe_result else []
            embed_res = await self._embed_batch([text])
            if not embed_res.success:
                return embed_res if as_safe_result else []
            vectors = embed_res.data
            if as_safe_result:
                payload = self._build_payload(text=text, vectors=vectors, style=style, from_cache=False)
                return SafeResult.ok(payload)
            return vectors

        has_persistence_context = bool(collection and document_id is not None and embedding_field)
        if not has_persistence_context:
            return SafeResult.fail("Document embedding requires collection, document_id, and embedding_field.")

        result = await self._process_document_embedding(
            text, collection, document_id, embedding_field, text_field,
            chunk_style, chunk_size, overlap, skip_if_present
        )

        if as_safe_result:
            return result
        return result.data.get("vectors", []) if result.success else []


# ---------------------------------------------------------------------
# Self-Contained Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":

    LONG_DEMO_TEXT = """The history of computing began long before the digital age. Early mechanical devices, like the abacus, were used for calculation for thousands of years. The true precursor to the modern computer, however, was Charles Babbage's Analytical Engine in the 19th century. Though never fully built in his lifetime, its design included an arithmetic logic unit, control flow in the form of conditional branching and loops, and integrated memory, making it the first design for a general-purpose, Turing-complete computer.

The electromechanical era followed, with devices like the Atanasoff-Berry Computer and the Harvard Mark I paving the way. The major breakthrough came with the advent of fully electronic computers during World War II. ENIAC (Electronic Numerical Integrator and Computer) was a colossal machine that used vacuum tubes instead of mechanical relays, increasing calculation speed by orders of magnitude. It was programmable, but required manual rewiring to change its operations, a tedious process that highlighted the need for a more flexible architecture.

This need was met by the von Neumann architecture, which introduced the concept of the stored-program computer. This design, where program instructions and data are stored in the same read-write memory, remains the fundamental basis for nearly all modern computers. The invention of the transistor in 1947, and later the integrated circuit, allowed computers to become smaller, faster, cheaper, and more reliable, moving from room-sized behemoths to machines that could fit on a desk.

The final leap was the microprocessor, which placed an entire central processing unit (CPU) onto a single integrated circuit chip. This innovation fueled the personal computer revolution of the 1970s and 80s, bringing computing power to individuals and small businesses. The subsequent development of graphical user interfaces and the global connectivity of the internet transformed the computer from a specialized tool for experts into an indispensable part of modern life for billions of people."""


    async def _main_demo():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        if not os.getenv("EMBEDDING_MODEL_PATH"):
            print("FATAL: Set the EMBEDDING_MODEL_PATH environment variable before running.")
            return
        if ObjectId is None:
            print("FATAL: `pip install bson` is required to run the demo.")
            return

        embedder = ZEmbedder(n_ctx=2048)

        coll = "zembedder_main_demo"
        doc_id = ObjectId("61c0c55e0000000000000003")

        print(f"\n--- Using collection '{coll}' and fixed document_id '{doc_id}' ---")

        print("\n--- STEP 1: Processing and chunking a long document (first time) ---")
        await embedder.repo.delete_document(coll, {"_id": doc_id})
        await embedder.repo.insert_document(coll, {"_id": doc_id, "source_text": LONG_DEMO_TEXT})

        res1 = await embedder.get_embedding(
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            collection=coll, document_id=doc_id, embedding_field="embeddings", text_field="source_text",
            chunk_style=CHUNK_STYLE_FIXED, chunk_size=400, overlap=40  # Using safe, smaller chunks
        )
        if res1.success:
            print(
                f"  => SUCCESS! From Cache: {res1.data.get('from_cache')}. Vectors Generated: {res1.data.get('vectors_count')}")
        else:
            print(f"  => FAILED: {res1.error}")

        print("\n--- STEP 2: Demonstrating the cache (second time) ---")
        res2 = await embedder.get_embedding(
            embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
            collection=coll, document_id=doc_id, embedding_field="embeddings"
        )
        if res2.success:
            print(
                f"  => SUCCESS! From Cache: {res2.data.get('from_cache')}. Vectors Found: {res2.data.get('vectors_count')}")
        else:
            print(f"  => FAILED: {res2.error}")

        print("\n--- STEP 3: Generating a simple query embedding ---")
        query_text = "What was the impact of the microprocessor?"
        query_vector = await embedder.get_embedding(text=query_text, as_safe_result=False)
        if query_vector and query_vector[0]:
            print(f"  => SUCCESS! Query vector created with dimensionality: {len(query_vector[0])}")
        else:
            print(f"  => FAILED to create query vector. Result: {query_vector}")

        embedder.close()
        print("\n--- Demo Complete ---")


    asyncio.run(_main_demo())

