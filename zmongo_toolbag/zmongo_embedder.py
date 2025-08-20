import hashlib
import os
import logging
import asyncio
import re
from pathlib import Path
from typing import List, Optional, Dict, Literal
from itertools import chain

from bson.errors import InvalidId
from bson.objectid import ObjectId
from dotenv import load_dotenv
import google.generativeai as genai

from zmongo import ZMongo
from data_processing import SafeResult

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(Path.home() / "resources" / ".env_local")

# -------- Chunking style literals --------
CHUNK_STYLE_SENTENCE: Literal["sentence"] = "sentence"
CHUNK_STYLE_PARAGRAPH: Literal["paragraph"] = "paragraph"
CHUNK_STYLE_FIXED: Literal["fixed"] = "fixed"
ChunkStyle = Literal["fixed", "sentence", "paragraph"]

# -------- Embedding style literals (task types) --------
EmbeddingStyle = Literal[
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "CODE_RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
]


class ZMongoEmbedder:
    """
    Generates and stores text embeddings using the Google Gemini API,
    with a highly efficient, cache-first batching workflow.

    Supports:
      - chunk_style: 'fixed' | 'sentence' | 'paragraph'
      - embedding_style (Gemini task_type)
      - output_dimensionality (128..3072); auto-normalizes if != 3072
    """

    def __init__(self, collection: str, gemini_api_key: Optional[str] = None):
        self.repository = ZMongo()
        self.collection = collection
        # Model code per Gemini docs (stable)
        self.embedding_model_name = "gemini-embedding-001"

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        genai.configure(api_key=api_key)

    # -------------------- Chunking --------------------

    def _split_text_into_chunks(
        self,
        text: str,
        *,
        chunk_style: ChunkStyle = CHUNK_STYLE_FIXED,
        chunk_size: int = 1500,
        overlap: int = 150
    ) -> List[str]:
        """
        Splits text into chunks based on style.

        chunk_size:
          - fixed: max chars per chunk
          - sentence/paragraph: approx max chars per chunk (units are packed)
        overlap:
          - fixed: overlapping chars
          - sentence/paragraph: overlapping unit count
        """
        if not text:
            return []

        if chunk_style == CHUNK_STYLE_FIXED:
            return self._split_fixed(text, chunk_size=chunk_size, overlap_chars=overlap)
        elif chunk_style == CHUNK_STYLE_SENTENCE:
            units = self._split_sentences(text)
            return self._pack_units(units, target_chars=chunk_size, overlap_units=max(0, overlap))
        elif chunk_style == CHUNK_STYLE_PARAGRAPH:
            units = self._split_paragraphs(text)
            return self._pack_units(units, target_chars=chunk_size, overlap_units=max(0, overlap))

        raise ValueError(f"Unsupported chunk_style: {chunk_style}")

    def _split_fixed(self, text: str, *, chunk_size: int, overlap_chars: int) -> List[str]:
        chunks: List[str] = []
        n = len(text)
        if n == 0:
            return chunks
        start = 0
        step = max(1, chunk_size - max(0, overlap_chars))
        while start < n:
            end = min(n, start + chunk_size)
            chunks.append(text[start:end])
            if end >= n:
                break
            start += step
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        norm = re.sub(r"[ \t]+", " ", text.strip())
        parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", norm)
        return [p.strip() for p in parts if p.strip()] or [norm]

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n\s*\n+", text.strip())
        return [p.strip() for p in parts if p.strip()] or [text.strip()]

    def _pack_units(self, units: List[str], *, target_chars: int, overlap_units: int) -> List[str]:
        if not units:
            return []
        chunks: List[str] = []
        i = 0
        overlap_units = max(0, overlap_units)
        while i < len(units):
            current: List[str] = []
            current_len = 0
            while i < len(units):
                u = units[i]
                add_len = len(u) + (1 if current else 0)
                if current and current_len + add_len > target_chars:
                    break
                current.append(u)
                current_len += add_len
                i += 1
            if not current:
                current = [units[i]]
                i += 1
            chunks.append("\n".join(current).strip())
            if overlap_units > 0 and i < len(units):
                i = max(0, i - overlap_units)
        return chunks

    # -------------------- Embedding API --------------------

    def _extract_vectors_from_result(self, raw_result) -> List[List[float]]:
        """
        Adapts to either legacy dict response or new client-style objects.
        """
        # Newer style: result.embeddings -> list of objects with .values
        if hasattr(raw_result, "embeddings"):
            return [[float(v) for v in emb.values] for emb in raw_result.embeddings]

        # Legacy dict style: {'embedding': [[...], [...]]} or {'embedding': {...}}
        if isinstance(raw_result, dict) and "embedding" in raw_result:
            emb = raw_result["embedding"]
            if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
                return [[float(x) for x in vec] for vec in emb]
            if isinstance(emb, dict) and "values" in emb:
                return [[float(x) for x in emb["values"]]]
        raise RuntimeError("Unrecognized embed_content response format")

    def _maybe_normalize(self, vectors: List[List[float]], output_dimensionality: Optional[int]) -> List[List[float]]:
        """
        Normalize vectors to unit norm if dimension != 3072 (per Gemini guidance).
        """
        if not vectors:
            return vectors
        if output_dimensionality is None or output_dimensionality == 3072:
            return vectors
        normed = []
        for vec in vectors:
            s = sum(x * x for x in vec) ** 0.5
            if s == 0.0:
                normed.append(vec)
            else:
                normed.append([x / s for x in vec])
        return normed

    async def _embed_chunks_via_api(
        self,
        chunks: List[str],
        *,
        embedding_style: EmbeddingStyle,
        output_dimensionality: Optional[int],
    ) -> Dict[str, List[float]]:
        """
        Sends a batch of chunk strings to Gemini embeddings.
        """
        if not chunks:
            return {}
        logger.info(
            f"Embedding {len(chunks)} chunk(s) with style={embedding_style}"
            + (f", dim={output_dimensionality}" if output_dimensionality else "")
        )
        try:
            loop = asyncio.get_running_loop()

            def _call():
                # Prefer new API signature if available; fall back otherwise.
                try:
                    from google.genai import types as genai_types  # available in newer SDKs
                    client = genai.Client()
                    cfg = genai_types.EmbedContentConfig(
                        task_type=embedding_style,
                        output_dimensionality=output_dimensionality if output_dimensionality else None,
                    )
                    return client.models.embed_content(
                        model=self.embedding_model_name,
                        contents=chunks,
                        config=cfg,
                    )
                except Exception:
                    # Fallback to older interface
                    return genai.embed_content(
                        model=self.embedding_model_name,
                        content=chunks,
                        task_type=embedding_style,
                        output_dimensionality=output_dimensionality,
                    )

            raw = await loop.run_in_executor(None, _call)
            vectors = self._extract_vectors_from_result(raw)
            vectors = self._maybe_normalize(vectors, output_dimensionality)

            # 1:1 mapping with input chunks
            return dict(zip(chunks, vectors))

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def _get_embeddings_from_chunks(
        self,
        chunks: List[str],
        *,
        embedding_style: EmbeddingStyle,
        output_dimensionality: Optional[int],
    ) -> Dict[str, List[float]]:
        """
        Cache-first: check ZMongo cache, embed misses with chosen style/dimension, then backfill cache.
        Cache key is the SHA256 of the exact chunk text + style + dim.
        """
        if not chunks:
            return {}

        def _hash(chunk: str) -> str:
            # Include style + dimension in the cache key so different configs don't collide
            tag = f"{embedding_style}:{output_dimensionality or 'default'}:"
            return hashlib.sha256((tag + chunk).encode("utf-8")).hexdigest()

        hashes = [_hash(c) for c in chunks]
        hash_to_chunk = dict(zip(hashes, chunks))

        cache_results = await self.repository.find_documents(
            "_embedding_cache", {"text_hash": {"$in": hashes}}
        )

        found_embeddings = {
            res["source_text"]: res["embedding"]
            for res in (cache_results.data or [])
            if cache_results.success
        }
        logger.info(f"Found {len(found_embeddings)} of {len(chunks)} chunks in ZMongo cache.")

        cached_hashes = {res["text_hash"] for res in (cache_results.data or [])}
        missing_hashes = set(hashes) - cached_hashes
        chunks_to_embed = [hash_to_chunk[h] for h in missing_hashes]

        if chunks_to_embed:
            api_embeddings = await self._embed_chunks_via_api(
                chunks_to_embed,
                embedding_style=embedding_style,
                output_dimensionality=output_dimensionality,
            )
            new_cache_entries = [
                {
                    "text_hash": _hash(chunk),
                    "embedding": emb,
                    "source_text": chunk,
                    "embedding_style": embedding_style,
                    "output_dimensionality": output_dimensionality or 3072,
                }
                for chunk, emb in api_embeddings.items()
            ]
            if new_cache_entries:
                await self.repository.insert_documents("_embedding_cache", new_cache_entries)
            found_embeddings.update(api_embeddings)

        return found_embeddings

    # -------------------- Public API --------------------

    async def embed_text(
        self,
        text: str,
        *,
        # chunking
        chunk_style: ChunkStyle = CHUNK_STYLE_FIXED,
        chunk_size: int = 1500,
        overlap: int = 150,
        # embedding
        embedding_style: EmbeddingStyle = "RETRIEVAL_DOCUMENT",
        output_dimensionality: Optional[int] = None,  # e.g., 768, 1536, or 3072
    ) -> List[List[float]]:
        """
        Embeds a single text using the specified chunking and embedding styles.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        chunks = self._split_text_into_chunks(
            text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap
        )
        if not chunks:
            return []

        embedding_map = await self._get_embeddings_from_chunks(
            chunks,
            embedding_style=embedding_style,
            output_dimensionality=output_dimensionality,
        )
        return [embedding_map[chunk] for chunk in chunks if chunk in embedding_map]

    async def embed_texts_batched(
        self,
        texts: List[str],
        *,
        # chunking
        chunk_style: ChunkStyle = CHUNK_STYLE_FIXED,
        chunk_size: int = 1500,
        overlap: int = 150,
        # embedding
        embedding_style: EmbeddingStyle = "RETRIEVAL_DOCUMENT",
        output_dimensionality: Optional[int] = None,
    ) -> Dict[str, List[List[float]]]:
        """
        Embeds multiple texts with uniform chunking/embedding parameters.
        """
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("texts must be a non-empty list of non-empty strings")

        text_to_chunks_map = {
            text: self._split_text_into_chunks(
                text, chunk_style=chunk_style, chunk_size=chunk_size, overlap=overlap
            )
            for text in texts
        }
        all_chunks = list(chain.from_iterable(text_to_chunks_map.values()))
        unique_chunks = sorted(set(all_chunks))

        embedding_map = await self._get_embeddings_from_chunks(
            unique_chunks,
            embedding_style=embedding_style,
            output_dimensionality=output_dimensionality,
        )

        result: Dict[str, List[List[float]]] = {}
        for text, chunks in text_to_chunks_map.items():
            result[text] = [embedding_map[chunk] for chunk in chunks if chunk in embedding_map]
        return result

    async def embed_and_store(
        self,
        document_id: str | ObjectId,
        text: str,
        *,
        embedding_field: str = "embeddings",
        # chunking
        chunk_style: ChunkStyle = CHUNK_STYLE_FIXED,
        chunk_size: int = 1500,
        overlap: int = 150,
        # embedding
        embedding_style: EmbeddingStyle = "RETRIEVAL_DOCUMENT",
        output_dimensionality: Optional[int] = None,
    ) -> SafeResult:
        """
        Embeds a single text and stores chunked embeddings into the target document.
        """
        try:
            obj_id = ObjectId(document_id) if isinstance(document_id, str) else document_id
        except InvalidId:
            error_msg = f"Error: Provided string '{document_id}' is not a valid ObjectId."
            logger.error(error_msg)
            return self.repository.fail(error_msg)

        embeddings = await self.embed_text(
            text,
            chunk_style=chunk_style,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_style=embedding_style,
            output_dimensionality=output_dimensionality,
        )
        if embeddings:
            return await self.repository.update_document(
                self.collection,
                {"_id": obj_id},
                {"$set": {embedding_field: embeddings}},
                upsert=True,
            )
        return self.repository.ok(data={"message": "No embeddings were generated or stored."})


# -------------------- Demo: run different methods on the same string --------------------

async def _demo():
    embedder = ZMongoEmbedder(collection="demo_embeddings")

    text = (
        "Artificial intelligence is transforming the legal industry. "
        "Lawyers now use AI for document review, case prediction, and drafting. "
        "These tools improve efficiency but also raise questions about ethics and accountability."
    )

    print("\n--- Chunking styles (embedding_style=RETRIEVAL_DOCUMENT, dim=768) ---")
    for cs in (CHUNK_STYLE_FIXED, CHUNK_STYLE_SENTENCE, CHUNK_STYLE_PARAGRAPH):
        vecs = await embedder.embed_text(
            text,
            chunk_style=cs,
            chunk_size=160,   # small to provoke multiple chunks
            overlap=1,
            embedding_style="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        )
        print(f"{cs:<10}: {len(vecs)} vector(s); first8={vecs[0][:8]}")

    print("\n--- Embedding styles (chunk_style=sentence, dim=768) ---")
    for es in ("SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"):
        vecs = await embedder.embed_text(
            text,
            chunk_style=CHUNK_STYLE_SENTENCE,
            chunk_size=220,
            overlap=0,
            embedding_style=es,       # task type
            output_dimensionality=768,
        )
        print(f"{es:<20}: {len(vecs)} vector(s); first8={vecs[0][:8]}")


if __name__ == "__main__":
    asyncio.run(_demo())

