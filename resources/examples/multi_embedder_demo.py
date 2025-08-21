import asyncio
from pathlib import Path
from dotenv import load_dotenv

from zmongo_embedder import (
    ZMongoEmbedder,
    CHUNK_STYLE_FIXED,
    CHUNK_STYLE_SENTENCE,
    CHUNK_STYLE_PARAGRAPH,
)

load_dotenv(Path.home() / "resources" / ".env_local")

async def main():
    embedder = ZMongoEmbedder(collection="demo_embeddings")
    text = ("AI is transforming the legal industry. Lawyers use AI for review, "
            "prediction, and drafting. That raises ethics questions.")

    # Different chunking styles (same embedding style)
    for cs in (CHUNK_STYLE_FIXED, CHUNK_STYLE_SENTENCE, CHUNK_STYLE_PARAGRAPH):
        vecs = await embedder.embed_text(
            text,
            chunk_style=cs,
            chunk_size=120,
            overlap=1,
            embedding_style="RETRIEVAL_DOCUMENT",
        )
        print(f"{cs}: {len(vecs)} vector(s); first8={vecs[0][:8]}")

    # Different embedding styles (same chunking)
    for es in ("SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
        vecs = await embedder.embed_text(
            text,
            chunk_style=CHUNK_STYLE_SENTENCE,
            chunk_size=200,
            overlap=0,
            embedding_style=es,
        )
        print(f"{es}: {len(vecs)} vector(s); first8={vecs[0][:8]}")

if __name__ == "__main__":
    asyncio.run(main())
