import asyncio
from bson import ObjectId

from zmongo_retriever.zmongo_toolbag.zmongo_embedder import (
    ZMongoEmbedder,
    CHUNK_STYLE_FIXED,
    CHUNK_STYLE_SENTENCE,
    CHUNK_STYLE_PARAGRAPH,
    field_name,
)

async def _demo():
    # Use the collection where demo docs + embeddings will live
    embedder = ZMongoEmbedder(collection="demo_embeddings")
    try:
        text = (
            "Artificial intelligence is transforming the legal industry. "
            "Lawyers now use AI for document review, case prediction, and drafting. "
            "These tools improve efficiency but also raise questions about ethics and accountability."
        )

        print("\n--- Chunking styles (embedding_style=RETRIEVAL_DOCUMENT, dim=768) ---")
        for cs in (CHUNK_STYLE_FIXED, CHUNK_STYLE_SENTENCE, CHUNK_STYLE_PARAGRAPH):
            vecs = await embedder.get_embedding(
                text,
                chunk_style=cs,
                chunk_size=160,   # small to provoke multiple chunks
                overlap=1,
                embedding_style="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            )
            first8 = vecs[0][:8] if vecs else []
            print(f"{cs:<10}: {len(vecs)} vector(s); first8={first8}")

        print("\n--- Embedding styles (chunk_style=sentence, dim=768) ---")
        for es in ("SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"):
            vecs = await embedder.get_embedding(
                text,
                chunk_style=CHUNK_STYLE_SENTENCE,
                chunk_size=220,
                overlap=0,
                embedding_style=es,     # task type
                output_dimensionality=768,
            )
            first8 = vecs[0][:8] if vecs else []
            print(f"{es:<20}: {len(vecs)} vector(s); first8={first8}")

        # ---- Persist an embedding on a real document ----
        print("\n--- Persisting embeddings to Mongo ---")
        doc_id = ObjectId()
        # Insert a base document with a 'text' field
        ins = await embedder.repo.insert_document(
            "demo_embeddings",
            {"_id": doc_id, "text": text}
        )
        assert ins.success, f"Insert failed: {ins.error}"

        # Name the embedding field using the convention [BASE]_[STYLE]_[CHUNK]
        target_field = field_name("text", "RETRIEVAL_DOCUMENT", "sentence")

        # Compute + save the embeddings onto the document
        res = await embedder.embed_and_store(
            document_id=doc_id,
            text=text,
            embedding_field=target_field,
            chunk_style=CHUNK_STYLE_SENTENCE,
            chunk_size=220,
            overlap=0,
            embedding_style="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
            include_vectors_in_result=True,  # SafeResult will include vectors
        )
        print("Saved OK?:", res.success)
        if res.success:
            print("Vectors persisted:", res.data.get("vectors_count"))
            # For demo purposes only, peek at the first 8 numbers of the first vector:
            vectors = res.data.get("vectors") or []
            if vectors:
                print("First vector first8:", vectors[0][:8])

        # Read back from Mongo and verify the field exists
        got = await embedder.repo.find_document("demo_embeddings", {"_id": doc_id})
        assert got.success and got.data, f"Find failed: {got.error}"
        present = target_field in got.data
        print(f"Field '{target_field}' present in doc?:", present)
        if present:
            print("Stored chunk count:", len(got.data[target_field]))
            print("Stored vector dim:", len(got.data[target_field][0]) if got.data[target_field] else 0)

    finally:
        embedder.close()


if __name__ == "__main__":
    asyncio.run(_demo())
