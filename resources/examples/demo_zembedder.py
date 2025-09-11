import asyncio
import logging
import os
from bson import ObjectId

from zmongo_toolbag.zembedder import CHUNK_STYLE_PARAGRAPH, EMBEDDING_STYLE_RETRIEVAL_DOCUMENT, ZEmbedder

# This script assumes the ZEmbedder class is in a file named 'zembedder.py'
# in the same directory.


# --- The Text for the Demo ---
# This text is structured into four distinct paragraphs. When using
# CHUNK_STYLE_PARAGRAPH, we expect it to be split into four chunks,
# resulting in four embedding vectors.
LONG_TEXT = """The history of computing began long before the digital age. Early mechanical devices, like the abacus, were used for calculation for thousands of years. The true precursor to the modern computer, however, was Charles Babbage's Analytical Engine in the 19th century. Though never fully built in his lifetime, its design included an arithmetic logic unit, control flow in the form of conditional branching and loops, and integrated memory, making it the first design for a general-purpose, Turing-complete computer.

The electromechanical era followed, with devices like the Atanasoff-Berry Computer and the Harvard Mark I paving the way. The major breakthrough came with the advent of fully electronic computers during World War II. ENIAC (Electronic Numerical Integrator and Computer) was a colossal machine that used vacuum tubes instead of mechanical relays, increasing calculation speed by orders of magnitude. It was programmable, but required manual rewiring to change its operations, a tedious process that highlighted the need for a more flexible architecture.

This need was met by the von Neumann architecture, which introduced the concept of the stored-program computer. This design, where program instructions and data are stored in the same read-write memory, remains the fundamental basis for nearly all modern computers. The invention of the transistor in 1947, and later the integrated circuit, allowed computers to become smaller, faster, cheaper, and more reliable, moving from room-sized behemoths to machines that could fit on a desk.

The final leap was the microprocessor, which placed an entire central processing unit (CPU) onto a single integrated circuit chip. This innovation fueled the personal computer revolution of the 1970s and 80s, bringing computing power to individuals and small businesses. The subsequent development of graphical user interfaces and the global connectivity of the internet transformed the computer from a specialized tool for experts into an indispensable part of modern life for billions of people.
"""


async def main():
    """
    A demo script to showcase the chunking and embedding of a long text document.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Check for the required environment variable
    if not os.getenv("EMBEDDING_MODEL_PATH"):
        print("ERROR: Set the LLAMA_MODEL_PATH environment variable before running the demo.")
        return

    # 2. Initialize ZEmbedder
    print("--- Initializing ZEmbedder ---")
    embedder = ZEmbedder()
    collection = "zembedder_chunking_demo"
    # Use a fixed, predictable document ID for repeatable tests
    document_id = ObjectId("61c0c55e0000000000000001")
    print(f"Using collection '{collection}' and document_id '{document_id}'")

    # 3. Prepare the database
    # Start with a clean slate to ensure we are not hitting a previous cache
    print("\n--- Preparing Database (ensuring a fresh start) ---")
    await embedder.repo.delete_document(collection, {"_id": document_id})
    await embedder.repo.insert_document(
        collection,
        {"_id": document_id, "source_text": LONG_TEXT, "title": "History of Computing"}
    )
    print("Clean document inserted into the database.")

    # 4. Run the embedding process
    print("\n--- Running get_embedding on the long text ---")
    print(f"Using chunk_style: '{CHUNK_STYLE_PARAGRAPH}'")

    result = await embedder.get_embedding(
        embedding_style=EMBEDDING_STYLE_RETRIEVAL_DOCUMENT,
        collection=collection,
        document_id=document_id,
        embedding_field="paragraph_embeddings",  # Use a descriptive field name
        text_field="source_text",
        chunk_style=CHUNK_STYLE_PARAGRAPH,  # Explicitly set the chunk style
    )

    # 5. Analyze and print the results
    print("\n--- Analyzing Results ---")
    if result.success:
        data = result.data
        vector_count = data.get("vectors_count")

        print(f"Success: {result.success}")
        print(f"Retrieved from Cache: {data.get('from_cache')}")
        print(f"Dimensionality per Vector: {data.get('dimensionality')}")
        print(f"Number of Vectors Generated: {vector_count}")

        if vector_count == 4:
            print("\nSUCCESS: The text was correctly split into 4 chunks (one for each paragraph) and embedded.")
        else:
            print(f"\nNOTE: Expected 4 vectors but got {vector_count}. Check the text formatting or chunking logic.")

    else:
        print(f"Embedding failed: {result.error}")

    # 6. Clean up
    embedder.close()
    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
