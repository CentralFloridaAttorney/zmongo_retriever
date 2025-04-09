# zmagnum_demo.py

import asyncio
from zmongo_toolbag.zmagnum import ZMagnum


async def main():
    zmag = ZMagnum(disable_cache=False)  # Enable caching

    collection = "demo_docs"

    # 🔹 Insert test documents
    documents = [{"name": f"doc{i}", "value": i} for i in range(5)]
    insert_result = await zmag.insert_documents(collection, documents)
    print("✅ Insert Result:", insert_result)

    # 🔹 Fetch a document (triggers DB fetch and caching)
    query = {"name": "doc2"}
    doc = await zmag.find_document(collection, query)
    print("🔍 First fetch from DB:", doc)

    # 🔹 Fetch the same doc again (should hit cache)
    cached_doc = await zmag.find_document(collection, query)
    print("🧠 Second fetch from cache:", cached_doc)

    # 🔹 Profile a sample synchronous operation
    def sample_math(x):
        return x * 2

    print("⏱ Profiling sample function:")
    result = zmag._profile("double_value", sample_math, 21)
    print("Result:", result)

    # 🔹 Delete all documents
    deleted_count = await zmag.delete_all_documents(collection)
    print(f"🗑 Deleted {deleted_count} documents.")

    # 🔹 Close connection
    await zmag.close()


if __name__ == "__main__":
    asyncio.run(main())
