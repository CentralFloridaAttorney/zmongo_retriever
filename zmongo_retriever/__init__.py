import json
from datetime import datetime

from bson.objectid import ObjectId
from pymongo import MongoClient











if __name__ == "__main__":
    ZTokenEstimator.main()

    retriever = ZMongoRetriever(chunk_size=1024)
    test_queries = ["65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47", "65eab5493c6a0853d9a9cce7", "65eab55e3c6a0853d9a9cd54", "65eab5363c6a0853d9a9cc80", "65eab52b3c6a0853d9a9cc47", "65eab5493c6a0853d9a9cce7", "65eab55e3c6a0853d9a9cd54"]
    documents = retriever.invoke(test_queries)
    for i, group in enumerate(documents):
        print(f"Group {i+1} - Total Documents: {len(group)}")
        for doc in group:
            print(f"Metadata: {doc.metadata}, Content Preview: {doc.page_content[:100]}...")
