# ZMongo Retriever Use Case Suitability

A detailed summary of where ZMongo Retriever excels based on real-world benchmarks.

---

## 📊 Use Case Suitability Table

| Use Case                          | ZMongo ✅ | Why                                                                 |
|----------------------------------|-----------|----------------------------------------------------------------------|
| **LLM/AI Workflows**             | ✅✅✅     | Fast cached reads, embedding support, async-first architecture       |
| **Async Web Servers**            | ✅✅       | Integrates with `asyncio`, excellent concurrent read performance     |
| **LegalTech / NLP Tools**        | ✅✅       | Metadata-safe, recursive-safe flattening, optimized for text         |
| **Edge AI & Agents**             | ✅✅       | In-memory performance without Redis dependency                       |
| **Bulk ETL Ingestion**           | 🟡        | Supports batch ops, but Mongo shell faster for raw throughput        |
| **Analytics Dashboards**         | 🟡✅       | Great for caching reads; Redis better for live metrics/pub-sub       |

---

## 🧪 Real-World Benchmark Comparison

```
ZMongo Retriever Real-World Benchmark Comparison
============================================================

Bulk Write (100k)
-----------------
  MongoDB Shell: 162207.8010 ops/sec  
         ZMongo: 107212.0408 ops/sec  

Concurrent Reads (5k)
---------------------
  MongoDB Shell: 7.9904 s  
          Redis: 0.6397 s  
         ZMongo: 0.1140 s  

Insert (500 docs)
-----------------
  MongoDB Shell: 0.9816 ms/doc  
         ZMongo: 1.3601 ms/doc  
          Redis: 0.0576 ms/doc  

Query Latency (cached)
----------------------
  MongoDB Shell: 1.0082 ms  
         ZMongo: 0.0094 ms  
          Redis: 0.0529 ms  

insert_documents (100k)
-----------------------
         ZMongo: 27605.6581 ops/sec  
  MongoDB Shell: 187871.6902 ops/sec  
          Redis: 17954.9665 ops/sec  
============================================================
```

---
