## ZMongo Comprehensive Usage Guide

The `ZMongo` class is a high-performance MongoDB wrapper providing several distinct operational modes, each tailored to different scenarios based on your application's performance, consistency, and safety needs.

### Available Modes:

* **Default Mode**
* **Fast Mode**
* **Buffered Mode**

---

### 1. Default Mode (`cache=True`, `fast_mode=False`, `buffer_only=False`)

**Description:**
This mode provides full functionality, including safety checks, automatic document serialization with Pydantic alias handling, and a built-in caching layer using `TTLCache` for repeated reads.

**Use When:**

* You prioritize data safety and consistency over raw insertion speed.
* You need automatic document serialization/deserialization.
* You're working with single or small batches of documents where performance overhead is minimal.
* Frequent and repeated queries occur; caching boosts read speed significantly.

**Example Usage:**

```python
await zmongo.insert_document("users", user_doc)
result = await zmongo.find_document("users", {"_id": user_id})
```

**Pros:**
✅ High safety and consistency
✅ Fast cached reads
✅ Automatic document serialization and alias handling

**Cons:**
❌ Slower bulk inserts compared to other modes

---

### 2. Fast Mode (`fast_mode=True`)

**Description:**
Fast mode disables additional overhead such as caching, serialization, and safety wrapping, interacting directly with MongoDB for optimal performance during bulk insert operations.

**Use When:**

* Maximum performance for bulk insertion is necessary.
* You're certain documents are already sanitized and don't require further serialization.
* You can handle the reduced safety measures (no caching, minimal error wrapping).

**Example Usage:**

```python
bulk_docs = [{"name": "User A"}, {"name": "User B"}, ...]
await zmongo.insert_documents("bulk_users", bulk_docs, fast_mode=True)
```

**Pros:**
✅ Highest possible insertion speed, similar to raw Motor performance
✅ Minimal CPU overhead

**Cons:**
❌ No caching or safety checks
❌ No automatic serialization/deserialization; raw dictionaries only

---

### 3. Buffered Mode (`buffer_only=True`)

**Description:**
Buffered mode inserts documents asynchronously into an in-memory buffer (`BufferedAsyncTTLCache`) before flushing them all at once to MongoDB. This write-back buffering strategy greatly improves bulk insert throughput while maintaining some async flexibility.

**Use When:**

* You have large-scale batch inserts or ETL operations that occur periodically.
* You desire batch inserts close to the performance of fast mode but still manage writes asynchronously.
* You're okay with slight latency between data insertion requests and actual database commits.

**Example Usage:**

```python
bulk_docs = [{"name": "Buffered User A"}, {"name": "Buffered User B"}, ...]

await zmongo.insert_documents("buffered_users", bulk_docs, buffer_only=True)
await zmongo.flush_buffered_inserts("buffered_users")
```

**Pros:**
✅ High-speed bulk inserts, near-fast mode performance
✅ Excellent for batch or ETL processing

**Cons:**
❌ Delayed data persistence (until explicit flush)
❌ Requires explicit management of flush operations

---

## ZMongo Performance (Operations per Second)

This section provides detailed operations per second (ops/sec) for each ZMongo operational mode based on recent benchmarks:

| Operation           | Default Mode      | Fast Mode         | Buffered Mode     |
|---------------------|-------------------|-------------------|-------------------|
| Insert (1000 docs)  | ~16,694 ops/sec   | ~71,429 ops/sec   | ~59,880 ops/sec   |
| Find (100 docs)     | ~333,333 ops/sec¹ | ~2,252 ops/sec    | ~2,237 ops/sec    |
| Update (50 docs)    | ~2,315 ops/sec    | ~2,110 ops/sec    | ~1,799 ops/sec    |
| Delete (1000 docs)² | ~833,333 ops/sec  | ~2,500,000 ops/sec| ~2,500,000 ops/sec|

¹ Cached reads significantly boost Default mode's find performance.  
² Deletes tested as bulk deletion of entire collection contents.

### Key Insights:

- **Insert Speed**:  
  **Fast Mode ≈ Buffered Mode** significantly outperform **Default Mode**.
  
- **Read Speed (cached)**:  
  **Default Mode** dramatically outperforms both Buffered and Fast modes due to caching.

- **Safety & Convenience**:  
  **Default Mode** provides the highest safety and convenience, followed by Buffered Mode, with Fast Mode providing minimal safety features.

### Recommended Use-Cases:

- **Default Mode**: Optimal for CRUD apps requiring high-speed repeated queries and data safety.
- **Buffered Mode**: Ideal for batch and ETL jobs requiring a good balance between safety and insertion speed.
- **Fast Mode**: Best for scenarios where maximum bulk insertion performance is the priority over safety.

---

## Recommendations for Optimal Usage:

| Scenario                              | Recommended Mode | Explanation                                          |
| ------------------------------------- | ---------------- | ---------------------------------------------------- |
| Web/API Backend (CRUD)                | Default          | Safety and caching matter more than raw insert speed |
| Bulk ETL / Analytics Data Loads       | Buffered         | Best balance between insert speed and flexibility    |
| High-Speed Logging or Telemetry       | Fast             | Maximum insertion throughput, minimal overhead       |
| Real-Time Reads with Repeated Queries | Default          | Cache dramatically improves response times           |

---

## How to Choose the Right Mode Quickly:

* **Use Default Mode** if you're unsure or working on typical CRUD operations.
* **Use Buffered Mode** for large periodic batches of inserts where you can afford asynchronous data flushes.
* **Use Fast Mode** only when speed absolutely outweighs safety and convenience, typically in tightly controlled bulk ingestion scenarios.

---

## Important Considerations:

* Always remember to explicitly call `flush_buffered_inserts(collection)` when using buffered mode.
* Fast mode bypasses most of ZMongo's benefits; handle with caution to avoid potential data integrity issues.
* Use default mode for critical data that requires immediate consistency and verification.

---

## Example Usage Patterns:

**Web Application (Default mode):**

```python
user = User(name="Alice", email="alice@example.com")
await zmongo.insert_document("users", user)
retrieved_user = await zmongo.find_document("users", {"email": "alice@example.com"})
```

**ETL Script (Buffered mode):**

```python
data_batch = load_bulk_data()
await zmongo.insert_documents("etl_data", data_batch, buffer_only=True)
await zmongo.flush_buffered_inserts("etl_data")
```

**Real-time Telemetry (Fast mode):**

```python
telemetry_events = generate_realtime_events()
await zmongo.insert_documents("telemetry", telemetry_events, fast_mode=True)
```

---

## Final Thoughts

Each mode in ZMongo is purpose-built to address specific trade-offs between performance, safety, and ease-of-use. By understanding these differences, you can achieve optimal performance and reliability tailored specifically to your application's unique requirements.

Use this guide to confidently choose and integrate the appropriate ZMongo mode into your projects.
