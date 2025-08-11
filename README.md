

### Core Features 🚀

ZMongo provides several key features to enhance MongoDB operations:

  * **Asynchronous by Default**: ZMongo is built on **motor** for non-blocking database I/O, making it ideal for modern web servers and data processing pipelines.
  * **Intelligent In-Memory Cache**: It automatically caches query results to reduce database load and provide near-instant reads for repeated requests. The cache is automatically invalidated on writes.
  * **High-Performance Bulk Operations**: ZMongo includes optimized methods for **bulk\_write**, **insert\_many**, and **delete\_many** to efficiently handle large volumes of data.
  * **Simplified Interface**: A **SafeResult** wrapper offers a clean and predictable API for common MongoDB operations, simplifying error handling and data access.
  * **Standalone & Dependency-Free**: It only requires a connection to MongoDB and doesn't need external services like Redis for its core caching and performance features.

-----

### Installation and Configuration 🛠️

ZMongo can be installed from source by first cloning the GitHub repository and then running the editable `pip install` command. The toolkit requires Python 3.10+ and a MongoDB instance.

```bash
git clone https://github.com/CentralFloridaAttorney/zmongo_retriever.git
cd zmongo_retriever
pip install -e .
```

To configure ZMongo, you must set the `MONGO_URI` and `MONGO_DATABASE_NAME` environment variables, typically in a `.env` file.

-----

### Performance Benchmarks 📈

A comparison of ZMongo against the **Motor** (async) and **PyMongo** (sync) drivers shows the following performance results:

  * **Reads & Inserts**: ZMongo has a minor overhead compared to the raw drivers, which is a trade-off for its additional features. **PyMongo** was found to be faster for inserting 1000 documents and finding 100 documents.
  * **Updates**: ZMongo demonstrated the best performance for update operations, likely due to its efficient cache invalidation strategy.
  * **Deletes**: **Motor** was the fastest driver for deleting all documents.

Overall, ZMongo offers competitive performance while providing a safer and more feature-rich developer experience.