# ZMongo Retriever

[ZMongoRetriever](docs/DOCUMENTATION.md) is a Python-based utility designed to facilitate the fetching and processing of documents from MongoDB collections. It is particularly useful in scenarios involving large text documents that need to be broken down into smaller, manageable chunks for further analysis or processing. Additionally, ZMongo Retriever supports the inclusion of existing metadata to enrich the documents further before processing.  It is highly recommended for you to use a [VIRTUAL ENVIRONMENT](docs%2FCREATE_VIRTUAL_ENVIRONMENT.md) for your program.


## Examples

[Jupyter Notebook Using Local Llama](examples/zmongo_retriever_demo.ipynb) | [Download Llama Model](examples/INSTALL_DOLPHIN_MISTRAL.md)

[Python Using OpenAI](examples/EXAMPLE.md) | [Get OPENAI_API_KEY](examples/GET_OPENAI_API_KEY.md)


## Example Output from:

U.S. BANK NAT. ASSN a/k/a U.S. Bank National Association v. QUADOMAIN CONDOMINUM ASSOCIATION, INC., 103 So. 3d 977 (Fla. 4th DCA 2012) 

```angular2html
 - U.S. Bank appeals an order denying its motion to vacate a final judgment of lien foreclosure
   - U.S. Bank held a first mortgage and note on a unit in a condominium managed by the Association
   - After the unit owner defaulted, U.S. Bank initiated a foreclosure action and obtained a certificate of title
   - Before judgment was entered, the unit owner passed away and the unit transferred to their heirs
   - U.S. Bank was granted leave to file a Supplemental Complaint to Re-foreclose Mortgage to Foreclose Omitted Defendant
   - U.S. Bank recorded a supplemental notice of lis pendens in conjunction with its re-foreclosure action
   - During this time, Association fees went unpaid and the Association recorded a claim of lien against U.S. Bank
   - The Association then filed a suit to foreclose its lien
   - U.S. Bank did not respond to the Association's complaint in the lien foreclosure case
   - The Association obtained a default final summary judgment and the unit was sold at a public sale
   - U.S. Bank filed a motion to vacate, arguing that the lien foreclosure action was barred by the supplemental lis pendens
   - The trial court denied the motion and U.S
```

## Installation

Before diving into ZMongo Retriever, ensure that you have MongoDB and the necessary Python dependencies installed. You can install the required Python packages using pip:

```bash
pip install pymongo langchain langchain_community langchain_text_splitters
```

For working with the examples and managing environment variables efficiently, install the following additional packages:

```bash
pip install python-dotenv llama-cpp-python langchain_openai 
```

## Key Features

- **Document Retrieval:** Fetch documents by ID or through text search queries directly from MongoDB collections.
- **Document Splitting:** Automatically splits large text documents into smaller chunks suitable for analysis or machine learning models.
- **Metadata Handling:** Allows for the seamless inclusion and combination of existing metadata with default metadata, enhancing the information available for each document.

## Usage

### Initialization

To get started with ZMongo Retriever, initialize the main class with your MongoDB connection details and the specific collection you intend to query:

```python
from zmongo_retriever import ZMongoRetriever

retriever = ZMongoRetriever(
    mongo_uri='mongodb://localhost:27017',
    db_name='your_database_name',
    collection_name='your_collection_name',
    page_content_field='field_containing_text',
    chunk_size=1024  # Adjust the chunk size as needed
)
```

### Fetching and Processing Documents

Invoke the `retriever` with your query to fetch and process documents. You can search by document ID or a text query. Optionally, you can pass existing metadata to combine with each document's default metadata:

```python
query = 'example search query'
query_by_id = False
existing_metadata = {'additional': 'metadata'}

documents = retriever.invoke(query, query_by_id=query_by_id, existing_metadata=existing_metadata)

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
```

If querying by ID, ensure to set `query_by_id=True`.

### Handling Large Documents

ZMongo Retriever uses the `RecursiveCharacterTextSplitter` from `langchain_text_splitters` to split large documents into smaller chunks. This functionality is crucial when dealing with documents that exceed the token limits of machine learning models or APIs.

### Combining Metadata

The utility allows for the enrichment of documents with additional metadata. This feature is particularly useful for applications that require detailed contextual information about the documents being processed.

## Conclusion

ZMongo Retriever streamlines the retrieval and preprocessing of MongoDB documents, making it an essential tool for developers and data scientists working with large datasets. By supporting custom queries, document splitting, and metadata enrichment, it ensures that documents are ready for analysis or machine learning applications.