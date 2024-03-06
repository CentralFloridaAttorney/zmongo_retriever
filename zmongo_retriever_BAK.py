import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, load_summarize_chain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pymongo import MongoClient, TEXT
from bson import ObjectId

# Load environment variables from .env file
load_dotenv('env_files/.env_omen')


class Document:
    def __init__(self, page_content, this_metadata):
        self.page_content = page_content
        self.metadata = this_metadata


def convert_dict_to_metadata(dict_data, existing_metadata=None, metadata_prefix='key_from_dict'):
    """
    Recursively convert JSON data to metadata format suitable for passing to the retriever.

    Args:
        dict_data (dict): JSON data to be converted to metadata.
        existing_metadata (dict): Existing metadata dictionary to add the converted data. Default is None.
        metadata_prefix (str): Prefix for keys to distinguish nested fields. Default is an empty string.

    Returns:
        dict: Metadata dictionary containing converted JSON data.
    """
    if existing_metadata is None:
        existing_metadata = {}

    for key, value in dict_data.items():
        new_prefix = f"{metadata_prefix}_{key}" if metadata_prefix else key
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            convert_dict_to_metadata(value, existing_metadata, new_prefix)
        elif isinstance(value, list):
            # Convert list items
            for idx, item in enumerate(value):
                item_prefix = f"{new_prefix}_{idx}"
                if isinstance(item, dict) or isinstance(item, list):
                    # If item is a dictionary or a list, recursively process it
                    convert_dict_to_metadata(item, existing_metadata, item_prefix)
                else:
                    # Convert non-dictionary/list items to string format and add to metadata
                    existing_metadata[item_prefix] = str(item)
        else:
            # Convert value to string format and add to metadata
            existing_metadata[new_prefix] = str(value)

    return existing_metadata

def get_opinion_from_zcase(zcase):
    casebody = zcase.get('casebody')
    data = casebody.get('data')
    opinions = data.get('opinions')
    this_opinion = opinions[0]
    opinion_text = this_opinion.get('text')
    return opinion_text

class ZMongoRetriever:
    def __init__(self,
                 db_name=os.getenv('MONGO_DATABASE_NAME'),
                 mongo_uri=os.getenv('MONGO_URI'),
                 collection_name=os.getenv('DEFAULT_COLLECTION_NAME'),
                 page_content_field='page_content'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.page_content_field = page_content_field
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1024)  # Adjust chunk_size as needed


    def _get_relevant_documents(self, query, query_by_id=False):
        documents = []
        if query_by_id:
            cursor = self.collection.find({"_id": ObjectId(query)})
        else:
            cursor = self.collection.find({"$text": {"$search": query}})

        for doc in cursor:
            if self.page_content_field == 'opinion' and self.collection_name == 'zcases':
                page_content = get_opinion_from_zcase(doc)
            else:
                page_content = doc.get(self.page_content_field, "Content not found")
            chunks = self.splitter.split_text(page_content)
            this_metadata = self.create_default_metadata(doc)
            these_documents = [Document(page_content=chunk, this_metadata=this_metadata) for chunk in chunks]
            documents.append(these_documents)
        return documents

    def create_default_metadata(self, zobject):
        """
        Creates default metadata for a zdocument.

        Args:
            zobject (dict): The MongoDB document from which metadata is derived.

        Returns:
            dict: A dictionary containing default metadata.
        """
        metadata = {
            "source": "local",  # Indicate the source of the document
            "document_id": str(zobject.get("_id", "N/A")),  # Unique identifier of the document
            "collection_name": self.collection.name,  # Collection from which the document originates
        }
        return metadata

    def invoke(self, query, query_by_id=False):
        documents = self._get_relevant_documents(query, query_by_id)
        return documents


# Example usage
if __name__ == "__main__":
    collection_name = 'zcases'  # Adjust based on your MongoDB setup
    page_content_field = 'opinion'  # Specify the field to use as page_content
    document_id = '65cf9acdb347eec24fd6b02a'  # Example document ID

    retriever = ZMongoRetriever(collection_name=collection_name, page_content_field=page_content_field)
    documents_by_id = retriever.invoke(document_id, query_by_id=True)
    prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
                  Return your response in bullet points which covers the key points of the text.
                  ```{text}```
                  BULLET POINT SUMMARY:
      """




    # not working with long documents
    # create a chain to answer questions
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm = LlamaCpp(
        model_path=os.getenv('MODEL_PATH'),
        max_tokens=0,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=4096,
        verbose=True,
        f16_kv=True
    )
    stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    summary_chain_result = stuff_chain.invoke({'input_documents': documents_by_id[0]})
    print(summary_chain_result)


    for doc in documents_by_id[0]:
        print(doc.page_content, doc.metadata)


    # split the documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents_by_id)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(documents_by_id[0], embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )


    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    result = qa.invoke({'query': 'write an IRAC brief of the case'})
    print(result)
