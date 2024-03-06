from langchain.chains import load_summarize_chain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from zmongo_retriever import ZMongoRetriever

# Set your variables
model_path = '/PycharmProjects/zcases/zassistant/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_0.gguf'
mongo_uri = 'mongodb://localhost:49999' # Your mongo_uri
this_collection_name = 'zcases'  # Your MongoDB collection
this_page_content_field = 'opinion'  # Specify the field to use as page_content
this_document_id = '65cf9acdb347eec24fd6b02a'  # Example ObjectId('_id') value
chunk_size = 1000 # larger values for chunk_size may solve problems with exceeding your token limit


retriever = ZMongoRetriever(mongo_uri=mongo_uri,
                            chunk_size=chunk_size,
                            collection_name=this_collection_name,
                            page_content_field=this_page_content_field)
documents_by_id = retriever.invoke(this_document_id, query_by_id=True)


# The following may not work with very large documents
prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
llm = LlamaCpp(
    model_path=model_path,
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