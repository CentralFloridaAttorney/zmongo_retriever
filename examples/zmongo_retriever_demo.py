import os

from dotenv import load_dotenv
from langchain.chains import load_summarize_chain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import PromptTemplate

from src.zmongo_retriever import ZMongoRetriever

load_dotenv('../excluded/.env')
# Set your variables
model_path_1 = '/mnt/storage/models/dolphin-2.1-mistral-7B-GGUF/dolphin-2.1-mistral-7b.Q4_0.gguf'
model_path_2 = '/PycharmProjects/zcases/zassistant/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_0.gguf'
mongo_uri = 'mongodb://localhost:49999' # Your mongo_uri
this_collection_name = 'zcases'  # Your MongoDB collection
this_page_content_field = 'opinion'  # Specify the field to use as page_content
predator_this_document_id = '65d995ee2051723e1bb6f154'  # Example ObjectId('_id') value
chunk_size = 1024 # larger values for chunk_size may solve problems with exceeding your token limit


retriever = ZMongoRetriever(mongo_uri=mongo_uri,
                            chunk_size=chunk_size,
                            collection_name=this_collection_name,
                            page_content_field=this_page_content_field)
documents_by_id = retriever.invoke(predator_this_document_id, query_by_id=True)

# Pass the Document
# The following may not work with documents > 4097 in length
prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
llm = LlamaCpp(
    model_path=model_path_1,
    max_tokens=0,
    n_gpu_layers=-1,
    n_batch=8192,
    verbose=True,
    f16_kv=True,
    n_ctx=8192
)
summary_chain = load_summarize_chain(OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY')), chain_type="stuff", prompt=prompt)
result = summary_chain.invoke({'input_documents': documents_by_id[0]})
print(result)
