from langchain.chains import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

import zconstants
from zmongo_retriever import ZMongoRetriever

# The embedder requires the use of openai
# OPENAI_API_KEY must be in your .env file

# Set your variables
mongo_uri = 'mongodb://localhost:49999' # Your mongo_uri
this_collection_name = zconstants.ZCASES_COLLECTION  # Your MongoDB collection
this_page_content_field = 'opinion'  # Specify the field to use as page_content
predator_this_document_id = '65b140719b04571b92cd8e03'  # Example ObjectId('_id') value
chunk_size = 1024 # larger values for chunk_size may solve problems with exceeding your token limit


retriever = ZMongoRetriever(mongo_uri=mongo_uri,
                            chunk_size=chunk_size,
                            collection_name=this_collection_name,
                            page_content_field=this_page_content_field)
documents_by_id = retriever.invoke(predator_this_document_id, existing_metadata={'summary_preceding_text': 'Summary: 1. The lower court entered an order granting the Plaintiffs motion for summary judgment. 2. The Defendant appealed the order.'})

# Pass the Document
# The following may not work with documents > 4097 in length
prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
summary_chain = load_summarize_chain(OpenAI(openai_api_key=zconstants.OPENAI_API_KEY), chain_type="stuff", prompt=prompt)
result = summary_chain.invoke({'input_documents': documents_by_id[0]})
print(result)
