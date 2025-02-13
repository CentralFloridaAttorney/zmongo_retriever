import os

from dotenv import load_dotenv
from langchain.chains import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from zmongo import zconstants
from zmongo.BAK.zmongo_retriever import ZMongoRetriever

# The embedder requires the use of openai
# OPENAI_API_KEY must be in your .env file

# Set your variables
mongo_uri = zconstants.MONGO_URI # Your mongo_uri
this_collection_name = zconstants.DEFAULT_COLLECTION_NAME  # Your MongoDB collection
this_page_content_key = 'casebody.data.opinions.0.text'  # Specify the field to use as page_content
predator_this_document_id = '65f28c8103fc21342e2dc04d'  # Example ObjectId('_id') value
chunk_size = 256 # smaller values for more content, larger values may solve problems with token limits
load_dotenv()

retriever = ZMongoRetriever(mongo_uri=mongo_uri,
                            chunk_size=chunk_size,
                            collection_name=this_collection_name,
                            page_content_key=this_page_content_key)
documents_by_page_content_key = retriever.invoke(object_ids=predator_this_document_id, page_content_key=this_page_content_key, existing_metadata={'summary_preceding_text': 'Summary: 1. The lower court entered an order granting the Plaintiffs motion for summary judgment. 2. The Defendant appealed the order.'})

# Pass the Document
# The following may not work with documents > 4097 in length
prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
summary_chain = load_summarize_chain(OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY')), chain_type="stuff", prompt=prompt)
result = summary_chain.invoke({'input_documents': documents_by_page_content_key[0]})
print(result)

# Example result {'input_documents': [<zmongo_retriever.Document object at 0x7afe3e3b7820>,
# <zmongo_retriever.Document object at 0x7afe3e3b7880>, <zmongo_retriever.Document object at 0x7afe3e3b78e0>,
# <zmongo_retriever.Document object at 0x7afe3e3b7940>, <zmongo_retriever.Document object at 0x7afe3e3b79a0>,
# <zmongo_retriever.Document object at 0x7afe3e3b7a00>, <zmongo_retriever.Document object at 0x7afe3e3b7a60>,
# <zmongo_retriever.Document object at 0x7afe3e3b7ac0>], 'output_text': " - Craig Lamb appealed a final judgment of
# foreclosure.\n   - Trial court erred in determining standing of Nationstar Mortgage, LLC and reversed.\n   - Court
# reviews sufficiency of evidence for standing de novo.\n   - Bank must establish standing at time of final judgment,
# in addition to when complaint is filed.\n   - Case was commenced by Aurora Loan Services, LLC.\n   - Nationstar
# filed a Motion for Substitution of Party Plaintiff and was allowed to substitute for Aurora.\n   - At trial,
# court took judicial notice of the order allowing substitution.\n   - Nationstar's witness testified that they
# acquired Aurora Loans and now service all of Aurora Loans.\n   - Original note was lost and a copy with a special
# indorsement to Aurora was placed into evidence.\n   - Standing can be proven through evidence of a valid
# assignment, proof of purchase of the debt, or an effective transfer.\n   - Witness testimony can serve as an
# affidavit of ownership.\n   - Nationstar did not prove standing through evidence of an assignment,
# as the assignment only assigned the mortgage.\n   - Nationstar also failed to prove standing through proof of
# purchase of the debt, evidence of an effective transfer, or an affidavit of ownership.\n   - Record lacks evidence
# that the note was transferred to Nationstar.\n   - Trial"}
