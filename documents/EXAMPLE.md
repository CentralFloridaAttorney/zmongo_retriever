# Example: Using ZMongoRetriever to Summarize MongoDB Documents

This guide walks you through a simple example of how to use `ZMongoRetriever` with a summarization chain. You’ll retrieve a document from MongoDB and summarize it using an AI model.

---

## 🛠️ What You’ll Need

- Python 3.8+
- MongoDB running locally or remotely
- A `.gguf` LLaMA model file (optional; needed only if not using OpenAI)
- An OpenAI API key (if using OpenAI for summarization)

---

## 📦 Step 1: Install Required Packages

Open your terminal and install the necessary libraries:

```bash
pip install langchain zmongo-retriever python-dotenv
```

If you want to use a local LLaMA model (like Mistral), also install:
```bash
pip install llama-cpp-python
```

---

## 📁 Step 2: Prepare Your Environment

Make sure you have a `.env` file with the following (if using OpenAI):

```
OPENAI_API_KEY=your-openai-key-here
```

---

## 🧪 Step 3: Write Your Script

Create a Python script and paste the following:

```python
import os
from dotenv import load_dotenv
from langchain.chains import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_openai import OpenAI
from zmongo_retriever.zmongo_toolbag.zmongo.BAK import ZMongoRetriever

# Load environment variables
load_dotenv('.env')

# MongoDB configuration
mongo_uri = 'mongodb://localhost:27017'
collection_name = 'zcases'
document_id = '65d995ee2051723e1bb6f154'  # Sample ID
page_content_field = 'opinion'  # Field to summarize

# Optional: path to your local LLaMA model
model_path = '/path/to/your/model.gguf'

# Initialize ZMongoRetriever
retriever = ZMongoRetriever(
    mongo_uri=mongo_uri,
    chunk_size=1024,
    collection_name=collection_name,
    page_content_field=page_content_field
)

# Fetch documents
documents = retriever.invoke(document_id, query_by_id=True)

# Set up a prompt for summarization
prompt_text = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
prompt = PromptTemplate(template=prompt_text, input_variables=["text"])

# Choose your LLM
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=0,
    n_gpu_layers=-1,
    n_batch=4096,
    verbose=True,
    f16_kv=True,
    n_ctx=4096
)

# Or use OpenAI instead:
# llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Load summarization chain
summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

# Run summarization
result = summary_chain.invoke({'input_documents': documents[0]})
print("\nSummary:\n", result)
```

---

## 🧠 Tips

- The `chunk_size` helps split large documents into manageable pieces for summarization.
- If you use a local model, make sure your system has enough memory to run it.
- The `page_content_field` must match the structure of your MongoDB documents.

---

## 📘 Need a Model?
If you don’t have a `.gguf` model yet, check out [INSTALL_DOLPHIN_MISTRAL.md](INSTALL_DOLPHIN_MISTRAL.md) for setup instructions.

---

That’s it! You’ve now connected MongoDB with a language model for AI summarization.

