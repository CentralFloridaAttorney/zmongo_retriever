# Python script: Combine ZRetriever with OpenAIModel to query MongoDB and generate completions

import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from bson import ObjectId
from dotenv import load_dotenv

from zmongo_retriever.zmongo_toolbag import LlamaModel
from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo
from zmongo_retriever.zmongo_toolbag.zretriever import ZRetriever
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY_APP")

class OpenAIModel:
    def __init__(self):
        self.model = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo-instruct")
        self.max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", 256))
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
        self.top_p = float(os.getenv("DEFAULT_TOP_P", 0.95))

    async def _call_openai(self, prompt: str, max_tokens=None, temperature=None, top_p=None, stop=None, echo=False) -> str:
        try:
            response = await asyncio.to_thread(openai.completions.create,
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                stop=stop,
                echo=echo,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"[OpenAI Error] {e}"

    async def summarize_text(self, text: str) -> str:
        prompt = f"Summarize the following for a legal researcher:\n\n{text}\n\nSummary:"
        return await self._call_openai(prompt, max_tokens=200)

async def run_openai_zretriever_pipeline():
    repo = ZMongo()
    retriever = ZRetriever(repository=repo, max_tokens_per_set=4096, chunk_size=512)
    collection_name = 'documents'
    document_ids = ['67e5ba645f74ae46ad39929d', '67ef0bd71a349c7c108331a6']

    documents = await retriever.invoke(collection=collection_name, object_ids=document_ids, page_content_key='text')
    openai_model = OpenAIModel()

    for idx, doc_set in enumerate(documents):
        combined_text = "\n\n".join(doc.page_content for doc in doc_set)
        print(f"\nDocument Set {idx + 1} (retrieved {len(doc_set)} chunks):")
        summary = await openai_model.summarize_text(combined_text[:1000])
        print("\nSummary:")
        print(summary)

if __name__ == "__main__":
    asyncio.run(run_openai_zretriever_pipeline())


# --------------------------- TESTS ----------------------------

class TestLlamaModel(unittest.TestCase):


    def test_llama_model_real(self):

        model = LlamaModel()
        prompt = model.generate_prompt_from_template("Describe the 2021 Ford Mustang Mach-E.")
        output = model.generate_text(prompt=prompt, max_tokens=128)

        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 10)  # Adjust threshold as needed


if __name__ == "__main__":
    unittest.main()
