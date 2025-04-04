import os
import openai
import asyncio
from typing import Optional, List, Any, Union

from bson import ObjectId
from dotenv import load_dotenv
from threading import Lock

from zmongo_retriever.zmongo_toolbag.zmongo import ZMongo

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY_APP")


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class OpenAIModel(metaclass=SingletonMeta):
    def __init__(self):
        self.model = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo-instruct")
        self.max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", 256))
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
        self.top_p = float(os.getenv("DEFAULT_TOP_P", 0.95))
        self._lock = asyncio.Lock()

    async def _call_openai(self,
                           prompt: str,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           stop: Optional[List[str]] = None,
                           echo: bool = False,
                           stream: bool = False) -> str:
        async with self._lock:
            try:
                response = await asyncio.to_thread(openai.completions.create,
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    top_p=top_p or self.top_p,
                    stop=stop,
                    echo=echo,
                    stream=stream,
                )

                if stream:
                    return ''.join([chunk.choices[0].text for chunk in response])
                return response.choices[0].text.strip()

            except Exception as e:
                return f"[OpenAI Error] {e}"

    async def generate_instruction(self, user_instruction: str) -> str:
        prompt = f"You are a helpful assistant.\n\nUser: {user_instruction}\nAssistant:"
        return await self._call_openai(prompt)

    async def generate_summary(self, raw_text: str) -> str:
        prompt = f"Summarize the following:\n\n{raw_text}\n\nSummary:"
        return await self._call_openai(prompt, max_tokens=200)

    async def generate_question_answer(self, context: str, question: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return await self._call_openai(prompt, max_tokens=150)

    async def generate_zelement_explanation(self, zelement_doc: dict) -> str:
        name = zelement_doc.get("name", "Unnamed")
        note = zelement_doc.get("note", "No notes.")
        creator = zelement_doc.get("creator", "Unknown")

        prompt = f"Explain the purpose of a ZElement entry.\nName: {name}\nNote: {note}\nCreator: {creator}\nExplanation:"
        return await self._call_openai(prompt)

    async def generate_from_template(self, template: str, variables: dict) -> str:
        prompt = template.format(**variables)
        return await self._call_openai(prompt)

    async def save_openai_result(self,
                                 collection_name: str,
                                 record_id: Union[str, ObjectId],
                                 field_name: str,
                                 generated_text: str,
                                 extra_fields: Optional[dict[str, Any]] = None) -> bool:
        """
        Updates a MongoDB document by inserting OpenAI-generated text into a specified field.
        """
        if not generated_text or not field_name:
            raise ValueError("Generated text and field name must be provided.")

        if isinstance(record_id, str):
            record_id = ObjectId(record_id)

        zmongo = ZMongo()
        update_data = {"$set": {field_name: generated_text}}

        if extra_fields:
            update_data["$set"].update(extra_fields)

        result = await zmongo.update_document(
            collection=collection_name,
            query={"_id": record_id},
            update_data=update_data
        )

        # Fix: Check success using returned dict instead of accessing .acknowledged
        return result.get("matchedCount", 0) > 0 or result.get("upsertedId") is not None


