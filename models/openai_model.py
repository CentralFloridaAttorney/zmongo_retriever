import os
import openai
import asyncio
from typing import Optional, List, Any, Union, Dict

from bson.objectid import ObjectId
from dotenv import load_dotenv
from threading import Lock

from zmongo_toolbag.zmongo import ZMongo

# Load environment variables
load_dotenv(Path.home() / "resources" / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")


class SingletonMeta(type):
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
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        self.max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", 256))
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
        self.top_p = float(os.getenv("DEFAULT_TOP_P", 0.95))
        self._lock = asyncio.Lock()

    async def _call_openai_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> str:
        async with self._lock:
            try:
                response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    top_p=top_p or self.top_p,
                    stream=stream
                )

                if stream:
                    return ''.join([chunk.choices[0].delta.get("content", "") for chunk in response])
                return response.choices[0].message.content.strip()

            except Exception as e:
                return f"[OpenAI Error] {e}"

    async def generate_instruction(self, user_instruction: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_instruction}
        ]
        return await self._call_openai_chat(messages)

    async def generate_summary(self, raw_text: str) -> str:
        messages = [
            {"role": "system", "content": "Summarize the following text."},
            {"role": "user", "content": raw_text}
        ]
        return await self._call_openai_chat(messages, max_tokens=200)

    async def generate_question_answer(self, context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a question answering assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        return await self._call_openai_chat(messages, max_tokens=150)

    async def generate_zelement_explanation(self, zelement_doc: dict) -> str:
        name = zelement_doc.get("name", "Unnamed")
        note = zelement_doc.get("note", "No notes.")
        creator = zelement_doc.get("creator", "Unknown")

        messages = [
            {"role": "system", "content": "Explain the purpose of a ZElement entry."},
            {"role": "user", "content": f"Name: {name}\nNote: {note}\nCreator: {creator}"}
        ]
        return await self._call_openai_chat(messages)

    async def generate_from_template(self, template: str, variables: dict) -> str:
        filled_prompt = template.format(**variables)
        messages = [{"role": "user", "content": filled_prompt}]
        return await self._call_openai_chat(messages)

    async def save_openai_result(self,
                                 collection_name: str,
                                 record_id: Union[str, ObjectId],
                                 field_name: str,
                                 generated_text: str,
                                 extra_fields: Optional[dict[str, Any]] = None,
                                 zmongo: Optional[ZMongo] = None) -> bool:
        if not generated_text or not field_name:
            raise ValueError("Generated text and field name must be provided.")

        if isinstance(record_id, str):
            record_id = ObjectId(record_id)

        should_close = False
        if zmongo is None:
            zmongo = ZMongo()
            should_close = True  # We created it here, so we should close it.

        update_data = {"$set": {field_name: generated_text}}

        if extra_fields:
            update_data["$set"].update(extra_fields)

        try:
            result = await zmongo.update_document(
                collection=collection_name,
                query={"_id": record_id},
                update_data=update_data,
                upsert=True,
            )
            return result.matched_count > 0 or result.upserted_id is not None
        finally:
            if should_close:
                zmongo.mongo_client.close()
