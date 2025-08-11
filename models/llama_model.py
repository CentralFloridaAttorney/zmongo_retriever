import os
from typing import Optional, Union, Any
from bson.objectid import ObjectId
from dotenv import load_dotenv

from llama_cpp import Llama
from zmongo_toolbag.zmongo import ZMongo
from models.base_result_saver import BaseResultSaver

# Load environment variables from .env file
load_dotenv(Path.home() / "resources" / ".env")


class LlamaModel(BaseResultSaver):
    """
    A class to interact with the Llama.cpp model using llama-cpp-python,
    and to save results using ZMongo via the BaseResultSaver interface.
    """

    def __init__(self, zmongo: Optional[ZMongo] = None):
        # Retrieve required environment variables
        self.model_path = os.getenv("GGUF_MODEL_PATH")
        if not self.model_path:
            print("ERROR: GGUF_MODEL_PATH environment variable is missing. Please include it in your .env file.")
            raise ValueError("Missing GGUF_MODEL_PATH environment variable.")

        try:
            self.n_ctx = int(os.getenv("N_CTX", 512))
        except ValueError:
            print("ERROR: N_CTX must be an integer. Using default 512.")
            self.n_ctx = 512

        try:
            self.n_batch = int(os.getenv("N_BATCH", 126))
        except ValueError:
            print("ERROR: N_BATCH must be an integer. Using default 126.")
            self.n_batch = 126

        self.llm = None
        self.load_model()

        self.zmongo = zmongo or ZMongo()
        self._should_close = zmongo is None

    def load_model(self):
        print("Loading model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load the model: {e}")
            raise e

    def generate_prompt_from_template(self, user_input: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful chatbot.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_input}<|im_end|>"
        )

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        echo: bool = False,
        stop: list = None
    ) -> str:
        max_tokens = max_tokens or int(os.getenv("DEFAULT_MAX_TOKENS", 256))
        temperature = temperature or float(os.getenv("DEFAULT_TEMPERATURE", 0.1))
        top_p = top_p or float(os.getenv("DEFAULT_TOP_P", 0.5))
        stop = stop or os.getenv("DEFAULT_STOP", "#").split(",")

        output = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
        )
        return output["choices"][0]["text"].strip()

    async def save(
        self,
        collection_name: str,
        record_id: Union[str, ObjectId],
        field_name: str,
        generated_text: str,
        extra_fields: Optional[dict[str, Any]] = None
    ) -> bool:
        if not generated_text or not field_name:
            raise ValueError("Generated text and field name must be provided.")

        if isinstance(record_id, str):
            record_id = ObjectId(record_id)

        update_data = {"$set": {field_name: generated_text}}
        if extra_fields:
            update_data["$set"].update(extra_fields)

        try:
            result = await self.zmongo.update_document(
                collection=collection_name,
                query={"_id": record_id},
                update_data=update_data,
                upsert=True,
            )
            return result.matched_count > 0 or result.upserted_id is not None
        finally:
            if self._should_close:
                self.zmongo.mongo_client.close()
