import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from bson import ObjectId

from gemini_chat_service import GeminiChatService
# Adjust these imports to match your project's structure
from zmongo_toolbag.zmongo_service import ZMongoService

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Main Class ---

class ChatCorrector:
    """
    A service that runs a background loop to find and correct failed chat responses.
    """

    def __init__(self,
                 chat_service: GeminiChatService,
                 knowledge_base_collection: str,
                 interval_ms: int = 10000):
        """
        Initializes the ChatCorrector.

        Args:
            chat_service (GeminiChatService): An instance of the chat service.
            knowledge_base_collection (str): The name of the collection with knowledge base documents.
            interval_ms (int): The interval in milliseconds to check for failed responses.
        """
        self.chat_service = chat_service
        self.repository = chat_service.zmongo_service.repository
        self.knowledge_base_collection = knowledge_base_collection
        self.interval_seconds = interval_ms / 1000.0
        self._is_running = False
        self._task = None
        # This phrase is the key indicator of a failed response
        self.failure_phrase = "don't have enough information"

    async def _run_correction_cycle(self):
        """The main logic for a single correction cycle."""
        logging.info("Corrector: Checking for failed chat responses...")

        # Find chat records that contain the failure phrase and haven't been corrected yet.
        query = {
            "response": {"$regex": self.failure_phrase, "$options": "i"},
            "correction_attempted": {"$exists": False}
        }
        find_res = await self.repository.find_documents("chat_history", query)

        if not find_res.success or not find_res.data:
            logging.info("Corrector: No failed responses found.")
            return

        for record in find_res.data:
            try:
                record_id_obj = ObjectId(record["_id"])
            except Exception:
                logging.error(f"Corrector: Found invalid ObjectId string in chat history: {record['_id']}")
                continue

            user_id = record["user_id"]
            original_prompt = record["prompt"]
            logging.info(f"Corrector: Found failed response {record_id_obj} for user '{user_id}'. Retrying...")

            # Get the user's chat history prior to this failed message
            history_query = {
                "user_id": user_id,
                "timestamp": {"$lt": record["timestamp"]}
            }
            history_res = await self.repository.find_documents("chat_history", history_query, sort=[("timestamp", -1)])

            prior_context = ""
            if history_res.success and history_res.data:
                history_texts = [f"User: {h['prompt']}\nAssistant: {h['response']}" for h in reversed(history_res.data)]
                prior_context = "\n\n".join(history_texts)

            # Formulate a new, more informed prompt
            new_prompt = (
                f"Please try to answer the user's question again. "
                f"Use our previous conversation and the knowledge base for context.\n\n"
                f"--- Previous Conversation ---\n{prior_context}\n\n"
                f"--- Original Question ---\n{original_prompt}"
            )

            # Retry the chat call with the enhanced prompt
            llm_res, _, _ = await self.chat_service.chat(
                user_id=user_id,
                prompt=new_prompt,
                knowledge_base_collection=self.knowledge_base_collection
            )

            # Update the original record with the new response
            update_data = {
                "$set": {
                    "correction_attempted": True,
                    "corrected_response": llm_res.data if llm_res.success else f"Correction failed: {llm_res.error}",
                    "correction_timestamp": datetime.now()
                }
            }
            await self.repository.update_document("chat_history", {"_id": record_id_obj}, update_data)
            logging.info(f"Corrector: Updated record {record_id_obj} with new response.")

    async def _loop(self):
        """The main background loop."""
        while self._is_running:
            try:
                await self._run_correction_cycle()
            except asyncio.CancelledError:
                logging.info("Corrector loop cancelled.")
                break
            except Exception as e:
                logging.error(f"Corrector: An error occurred in the correction loop: {e}")
            await asyncio.sleep(self.interval_seconds)

    def start(self):
        """Starts the background correction loop."""
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._loop())
            logging.info("Chat Corrector started.")

    def stop(self):
        """Stops the background correction loop gracefully."""
        if self._is_running and self._task:
            self._is_running = False
            self._task.cancel()
            logging.info("Chat Corrector stopped.")


async def main():
    """
    A main function to demonstrate the ChatCorrector in action with a real failure scenario.
    """
    # --- 1. Setup Services ---
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("MONGO_DATABASE_NAME", "chat_corrector_demo_db")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not all([MONGO_URI, GEMINI_API_KEY]):
        print("Please set MONGO_URI and GEMINI_API_KEY in your environment.")
        return

    zmongo_service = ZMongoService(mongo_uri=MONGO_URI, db_name=DB_NAME, gemini_api_key=GEMINI_API_KEY)
    chat_service = GeminiChatService(zmongo_service=zmongo_service, gemini_api_key=GEMINI_API_KEY)

    kb_collection = "my_knowledge_base"
    history_collection = "chat_history"

    # Clean up previous runs
    await zmongo_service.repository.db.drop_collection(kb_collection)
    await zmongo_service.repository.db.drop_collection(history_collection)

    # --- 2. Create a Real Failure Scenario ---
    # Add an irrelevant document to the knowledge base first.
    await zmongo_service.add_and_embed(
        kb_collection,
        {"topic": "geography", "content": "The Pacific Ocean is the largest and deepest of the world's five oceans."},
        "content"
    )

    # Now, ask a question that cannot be answered from the current knowledge base.
    # This will generate a genuine "I don't have enough information" response.
    print("\n--- Generating a real failed chat response ---")
    await chat_service.chat("user_B", "What is the most popular pizza topping?", kb_collection)

    # Now, add the relevant information to the knowledge base.
    print("\n--- Updating knowledge base with relevant information ---")
    await zmongo_service.add_and_embed(
        kb_collection,
        {"topic": "food", "content": "Pepperoni is widely considered the most popular pizza topping in the world."},
        "content"
    )

    # --- 3. Run the Corrector ---
    corrector = ChatCorrector(chat_service, kb_collection, interval_ms=5000)
    corrector.start()

    print("\n--- Chat Corrector is running in the background. Waiting for it to work... ---")
    # Increase sleep time to allow for the network call to the Gemini API
    await asyncio.sleep(15)
    corrector.stop()

    # --- 4. Verify the Correction ---
    print("\n--- Verifying the results in the database ---")
    corrected_doc_res = await zmongo_service.repository.find_document(
        history_collection, {"correction_attempted": True}
    )

    if corrected_doc_res.success and corrected_doc_res.data:
        print("SUCCESS: Found a corrected chat record.")
        print(f"Original Prompt: {corrected_doc_res.data['prompt']}")
        print(f"Original Failed Response: {corrected_doc_res.data['response']}")
        print(f"Corrected Response: {corrected_doc_res.data['corrected_response']}")
    else:
        print("FAILURE: Did not find a corrected chat record.")

    await zmongo_service.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
