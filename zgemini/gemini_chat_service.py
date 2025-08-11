import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.home() / "resources" / ".env_local")

# Import necessary classes from zmongo_service.py
from zmongo_toolbag.zmongo_service import ZMongoService, SafeResult, Document
import google.generativeai as genai
import nest_asyncio
from datetime import datetime

# Apply the nest_asyncio patch for running nested event loops
nest_asyncio.apply()


class GeminiChatService:
    """
    A service that uses a ZMongoService instance to find relevant
    documents and then uses the Gemini API to formulate a response.
    It also saves the chat history and references to the database.
    """

    # Define a threshold for when to rephrase the user's query.
    # A high score indicates that a highly-relevant document was found.
    REPHRASE_THRESHOLD = 0.90

    def __init__(self, zmongo_service: ZMongoService, gemini_api_key: str):
        """
        Initializes the chat service.

        Args:
            zmongo_service (ZMongoService): The service instance to use for
                                            database interactions and searching.
            gemini_api_key (str): Your API key for the Gemini API.
        """
        self.zmongo_service = zmongo_service
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        self.repository = self.zmongo_service.repository
        logging.info("GeminiChatService initialized.")

    async def _generate_response(self, chat_prompt: str) -> SafeResult:
        """
        Generates a response from the Gemini API.

        Args:
            chat_prompt (str): The prompt string to send to the LLM.

        Returns:
            SafeResult: A SafeResult object containing the generated text
                        or an error message.
        """
        try:
            response = await self.model.generate_content_async(chat_prompt)
            # The API returns a 'candidates' list. We take the first one.
            if response.candidates and response.candidates[0].content:
                text_response = response.candidates[0].content.parts[0].text
                return SafeResult.ok(text_response)
            return SafeResult.fail("Gemini API returned an empty response.")
        except Exception as e:
            return SafeResult.fail(error=f"Gemini API error: {str(e)}", data=e)

    async def chat(self,
                   user_id: str,
                   prompt: str,
                   knowledge_base_collection: str,
                   min_score_threshold: float = 0.50
                   ) -> (SafeResult, SafeResult, dict):
        """
        Performs a full chat interaction with the user.
        ...
        Returns:
            (SafeResult, SafeResult, dict): A tuple containing the LLM's
                                            response, the save result, and
                                            the chat record dictionary.
        """
        logging.info(f"Chat initiated by user '{user_id}' with prompt: '{prompt}'")

        # Step 1: Search the knowledge base for relevant documents
        search_results: list[Document] = await self.zmongo_service.search(
            collection_name=knowledge_base_collection,
            query_text=prompt,
            similarity_threshold=min_score_threshold
        )

        # Step 2: Conditionally rephrase the user's prompt
        final_prompt_for_llm = prompt
        rephrased_prompt_used = False
        if search_results and search_results[0].metadata.get('retrieval_score', 0.0) > self.REPHRASE_THRESHOLD:
            top_doc_content = search_results[0].page_content
            rephrase_instructions = (
                f"Rephrase the following user question to be more similar to the provided context. "
                f"Original question: '{prompt}'\n\nContext: '{top_doc_content}'\n\nRephrased question:"
            )
            rephrase_res = await self._generate_response(rephrase_instructions)
            if rephrase_res.success and rephrase_res.data:
                final_prompt_for_llm = rephrase_res.data
                rephrased_prompt_used = True
                logging.info(f"Prompt rephrased from '{prompt}' to '{final_prompt_for_llm}'")

        # Step 3: Formulate the final prompt for the LLM with context
        context = ""
        references = []
        if search_results:
            context_docs = [f"### Context from '{doc.metadata.get('title')}':\n{doc.metadata.get('content')}"
                            for doc in search_results]
            context = "\n\n".join(context_docs)
            references = [
                {
                    "document_id": doc.metadata.get('source_document_id'),
                    "score": doc.metadata.get('retrieval_score'),
                    "title": doc.metadata.get('title'),
                    "page_content": doc.metadata.get('content'),
                } for doc in search_results
            ]

        llm_prompt = (
            f"You are a helpful assistant. Use the following context to answer the user's question. "
            f"If the answer is not in the context, say that you don't have enough information.\n\n"
            f"{context}\n\nUser: {final_prompt_for_llm}"
        )

        # Step 4: Generate a response from the LLM
        llm_res = await self._generate_response(llm_prompt)

        chat_record = {
            "user_id": user_id,
            "prompt": prompt,
            "rephrased_prompt": final_prompt_for_llm if rephrased_prompt_used else None,
            "response": llm_res.data if llm_res.success else llm_res.error,
            "timestamp": datetime.now(),
            "references": references
        }

        # Step 5: Save the chat history to a collection
        save_res = await self.repository.insert_document("chat_history", chat_record)

        if not save_res.success:
            logging.error(f"Failed to save chat record: {save_res.error}")

        return llm_res, save_res, chat_record


async def main():
    """
    Main function to demonstrate the GeminiChatService.
    """

    # Load environment variables
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not all([MONGO_URI, MONGO_DATABASE_NAME, GEMINI_API_KEY]):
        print("Please set MONGO_URI, MONGO_DATABASE_NAME, and GEMINI_API_KEY in your .env file.")
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the core services
    try:
        zmongo_service = ZMongoService(
            mongo_uri=MONGO_URI,
            db_name=MONGO_DATABASE_NAME,
            gemini_api_key=GEMINI_API_KEY
        )
        chat_service = GeminiChatService(zmongo_service=zmongo_service, gemini_api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        return

    # User-specific information
    test_user_id = "user_123"
    knowledge_base_collection = "my_knowledge_base"

    # Ensure there is at least one document in the knowledge base
    print("--- Adding a sample document to the knowledge base ---")
    doc_to_add = {
        "title": "A Guide to Pizza",
        "content": "Gino's Pizza Restaurant has the World's Best Pepperoni pizza."
    }
    await zmongo_service.add_and_embed(knowledge_base_collection, doc_to_add, text_field="content")

    try:
        # --- Simulate a chat interaction ---
        print("\n--- Starting a chat with GeminiChatService ---")
        user_prompt = "What is the Pacific Ocean?"
        llm_res, save_res, chat_record = await chat_service.chat(
            user_id=test_user_id,
            prompt=user_prompt,
            knowledge_base_collection=knowledge_base_collection,
            min_score_threshold=0.10
        )

        print("\n--- Chat Results ---")
        if llm_res.success:
            print(f"Gemini Response: {llm_res.data}")
        else:
            print(f"Error from Gemini: {llm_res.error}")

        if save_res.success:
            print(f"Chat record saved with ID: {save_res.data.inserted_id}")
            # --- FIX: Access the rephrased prompt from the chat_record dict ---
            print(f"Rephrased prompt (if used): {chat_record.get('rephrased_prompt')}")
        else:
            print(f"Failed to save chat record: {save_res.error}")

    except Exception as e:
        print(f"\nAn error occurred during chat: {e}")
    finally:
        # Close the connection when done
        print("\n--- Closing Connection ---")
        await zmongo_service.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
