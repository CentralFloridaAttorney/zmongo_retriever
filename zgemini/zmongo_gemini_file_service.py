import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from google.api_core import exceptions
import google.generativeai as genai

# Assuming your toolbag is structured as a package or in the same directory
from zmongo import ZMongo

# --- Configuration ---
load_dotenv(Path.home() / "resources" / ".env_local")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZMongoGeminiFileService:
    """
    A service class to interact with the Gemini Files API for uploading,
    managing, and using files in prompts.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, zmongo_instance: Optional[ZMongo] = None):
        """
        Initializes the file service.

        Args:
            gemini_api_key (Optional[str]): Your Google Gemini API key. If None,
                                            it's read from the environment.
            zmongo_instance (Optional[ZMongo]): An instance of ZMongo for any
                                                 database interactions if needed in the future.
        """
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in your environment or passed to the constructor.")

        # In the new SDK, the client is initialized via genai.configure
        genai.configure(api_key=api_key)

        self.zmongo = zmongo_instance or ZMongo()
        logger.info("ZMongoGeminiFileService initialized.")

    async def upload_file(self, file_path: str) -> Optional[genai.files.File]:
        """
        Uploads a file to the Gemini API.

        Args:
            file_path (str): The local path to the file to upload.

        Returns:
            Optional[genai.files.File]: The file object if successful, otherwise None.
        """
        try:
            logger.info(f"Uploading file: {file_path}")
            # The new SDK uses a synchronous upload method, so we run it in an executor
            loop = asyncio.get_running_loop()
            uploaded_file = await loop.run_in_executor(None, lambda: genai.upload_file(path=file_path))
            logger.info(f"Successfully uploaded file '{uploaded_file.name}' with URI '{uploaded_file.uri}'.")
            return uploaded_file
        except FileNotFoundError:
            logger.error(f"File not found at path: {file_path}")
            return None
        except Exception as e:
            logger.error(f"An error occurred during file upload: {e}")
            return None

    async def get_file_metadata(self, file_name: str) -> Optional[genai.files.File]:
        """
        Retrieves the metadata for a specific uploaded file.

        Args:
            file_name (str): The name of the file (e.g., 'files/abc-123').

        Returns:
            Optional[genai.files.File]: The file metadata object if found, otherwise None.
        """
        try:
            logger.info(f"Getting metadata for file: {file_name}")
            loop = asyncio.get_running_loop()
            file_metadata = await loop.run_in_executor(None, lambda: genai.get_file(name=file_name))
            return file_metadata
        except exceptions.NotFound:
            logger.warning(f"File not found: {file_name}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while getting file metadata: {e}")
            return None

    async def list_files(self) -> List[genai.files.File]:
        """
        Lists all files currently uploaded to the service.

        Returns:
            List[genai.files.File]: A list of file objects.
        """
        try:
            logger.info("Listing all uploaded files.")
            loop = asyncio.get_running_loop()
            # The list_files method is a generator, so we convert it to a list
            files_iterator = await loop.run_in_executor(None, genai.list_files)
            return list(files_iterator)
        except Exception as e:
            logger.error(f"An error occurred while listing files: {e}")
            return []

    async def delete_file(self, file_name: str) -> bool:
        """
        Deletes a file from the service.

        Args:
            file_name (str): The name of the file to delete.

        Returns:
            bool: True if deletion was successful, otherwise False.
        """
        try:
            logger.info(f"Deleting file: {file_name}")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: genai.delete_file(name=file_name))
            logger.info(f"Successfully deleted file: {file_name}")
            return True
        except exceptions.NotFound:
            logger.warning(f"File not found for deletion: {file_name}")
            return False
        except Exception as e:
            logger.error(f"An error occurred during file deletion: {e}")
            return False

    async def generate_content_with_file(self, prompt: str, file: genai.files.File,
                                         model_name: str = "gemini-1.5-flash") -> Optional[str]:
        """
        Generates content from a prompt that includes a file.

        Args:
            prompt (str): The text prompt.
            file (genai.files.File): The uploaded file object to include in the prompt.
            model_name (str): The name of the Gemini model to use.

        Returns:
            Optional[str]: The generated text response, or None on failure.
        """
        try:
            logger.info(f"Generating content with file '{file.name}' using model '{model_name}'.")
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async([prompt, file])
            return response.text
        except Exception as e:
            logger.error(f"An error occurred during content generation: {e}")
            return None


async def main():
    """Example usage of the ZMongoGeminiFileService."""
    # Create a dummy file for testing
    dummy_file_path = "sample_for_upload.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a sample file for testing the Gemini Files API.")

    try:
        service = ZMongoGeminiFileService()

        # 1. Upload a file
        print("\n--- 1. Uploading File ---")
        uploaded_file = await service.upload_file(dummy_file_path)
        if not uploaded_file:
            return

        # 2. Get metadata for the uploaded file
        print("\n--- 2. Getting File Metadata ---")
        metadata = await service.get_file_metadata(uploaded_file.name)
        if metadata:
            print(f"  - Name: {metadata.name}")
            print(f"  - Display Name: {metadata.display_name}")
            print(f"  - URI: {metadata.uri}")
            print(f"  - Size: {metadata.size_bytes} bytes")
            print(f"  - Mime Type: {metadata.mime_type}")

        # 3. List all files
        print("\n--- 3. Listing All Files ---")
        all_files = await service.list_files()
        if all_files:
            print(f"Found {len(all_files)} file(s):")
            for f in all_files:
                print(f"  - {f.name} ({f.display_name})")
        else:
            print("No files found.")

        # 4. Generate content using the file
        print("\n--- 4. Generating Content with File ---")
        prompt = "Summarize the content of this file in one sentence."
        response_text = await service.generate_content_with_file(prompt, uploaded_file)
        if response_text:
            print(f"Model Response: {response_text}")

        # 5. Delete the file
        print("\n--- 5. Deleting File ---")
        delete_success = await service.delete_file(uploaded_file.name)
        if delete_success:
            # Verify deletion
            remaining_files = await service.list_files()
            print(f"Files remaining after deletion: {len(remaining_files)}")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}", exc_info=True)
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    # To run this example, ensure your GEMINI_API_KEY is set in your .env file
    asyncio.run(main())
