import os
import urllib.request
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LlamaModel:
    """
    A class to interact with the Llama.cpp model using llama-cpp-python.

    This class expects the following environment variables to be set in your .env file:
      - GGUF_MODEL_PATH: The local path to the model file.
      - GGML_MODEL_URL: The URL to download the model file if it's missing.
      - N_CTX: (Optional) Context size for the model (default: 512).
      - N_BATCH: (Optional) Batch size for processing (default: 126).
    """

    def __init__(self):
        # Retrieve critical environment variables
        self.model_path = os.getenv("GGUF_MODEL_PATH")
        if not self.model_path:
            print("ERROR: GGUF_MODEL_PATH environment variable is missing. Please include it in your .env file.")
            raise ValueError("Missing GGUF_MODEL_PATH environment variable.")

        self.model_url = os.getenv("GGML_MODEL_URL")
        if not self.model_url:
            print("ERROR: GGML_MODEL_URL environment variable is missing. Please include it in your .env file.")
            raise ValueError("Missing GGML_MODEL_URL environment variable.")

        # Retrieve optional parameters with defaults
        try:
            self.n_ctx = int(os.getenv("N_CTX", 512))
        except ValueError:
            print("ERROR: N_CTX environment variable must be an integer. Using default value 512.")
            self.n_ctx = 512

        try:
            self.n_batch = int(os.getenv("N_BATCH", 126))
        except ValueError:
            print("ERROR: N_BATCH environment variable must be an integer. Using default value 126.")
            self.n_batch = 126

        self.llm = None

        # Attempt to load the model; if the file doesn't exist, the load_model function will provide guidance.
        self.load_model()

    def download_model(self):
        """
        Downloads the model file if it does not already exist.
        """
        if not os.path.isfile(self.model_path):
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading model...")
            try:
                urllib.request.urlretrieve(self.model_url, self.model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to download the model from {self.model_url}. {e}")
                raise e
        else:
            print("Model file already exists.")

    def load_model(self):
        """
        Loads the model using llama-cpp-python.

        Before attempting to load, it checks if the model file exists.
        """
        print("Loading model...")
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}.")
            print("Please ensure that GGUF_MODEL_PATH in your .env file points to a valid model file,")
            print("or run the download_model() method to download it from GGML_MODEL_URL.")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch
            )
            print("Model loaded successfully.")
        except Exception as e:
            print("ERROR: Failed to load the model.")
            print(
                "Please check that the model file is valid and that all required environment variables are set correctly.")
            print(f"Exception: {e}")
            raise e

    def generate_prompt_from_template(self, user_input: str) -> str:
        """
        Generates a prompt using a predefined template.

        Args:
            user_input (str): The user input to include in the prompt.

        Returns:
            str: The formatted prompt.
        """
        chat_prompt_template = (
            "<|im_start|>system\n"
            "You are a helpful chatbot.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_input}<|im_end|>"
        )
        return chat_prompt_template

    def generate_text(
            self,
            prompt: str,
            max_tokens: int = None,
            temperature: float = None,
            top_p: float = None,
            echo: bool = False,
            stop: list = None
    ) -> str:
        """
        Generates text using the LLM based on the provided prompt and parameters.

        Args:
            prompt (str): The input prompt for the model.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature.
            top_p (float, optional): Nucleus sampling probability.
            echo (bool, optional): Whether to echo the prompt in the output.
            stop (list, optional): Stop tokens to halt generation.

        Returns:
            str: The generated text.
        """
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
        output_text = output["choices"][0]["text"].strip()
        return output_text
