import os
import urllib.request
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class LlamaModel:
    """
    A class to interact with the Llama.cpp model using llama-cpp-python.
    """

    def __init__(self):
        self.model_path = os.getenv(
            "GGUF_MODEL_PATH",
        )
        self.model_url = os.getenv("GGML_MODEL_URL")
        self.n_ctx = int(os.getenv("N_CTX", 512))
        self.n_batch = int(os.getenv("N_BATCH", 126))
        self.llm = None

        # if self.model_url:
        #     self.download_model()
        self.load_model()

    def download_model(self):
        """
        Downloads the model file if it does not already exist.
        """
        if not os.path.isfile(self.model_path):
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading model...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            print("Model downloaded successfully.")
        else:
            print("Model file already exists.")

    def load_model(self):
        """
        Loads the model using llama-cpp-python.
        """
        print("Loading model...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch
        )
        print("Model loaded successfully.")

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


# Example usage
def main():
    llama_model = LlamaModel()

    user_input = (
        "Write a Dungeons & Dragons encounter using D20 rules, include full descriptive text for the dungeon master to read when running the encounter.  This is for new dungeon masters."
        "The adventurers awake from a drunken slumber in the corner of a tavern, ."
    )

    prompt = llama_model.generate_prompt_from_template(user_input)

    output_text = llama_model.generate_text(
        prompt=prompt,
        max_tokens=30000,
    )
    print("Generated Text:\n")
    print(output_text)


if __name__ == "__main__":
    main()
