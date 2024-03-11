from dotenv import load_dotenv


setup_for = "omen" # "omen", "alienware" -- need to add "predator"



# ZMongoEmbedder Values
EMBEDDING_CONTEXT_LENGTH = 1536
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_MODEL = "text-embedding-ada-002"

# zassistant.zcase_summary_template Values
ZCASE_SUMMARY_TEMPLATE_HOST = "0.0.0.0"
ZCASE_SUMMARY_TEMPLATE_PORT = 49996

# flask_app.py Values
FRONTEND_HOST = "192.168.1.228"  # omen 4
FRONTEND_PORT = 50000

import os


def get_project_root(start_dir=None):
    """
    Get the project root directory by recursively traversing parent directories until a directory is found
    whose parent does not contain an __init__.py file, effectively finding the root of the project.

    Parameters:
        start_dir (str, optional): The directory to start the search from. If not provided, the current working directory is used.

    Returns:
        str: The path to the project root directory.
    """
    if start_dir is None:
        start_dir = os.getcwd()

    current_dir = start_dir
    while True:
        parent_dir = os.path.dirname(current_dir)

        # Check if the parent directory does not contain an __init__.py file
        parent_contains_init = '__init__.py' in os.listdir(parent_dir)
        current_contains_init = '__init__.py' in os.listdir(current_dir)

        # If the current directory contains an __init__.py file but the parent does not, current is the project root
        if current_contains_init and not parent_contains_init:
            return current_dir

        # Move to the parent directory if not yet at the root directory
        if parent_dir == current_dir or not parent_contains_init:
            raise FileNotFoundError(
                "Failed to find the project root. Reached the filesystem root or a directory without __init__.py.")

        current_dir = parent_dir


# Project Values
try:
    PROJECT_PATH = get_project_root()
except:
    PROJECT_PATH = ''

PROJECT_BACKUP_DIR = "zcase_backups"
BASE_DIR = "zcases"  # omen
DOCUMENTS_DIR = "document"

# Environment Variables
env_path = os.path.join(PROJECT_PATH, '.env')
load_dotenv(env_path)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# Model Path Value
if setup_for == "omen":
    MODEL_PATH = os.path.join(PROJECT_PATH, 'zassistant', 'Mistral-7B-Instruct-v0.1-GGUF', 'mistral-7b-instruct-v0.1.Q4_0.gguf')
elif setup_for == "alienware":
    MODEL_PATH = "D:/_models/mistral-7b-instruct-v0.1.Q4_0.gguf"
else:
    MODEL_PATH = os.path.join(PROJECT_PATH, 'zassistant', 'Mistral-7B-Instruct-v0.1-GGUF', 'mistral-7b-instruct-v0.1.Q4_0.gguf')

# MongoDB Values
MONGO_URI = "mongodb://localhost:49999"
MONGO_DATABASE_NAME = "zcases"
DEFAULT_COLLECTION_NAME = "zcases"
ZCASES_COLLECTION = "zcases"

