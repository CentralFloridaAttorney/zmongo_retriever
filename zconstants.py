import os

from dotenv import load_dotenv
# This selects the model paths for the system
setup_for = "alienware"  # "omen", "alienware", "predator", "conversex"
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

# Model Path Value
if setup_for == "omen":
    MODEL_PATH = "/media/overlordx/DATA/_models/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_0.gguf"
    STABLE_DIFFUSION_MODEL_CKPT = "CompVis/stable-diffusion-v1-4"
elif setup_for == "alienware":
    MODEL_PATH = "D:/_models/mistral-7b-instruct-v0.1.Q4_0.gguf"
    STABLE_DIFFUSION_MODEL_CKPT = "D:/_models/v1-5-pruned-emaonly.ckpt"
elif setup_for == "conversex":
    MODEL_PATH = "/mnt/storage/models/dolphin-2.1-mistral-7B-GGUF/dolphin-2.1-mistral-7b.Q4_0.gguf"
elif setup_for == "predator":
    MODEL_PATH = "/mnt/storage/models/dolphin-2.1-mistral-7B-GGUF/dolphin-2.1-mistral-7b.Q4_0.gguf"
    STABLE_DIFFUSION_MODEL_CKPT = "/mnt/storage/models/stable-diffusion-v1-5/"

else:
    MODEL_PATH = os.path.join(PROJECT_PATH, 'Mistral-7B-Instruct-v0.1-GGUF', 'mistral-7b-instruct-v0.1.Q4_0.gguf')


OPENAI_ENGINE_NAME = "gpt-3.5-turbo-1106"
DEFAULT_REQUEST = 'You are an expert drafter of legal documents.  Write well organized, detailed analysis of the case.'
MONGO_DATABASE_NAME = "zcases_032124"
DEFAULT_COLLECTION_NAME = "zcases"
PAGE_CONTENT_KEY = 'casebody.data.opinions.0.text'
MAX_TOKENS = 4096

# used in class sentry_cli_installer.SentryCLIInstaller:
SENTRY_CLI_VERSION = '0.0.1'

OCR_LOOP_DELAY = 5
LLM_LOOP_DELAY = 5
# ZMongoEmbedder Values
EMBEDDING_CONTEXT_LENGTH = 1536
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_MODEL = "text-embedding-ada-002"

# zassistant.zcase_summary_template Values
ZCASE_SUMMARY_TEMPLATE_HOST = "0.0.0.0"
ZCASE_SUMMARY_TEMPLATE_PORT = 49996

# flask_app.py Values
FRONTEND_PORT = 50000
FRONTEND_HOST = "0.0.0.0"

# Imagine Values
IMAGINE_DEFAULT_DEVICE = "cuda"
IMAGINE_DEFAULT_PROMPT = "a photograph of a cute puppy"
IMAGINE_DEFAULT_FILE_IDENTIFIER = "puppy"
IMAGINE_HEIGHT = 312
IMAGINE_WIDTH = 312
IMAGINE_INFERENCE_STEPS = 50
IMAGINE_PROMPT_STRENGTH = 10.0
IMAGINE_MULTIPLE = False
IMAGE_REQUEST_DB = "image_request"
IMAGINE_OUTPUT_DIR = os.path.join(PROJECT_PATH, 'imagine_output')
IMAGINE_MANAGER_SLEEP_TIME = 10  # seconds


# ChromaDB specific configurations
# Assuming you might need these based on the context of using ChromaDB and MongoDB
# Add these if you have specific environment variables for ChromaDB
# Example placeholders (replace with actual keys or paths if needed)
CHROMA_DB_DIR = os.path.join(PROJECT_PATH, 'zconstants', 'chroma_db')
CHROMA_SERVER_HOST = 'localhost'
CHROMA_SERVER_PORT = 8000
CHROMA_SERVER_AUTH_CREDENTIALS_FILE = 'server.htpasswd'

# Environment Variables
env_path = os.path.join(PROJECT_PATH, '.env')
load_dotenv(env_path)
CASE_LAW_API_KEY = os.getenv("CASE_LAW_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

PROJECT_BACKUP_DIR = "zcase_backups"
BASE_DIR = "zcases"  # omen
THUMBNAIL_DIR = os.path.join(PROJECT_PATH, 'static', 'thumbnails')
IMAGE_DIR = os.path.join(PROJECT_PATH, 'static', 'images')
VIDEO_DIR = os.path.join(PROJECT_PATH, 'static', 'videos')
DOCUMENTS_DIR = os.path.join(PROJECT_PATH, 'flask_backend', 'static', 'document')
OUTPUT_DIR = os.path.join(PROJECT_PATH, 'flask_backend', 'static', 'output')

# MongoDB Values
MONGO_URI = "mongodb://localhost:49999"
MONGO_BACKUP_DIR = "mongo_backups"
CAUSES_OF_ACTION_COLLECTION = "causes_of_action"
CHATS_COLLECTION = "chats"
ZElement_COLLECTION = "zelement"
ZElement_TYPES_COLLECTION = "doi_types"
DOCUMENT_COLLECTION = "document"
DOCKET_TYPES_COLLECTION = "docket_types"
DOCTYPES_COLLECTION = "doctypes"
FACT_TYPES_COLLECTION = "fact_types"
FLASK_LOGS_COLLECTION = "flask_logs"
LEGALDOC_TAGS_COLLECTION = "legaldoc_tags"
LEGALDOCS_COLLECTION = "legaldocs"
LOGS_COLLECTION = "logs"
ROLES_COLLECTION = "roles"
ULTIMATE_FACT_TYPES_COLLECTION = "ultimate_fact_types"
USER_COLLECTION = "user"
ZASSISTANT_LOGS_COLLECTION = "zassistant_logs"
ZCASES_COLLECTION = "zcases"
