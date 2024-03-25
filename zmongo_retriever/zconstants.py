import os

from dotenv import load_dotenv

setup_for = "predator"  # "omen", "alienware", "predator", "conversex"

# used in class sentry_cli_installer.SentryCLIInstaller:
SENTRY_CLI_VERSION = '0.0.1'

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

# MongoDB Values
MONGO_URI = "mongodb://localhost:49999"
MONGO_DATABASE_NAME = "zcases"
DEFAULT_COLLECTION_NAME = "zcases"
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


# Project Values
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


try:
    PROJECT_PATH = get_project_root()
except:
    PROJECT_PATH = ''

# Model Path
if setup_for == "omen":
    MODEL_PATH = os.path.join(PROJECT_PATH, 'zassistant', 'Mistral-7B-Instruct-v0.1-GGUF', 'mistral-7b-instruct-v0.1.Q4_0.gguf')
elif setup_for == "alienware":
    MODEL_PATH = "D:/_models/mistral-7b-instruct-v0.1.Q4_0.gguf"
elif setup_for == "conversex":
    MODEL_PATH = "/mnt/storage/models/dolphin-2.1-mistral-7B-GGUF/dolphin-2.1-mistral-7b.Q4_0.gguf"
elif setup_for == "predator":
    MODEL_PATH = "/mnt/storage/models/dolphin-2.1-mistral-7B-GGUF/dolphin-2.1-mistral-7b.Q4_0.gguf"
else:
    MODEL_PATH = os.path.join(PROJECT_PATH, 'zassistant', 'Mistral-7B-Instruct-v0.1-GGUF', 'mistral-7b-instruct-v0.1.Q4_0.gguf')

# ChromaDB specific configurations
CHROMA_DB_DIR = os.path.join(PROJECT_PATH, 'chroma_backups')
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
DOCUMENTS_DIR = "document"
