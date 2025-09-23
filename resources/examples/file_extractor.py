"""
Document Text Extractor and MongoDB Loader
==========================================

This script extracts text content from .txt, .docx, and .pdf files found
within the given file or directory paths and stores the result in a MongoDB
'documents' collection using the ZMongo helper.

It prevents duplicate entries by generating a SHA256 hash of the content
and checking if that hash already exists in the database before insertion.

Usage:
    # 1. (One-time setup) Run the setup script to create the unique index:
    #    python setup_mongo.py
    #
    # 2. Process files or directories:
    #    python file_extractor.py /path/to/your/documents_folder
"""
import os
import asyncio
import argparse
import logging
import hashlib
from datetime import datetime, timezone

try:
    import docx
except ImportError:
    print("Warning: 'python-docx' is not installed. .docx processing will fail.")
    print("Install it with: pip install python-docx")
    docx = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: 'PyMuPDF' is not installed. .pdf processing will fail.")
    print("Install it with: pip install PyMuPDF")
    fitz = None

# Local imports from your provided files
from zmongo import ZMongo
from data_processing import SafeResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> SafeResult:
    """
    Extracts text from a file based on its extension.

    Args:
        file_path: The full path to the file.

    Returns:
        A SafeResult object containing the extracted text on success,
        or an error message on failure.
    """
    if not os.path.exists(file_path):
        return SafeResult.fail(f"File not found: {file_path}")

    _, extension = os.path.splitext(file_path.lower())
    logger.info(f"Attempting to extract text from '{os.path.basename(file_path)}'...")

    try:
        text = ""
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif extension == '.docx':
            if not docx:
                return SafeResult.fail("python-docx is not installed.")
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif extension == '.pdf':
            if not fitz:
                return SafeResult.fail("PyMuPDF is not installed.")
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
        else:
            return SafeResult.fail(f"Unsupported file type: {extension}")

        logger.info(f"Successfully extracted {len(text)} characters.")
        return SafeResult.ok(text)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return SafeResult.fail(f"Error processing {file_path}: {e}", exc=e)


async def main(file_paths: list[str]):
    """
    Main async function to process a list of files.
    """
    db = ZMongo()
    try:
        ping_result = await db.ping()
        if not ping_result.success:
            logger.error(f"Could not connect to MongoDB: {ping_result.error}")
            return
        logger.info("Successfully connected to MongoDB.")

        for path in file_paths:
            extraction_result = extract_text(path)

            if not extraction_result.success:
                logger.error(f"Skipping file due to extraction error: {extraction_result.error}")
                continue

            content = extraction_result.data
            if not content or not content.strip():
                logger.warning(f"Skipping file with no extracted content: {os.path.basename(path)}")
                continue

            # Create a SHA256 hash of the content to use for duplicate checking.
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Check if a document with this content hash already exists.
            query = {"content_hash": content_hash}
            existing_doc_result = await db.find_document("documents", query, projection={"_id": 1})

            if not existing_doc_result.success:
                logger.error(f"DB check failed for {os.path.basename(path)}: {existing_doc_result.error}. Skipping.")
                continue

            if existing_doc_result.data:
                existing_id = existing_doc_result.data.get('_id')
                logger.info(f"Duplicate content found for '{os.path.basename(path)}'. Record exists (ID: {existing_id}). Skipping.")
                continue

            document_to_store = {
                "filename": os.path.basename(path),
                "full_path": os.path.abspath(path),
                "content": content,
                "content_hash": content_hash,
                "char_count": len(content),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"Storing extracted text from '{document_to_store['filename']}' to MongoDB...")
            insert_result = await db.insert_document("documents", document_to_store)

            if insert_result.success:
                inserted_id = insert_result.data.get("inserted_id")
                logger.info(f"Successfully stored new document with ID: {inserted_id}")
            else:
                logger.error(f"Failed to store document: {insert_result.error}")
    finally:
        db.close()
        logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from .txt, .docx, and .pdf files and load into MongoDB."
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        type=str,
        nargs='+',
        help="One or more file or directory paths to process."
    )
    args = parser.parse_args()

    supported_extensions = ['.txt', '.docx', '.pdf']
    files_to_process = []

    for path in args.paths:
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                files_to_process.append(path)
        elif os.path.isdir(path):
            logger.info(f"Scanning directory: {path}")
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in supported_extensions:
                        files_to_process.append(os.path.join(root, filename))
        else:
            logger.warning(f"Path is not valid, skipping: {path}")

    if not files_to_process:
        logger.info("No supported files found to process.")
        exit()

    extensions_found = {os.path.splitext(f)[1].lower() for f in files_to_process}
    missing_lib = False
    if '.docx' in extensions_found and not docx:
        logger.error("Found .docx file(s), but 'python-docx' is not installed.")
        missing_lib = True
    if '.pdf' in extensions_found and not fitz:
        logger.error("Found .pdf file(s), but 'PyMuPDF' is not installed.")
        missing_lib = True

    if not missing_lib:
        logger.info(f"Found {len(files_to_process)} file(s) to process.")
        asyncio.run(main(files_to_process))
    else:
        logger.error("Exiting due to missing libraries.")

