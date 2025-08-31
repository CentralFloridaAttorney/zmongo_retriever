# zonehotdb.py
# Python 3.10+
#
# =============================================================================
# ZOneHotDB: A Lossless Text-to-Index Conversion Library
# =============================================================================
"""
ZOneHotDB provides a robust system for converting arbitrary text documents into a
compact, indexed format that allows for perfect, character-for-character
reconstruction of the original content.

This library is designed for scenarios where preserving the exact original text,
including all whitespace, punctuation, and complex capitalization, is critical.
It uses a MongoDB backend (via the async ZMongo library) to efficiently store
and manage the vocabulary and encoded document data.

Core Features:
-------------
- **Lossless Reconstruction:** The primary design goal. Any text encoded can be
  decoded back to its exact original form.

- **Comprehensive Tokenizer:** A regular expression-based tokenizer captures
  not just words, but also all whitespace (spaces, newlines, tabs) and every
  individual punctuation mark as distinct tokens.

- **Efficient Vocabulary:** Stores only the lowercase version of each token in the
  database to prevent vocabulary bloat from case variations (e.g., "The" and
  "the" are a single entry).

- **Advanced Capitalization Mask:** A sophisticated system stores an instruction
  "mask" for each token, allowing for the perfect reconstruction of any
  capitalization pattern:
    - '0': Token is lowercase.
    - 'T': Token is Title Cased.
    - 'U': Token is UPPERCASED.
    - 'I,...': A string of indices for any other MixedCase variation.

- **MongoDB Backend:** Leverages an async MongoDB driver for scalable and
  persistent storage of both the vocabulary and the encoded document data.

- **Modern & Robust:** Built with Python 3.10+, featuring async operations,
  type hints, and race-condition-safe vocabulary generation using locks.

Database Schema:
----------------
1.  **Documents Collection (e.g., "documents"):**
    - `_id`: Native MongoDB ObjectId for uniqueness.
    - `link_key`: A user-provided logical ID for the document (e.g., a filename).
    - `encoded_indices`: A comma-separated string of integer indices mapping to
      the vocabulary.
    - `capitalization_mask`: A semicolon-separated string of capitalization
      instructions, parallel to `encoded_indices`.

2.  **Vocabulary Collection (e.g., "onehot_vocabulary"):**
    - A meta-document `{"_id": "_meta"}` tracks the next available index.
    - Each token is a document where `_id` is the token string (e.g., "fox")
      and `idx` is its unique, 1-based integer index.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Ensure correct imports for the library ---
try:
    from zmongo_toolbag.zmongo import ZMongo
except (ImportError, ModuleNotFoundError):
    # Adjust path for local testing if needed
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from zmongo_toolbag.zmongo import ZMongo


# ---------- env & logging ----------
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --- Constants and Configuration ---

LINK_KEY_FIELD = "link_key"
ENCODED_INDICES_FIELD = "encoded_indices"
CAPITALIZATION_MASK_FIELD = "capitalization_mask"
DEFAULT_DOCUMENTS_COLLECTION = "documents"
DEFAULT_VOCAB_COLLECTION = "onehot_vocabulary"

# Regex to capture: alphanumeric words, sequences of whitespace, or single non-word/space characters.
DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9']+|\s+|[^\sA-Za-z0-9']"
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "at",
    "with", "by", "from", "as", "is", "are", "was", "were",
}


def _clean_word(s: str) -> str:
    """
    Pass-through normalization for tokens. The tokenizer handles separation,
    so we store the raw token to ensure perfect reconstruction.
    """
    return s or ""


@dataclass
class VocabConfig:
    """
    Configuration for tokenization and encoding logic.

    Attributes:
        lowercase: If True, all tokens are converted to lowercase for vocabulary
                   storage. Essential for capitalization encoding.
        token_pattern: The regular expression used to split text into tokens.
        remove_stopwords: If True, removes common English stopwords. For perfect
                          reconstruction, this should be False.
        encode_capitalization: If True, generates and stores the capitalization
                               mask for perfect case reconstruction.
    """
    lowercase: bool = True
    token_pattern: str = DEFAULT_TOKEN_PATTERN
    remove_stopwords: bool = True
    encode_capitalization: bool = True


class _Lexicon:
    """
    Internal class to manage the vocabulary in the database.

    It assigns a unique, persistent, 1-based integer index to each token and
    is responsible for all vocabulary lookups and creations. It uses locks
    to ensure atomic, race-condition-safe operations under high concurrency.
    """

    def __init__(self, repo: ZMongo, collection_name: str = DEFAULT_VOCAB_COLLECTION):
        self.repo = repo
        self.collection_name = collection_name
        self._index_lock = asyncio.Lock()
        self._token_creation_lock = asyncio.Lock()

    async def _ensure_meta(self) -> None:
        """Ensures the vocabulary's metadata document for tracking the next index exists."""
        res = await self.repo.find_document(self.collection_name, {"_id": "_meta"})
        if not res.success:
            raise RuntimeError(f"Database error ensuring meta: {res.error}")
        if res.data:
            return
        # Initialize the meta document if it doesn't exist.
        init_res = await self.repo.update_document(
            self.collection_name,
            {"_id": "_meta"},
            {"$set": {"next_index": 1, "created_at": dt.datetime.now().isoformat()}},
            upsert=True,
        )
        if not init_res.success:
            raise RuntimeError(f"Failed to initialize meta document: {init_res.error}")

    async def _get_next_index(self) -> int:
        """Atomically increments and returns the next available vocabulary index."""
        async with self._index_lock:
            await self._ensure_meta()
            # Read-then-increment is safe within the lock.
            find_res = await self.repo.find_document(self.collection_name, {"_id": "_meta"})
            if not find_res.success or not find_res.data:
                raise RuntimeError(find_res.error or "Failed to read _meta document for index.")

            current_index = int(find_res.data["next_index"])

            update_res = await self.repo.update_document(
                self.collection_name, {"_id": "_meta"}, {"$inc": {"next_index": 1}},
            )
            if not update_res.success:
                raise RuntimeError(update_res.error or "Failed to increment _meta index.")

            return current_index

    async def get_or_create_token_index(self, token: str) -> int:
        """
        Returns the 1-based index for a token, creating it if it doesn't exist.
        Uses a double-checked locking pattern to prevent race conditions.
        """
        token = _clean_word(token)
        if not token:
            return 0

        # Fast path: Check for existence without a lock for performance.
        res = await self.repo.find_document(self.collection_name, {"_id": token})
        if res.success and res.data:
            return int(res.data["idx"])

        # Slow path: Acquire a lock to ensure only one task creates the new token.
        async with self._token_creation_lock:
            # Double-check inside the lock in case another task created it while we waited.
            res_after_lock = await self.repo.find_document(self.collection_name, {"_id": token})
            if res_after_lock.success and res_after_lock.data:
                return int(res_after_lock.data["idx"])

            # If it still doesn't exist, we are responsible for creating it.
            new_index = await self._get_next_index()
            ins_res = await self.repo.update_document(
                self.collection_name,
                {"_id": token},
                {"$set": {"idx": new_index, "created_at": dt.datetime.now().isoformat()}},
                upsert=True,
            )
            if not ins_res.success:
                raise RuntimeError(f"Failed to insert new token '{token}': {ins_res.error}")
            return new_index

    async def get_word_from_index(self, index: int) -> Optional[str]:
        """Returns the word corresponding to a 1-based index."""
        res = await self.repo.find_document(self.collection_name, {"idx": int(index)})
        if not res.success:
            raise RuntimeError(res.error)
        return res.data["_id"] if res.data else None

    async def get_all_words(self) -> List[str]:
        """Returns all words in the vocabulary, ordered by their index."""
        res = await self.repo.find_documents(self.collection_name, {"_id": {"$ne": "_meta"}}, sort=[("idx", 1)],
                                             limit=1_000_000)
        if not res.success:
            raise RuntimeError(res.error)
        return [d["_id"] for d in (res.data or [])]

    async def get_size(self) -> int:
        """Returns the total number of words in the vocabulary."""
        res = await self.repo.count_documents(self.collection_name, {"_id": {"$ne": "_meta"}})
        if not res.success:
            raise RuntimeError(res.error)
        return int(res.data["count"])


class ZOneHotDB:
    """
    A database interface for lossless text-to-index conversion.

    This class provides the main API for tokenizing text, storing the encoded
    representation in MongoDB, and reconstructing the original text from the
    stored data.
    """

    def __init__(
            self,
            documents_collection_name: str = DEFAULT_DOCUMENTS_COLLECTION,
            vocab_collection_name: str = DEFAULT_VOCAB_COLLECTION,
            *,
            repo: Optional[ZMongo] = None,
            config: Optional[VocabConfig] = None,
            stopwords: Optional[Sequence[str]] = None,
    ):
        """
        Initializes the ZOneHotDB interface.

        Args:
            documents_collection_name: Name of the MongoDB collection for documents.
            vocab_collection_name: Name of the MongoDB collection for the vocabulary.
            repo: An optional, pre-configured ZMongo instance.
            config: An optional VocabConfig object to customize tokenization.
            stopwords: An optional sequence of stopwords to override the default.
        """
        self.repo = repo or ZMongo()
        self.documents_collection = documents_collection_name
        self.config = config or VocabConfig()
        self.stopwords = set(stopwords if stopwords is not None else DEFAULT_STOPWORDS)
        self.lexicon = _Lexicon(self.repo, vocab_collection_name)

        # Enforce lowercase vocabulary if capitalization encoding is enabled.
        if self.config.encode_capitalization:
            self.config.lowercase = True

    # --- Core Document Operations ---

    async def document_exists(self, document_id: str) -> bool:
        """Checks if a document with the given logical `document_id` exists."""
        res = await self.repo.find_document(self.documents_collection, {LINK_KEY_FIELD: document_id})
        if not res.success:
            raise RuntimeError(res.error)
        return res.data is not None

    async def set_document_field(self, document_id: str, field_name: str, field_value: Any) -> None:
        """
        Sets or updates a specific field for a given document, creating the
        document if it does not already exist.
        """
        # First, ensure the document exists with a `link_key`.
        ensure_res = await self.repo.update_document(
            self.documents_collection,
            {LINK_KEY_FIELD: document_id},
            {"$setOnInsert": {LINK_KEY_FIELD: document_id, "created_at": dt.datetime.now().isoformat()}},
            upsert=True,
        )
        if not ensure_res.success:
            raise RuntimeError(f"Failed to ensure document '{document_id}': {ensure_res.error}")

        # Now, set the desired field.
        update_res = await self.repo.update_document(
            self.documents_collection,
            {LINK_KEY_FIELD: document_id},
            {"$set": {field_name: field_value, "updated_at": dt.datetime.now().isoformat()}},
        )
        if not update_res.success:
            raise RuntimeError(f"Failed to set field '{field_name}' on '{document_id}': {update_res.error}")

    async def get_document_field(self, document_id: str, field_name: str) -> Any:
        """Retrieves the value of a single field from a document."""
        res = await self.repo.find_document(self.documents_collection, {LINK_KEY_FIELD: document_id})
        if not res.success:
            raise RuntimeError(res.error)
        return res.data.get(field_name) if res.data else None

    # --- Encoding and Decoding ---

    def _tokenize(self, text: str) -> List[str]:
        """Internal method to split text into tokens based on the configured regex."""
        patt = re.compile(self.config.token_pattern)
        return patt.findall(text)

    async def encode_and_store_text(self, document_id: str, text_content: str) -> None:
        """
        Tokenizes text, maps tokens to vocabulary indices, and stores the
        result, including the advanced capitalization mask.
        """
        raw_tokens = self._tokenize(text_content or "")

        final_indices = []
        capitalization_mask = []

        for token in raw_tokens:
            # Vocabulary always stores the lowercase version for efficiency.
            processed_token = token.lower() if self.config.lowercase else token

            if self.config.remove_stopwords and processed_token in self.stopwords:
                continue

            index = await self.lexicon.get_or_create_token_index(processed_token)
            if not index: continue

            final_indices.append(index)

            # Generate the capitalization mask if enabled.
            if self.config.encode_capitalization:
                is_word = any(c.isalpha() for c in token)
                mask_value = '0'  # Default for non-words or already lowercase words
                if is_word:
                    if token.islower():
                        mask_value = '0'
                    elif token.isupper():
                        mask_value = 'U'
                    elif token.istitle():
                        mask_value = 'T'
                    else:  # Mixed case requires storing indices.
                        upper_indices = [str(i) for i, char in enumerate(token) if char.isupper()]
                        if upper_indices:
                            mask_value = "I," + ",".join(upper_indices)
                capitalization_mask.append(mask_value)

        # Store both the indices and the mask in the document.
        await self.set_document_field(document_id, ENCODED_INDICES_FIELD, ",".join(map(str, final_indices)))

        if self.config.encode_capitalization:
            await self.set_document_field(document_id, CAPITALIZATION_MASK_FIELD, ";".join(capitalization_mask))

    async def get_encoded_indices(self, document_id: str) -> List[int]:
        """Retrieves the stored list of 1-based vocabulary indices for a document."""
        csv_indices = await self.get_document_field(document_id, ENCODED_INDICES_FIELD)
        if not isinstance(csv_indices, str) or not csv_indices:
            return []
        return [int(p) for p in csv_indices.split(",") if p.isdigit()]

    async def get_capitalization_mask(self, document_id: str) -> List[str]:
        """Retrieves the stored capitalization mask as a list of strings."""
        mask_str = await self.get_document_field(document_id, CAPITALIZATION_MASK_FIELD)
        if not isinstance(mask_str, str) or not mask_str:
            return []
        return mask_str.split(";")

    # --- Utility and Analysis ---

    async def get_onehot_dataframe(
            self,
            document_id: str,
            use_word_columns: bool = True,
            count_term_frequency: bool = False
    ) -> pd.DataFrame:
        """
        Constructs a one-hot encoded DataFrame for the specified document.
        This is useful for data analysis and machine learning tasks.

        Args:
            document_id: The identifier of the document to process.
            use_word_columns: If True, DataFrame columns are the actual words from
                              the vocabulary instead of integer indices.
            count_term_frequency: If True, cell values will be the term count
                                  (frequency) instead of a binary (0/1) flag.

        Returns:
            A pandas DataFrame representing the document's one-hot encoding.
        """
        indices = await self.get_encoded_indices(document_id)
        vocab_size = await self.lexicon.get_size()

        if not indices:
            return pd.DataFrame(np.zeros((1, max(1, vocab_size)), dtype=int))

        vec = np.zeros((1, max(1, vocab_size)), dtype=int)

        for idx in indices:
            if 0 < idx <= vocab_size:
                col_index = idx - 1  # Convert 1-based vocab to 0-based df index
                if count_term_frequency:
                    vec[0, col_index] += 1
                else:
                    vec[0, col_index] = 1

        df = pd.DataFrame(vec)
        if use_word_columns:
            words = await self.lexicon.get_all_words()
            if len(words) == df.shape[1]:
                df.columns = words
        return df

    async def clear_documents_collection(self) -> None:
        """Utility method to clear all documents from the collection."""
        res = await self.repo.delete_documents(self.documents_collection, {})
        if not res.success:
            raise RuntimeError(res.error)


# --- Demo Usage ---
async def _demo() -> None:
    """Demonstrates the full encode-decode-verify cycle using a local text file."""
    logging.basicConfig(level=logging.INFO)
    print("--- Running ZOneHotDB Demo ---")
    async with ZMongo() as repo:
        # Configure for perfect reconstruction (no stopwords, capitalization on).
        demo_config = VocabConfig(encode_capitalization=True, remove_stopwords=False)
        db = ZOneHotDB(documents_collection_name="demo_docs", repo=repo, config=demo_config)

        # Clean up any previous demo runs.
        await db.clear_documents_collection()
        await repo.delete_documents(db.lexicon.collection_name, {})

        doc_id = "local_file_demo"
        # Using a simple text file for this demo
        file_path = Path.home() / ".resources" / "static" / "demo_text.txt"
        print(f"\nAttempting to load text from: {file_path}")

        try:
            # Create a demo file if it doesn't exist
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                demo_content = """Welcome to ZOneHotDB, a test of mixedCase reconstruction!
This line is ALL CAPS.
    And this one is indented."""
                file_path.write_text(demo_content, encoding='utf-8')
                print("Created a demo text file.")

            original_text = file_path.read_text(encoding='utf-8')
            print(f"Successfully loaded {len(original_text)} characters from the file.")
        except Exception as e:
            print(f"An error occurred with the file: {e}")
            return

        print(f"\n1. Encoding and storing text for document: '{doc_id}'")
        await db.encode_and_store_text(doc_id, original_text)

        indices = await db.get_encoded_indices(doc_id)
        mask = await db.get_capitalization_mask(doc_id)
        print(f" -> Stored {len(indices)} indices and {len(mask)} mask entries.")

        print("\n2. Reconstructing the document from indices...")
        word_tasks = [db.lexicon.get_word_from_index(idx) for idx in indices]
        words = await asyncio.gather(*word_tasks)

        reconstructed_tokens = []
        for i, word in enumerate(words):
            if word is None: continue

            mask_value = mask[i] if i < len(mask) else '0'
            reconstructed_token = word

            if mask_value == 'T':
                reconstructed_token = word.capitalize()
            elif mask_value == 'U':
                reconstructed_token = word.upper()
            elif mask_value.startswith('I,'):
                try:
                    indices_str = mask_value[2:]
                    upper_indices = {int(idx) for idx in indices_str.split(',')}
                    char_list = list(word)
                    for idx in upper_indices:
                        if idx < len(char_list):
                            char_list[idx] = char_list[idx].upper()
                    reconstructed_token = "".join(char_list)
                except (ValueError, IndexError):
                    reconstructed_token = word  # Fallback on error

            reconstructed_tokens.append(reconstructed_token)

        reconstructed_text = "".join(reconstructed_tokens)

        print("\n--- Verification ---")
        if original_text == reconstructed_text:
            print("✅ SUCCESS: Reconstructed text perfectly matches the original file content.")
        else:
            print("❌ FAILURE: Reconstructed text does NOT match the original.")
            print(f"Original length: {len(original_text)}, Reconstructed length: {len(reconstructed_text)}")
            # Find and display the first point of failure for easier debugging.
            for i, (orig_char, recon_char) in enumerate(zip(original_text, reconstructed_text)):
                if orig_char != recon_char:
                    print(f"Mismatch found at character {i}:")
                    context = 20
                    start, end = max(0, i - context), i + context
                    print(f"  Original:     ...{repr(original_text[start:end])}...")
                    print(f"  Reconstructed:  ...{repr(reconstructed_text[start:end])}...")
                    print(
                        f"  Character values: Original '{repr(orig_char)}' ({ord(orig_char)}) vs Reconstructed '{repr(recon_char)}' ({ord(recon_char)})")
                    break

        print("\n--- Demo Complete ---")


if __name__ == "__main__":
    asyncio.run(_demo())

