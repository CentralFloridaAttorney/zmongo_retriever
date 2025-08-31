# zonehotdb.py
# Python 3.10+
#
# =============================================================================
# ZOneHotDB: A Lossless Formatted Text-to-Index Conversion Library
# =============================================================================
"""
ZOneHotDB provides a robust system for converting formatted text documents
(like HTML) into a compact, indexed format that allows for visually perfect
reconstruction of the original content.

This library is designed for scenarios where preserving rich text formatting—
including whitespace, punctuation, capitalization, bold, italics, and
underline—is critical. It uses BeautifulSoup to parse document structure and
a MongoDB backend to store the vocabulary and encoded data.

Core Features:
-------------
- **Visually Lossless Reconstruction:** The primary design goal. Any formatted
  text encoded can be decoded back to a version that renders identically.

- **Intelligent HTML Parsing:** Uses BeautifulSoup to understand the semantic
  structure of a document, identifying which formatting tags apply to which
  pieces of text.

- **Efficient Vocabulary:** Stores only the lowercase version of each token in the
  database to prevent vocabulary bloat from case variations.

- **Advanced Formatting Mask:** A sophisticated system stores a "mask" of
  instructions for each token, allowing for the perfect reconstruction of any
  formatting pattern:
    - **Capitalization:** '0', 'T' (Title), 'U' (UPPER), 'I,...' (Indices).
    - **Styling:** 'B' (Bold), 'I' (Italic), 'U' (Underline).
    - A bold, title-cased token would have a mask like "T,B".

- **MongoDB Backend:** Leverages an async MongoDB driver for scalable and
  persistent storage.

Dependencies:
- beautifulsoup4: For parsing HTML structure.
  Install with: pip install beautifulsoup4
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set

import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from dotenv import load_dotenv

# --- Ensure correct imports for the library ---
try:
    from .zmongo import ZMongo
except (ImportError, ModuleNotFoundError):
    from zmongo_toolbag.zmongo import ZMongo  # type: ignore

# ---------- env & logging ----------
load_dotenv(Path.home() / ".resources" / ".env_zai_core")
load_dotenv(Path.home() / ".resources" / ".secrets")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# --- Constants and Configuration ---

LINK_KEY_FIELD = "link_key"
ENCODED_INDICES_FIELD = "encoded_indices"
FORMATTING_MASK_FIELD = "formatting_mask"
DEFAULT_DOCUMENTS_COLLECTION = "documents"
DEFAULT_VOCAB_COLLECTION = "onehot_vocabulary"

# Regex to capture: alphanumeric words, sequences of whitespace, or single non-word/space characters.
DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9']+|\s+|[^\sA-Za-z0-9']"
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "at",
    "with", "by", "from", "as", "is", "are", "was", "were",
}


@dataclass
class VocabConfig:
    """Configuration for tokenization and encoding logic."""
    token_pattern: str = DEFAULT_TOKEN_PATTERN
    remove_stopwords: bool = True


class _Lexicon:
    """Internal class to manage the vocabulary in the database."""

    def __init__(self, repo: ZMongo, collection_name: str = DEFAULT_VOCAB_COLLECTION):
        self.repo = repo
        self.collection_name = collection_name
        self._index_lock = asyncio.Lock()
        self._token_creation_lock = asyncio.Lock()

    async def _ensure_meta(self) -> None:
        res = await self.repo.find_document(self.collection_name, {"_id": "_meta"})
        if not res.success: raise RuntimeError(f"DB error: {res.error}")
        if res.data: return
        init_res = await self.repo.update_document(
            self.collection_name, {"_id": "_meta"},
            {"$set": {"next_index": 1}}, upsert=True
        )
        if not init_res.success: raise RuntimeError(f"Failed to init meta: {init_res.error}")

    async def _get_next_index(self) -> int:
        async with self._index_lock:
            await self._ensure_meta()
            find_res = await self.repo.find_document(self.collection_name, {"_id": "_meta"})
            if not find_res.success or not find_res.data:
                raise RuntimeError(find_res.error or "Failed to read meta.")
            current_index = int(find_res.data["next_index"])
            update_res = await self.repo.update_document(
                self.collection_name, {"_id": "_meta"}, {"$inc": {"next_index": 1}}
            )
            if not update_res.success:
                raise RuntimeError(update_res.error or "Failed to increment meta.")
            return current_index

    async def get_or_create_token_index(self, token: str) -> int:
        if not token: return 0
        res = await self.repo.find_document(self.collection_name, {"_id": token})
        if res.success and res.data:
            return int(res.data["idx"])
        async with self._token_creation_lock:
            res_after_lock = await self.repo.find_document(self.collection_name, {"_id": token})
            if res_after_lock.success and res_after_lock.data:
                return int(res_after_lock.data["idx"])
            new_index = await self._get_next_index()
            ins_res = await self.repo.update_document(
                self.collection_name, {"_id": token},
                {"$set": {"idx": new_index}}, upsert=True
            )
            if not ins_res.success:
                raise RuntimeError(f"Failed to insert new token '{token}': {ins_res.error}")
            return new_index

    async def get_word_from_index(self, index: int) -> Optional[str]:
        res = await self.repo.find_document(self.collection_name, {"idx": int(index)})
        if not res.success: raise RuntimeError(res.error)
        return res.data["_id"] if res.data else None


class ZOneHotDB:
    """A database interface for lossless formatted text-to-index conversion."""

    def __init__(
            self,
            documents_collection_name: str = DEFAULT_DOCUMENTS_COLLECTION,
            vocab_collection_name: str = DEFAULT_VOCAB_COLLECTION,
            *,
            repo: Optional[ZMongo] = None,
            config: Optional[VocabConfig] = None,
            stopwords: Optional[Sequence[str]] = None,
    ):
        self.repo = repo or ZMongo()
        self.documents_collection = documents_collection_name
        self.config = config or VocabConfig()
        self.stopwords = set(stopwords if stopwords is not None else DEFAULT_STOPWORDS)
        self.lexicon = _Lexicon(self.repo, vocab_collection_name)

    async def set_document_field(self, document_id: str, field_name: str, field_value: Any) -> None:
        """Sets or updates a specific field for a given document."""
        ensure_res = await self.repo.update_document(
            self.documents_collection, {LINK_KEY_FIELD: document_id},
            {"$setOnInsert": {LINK_KEY_FIELD: document_id}}, upsert=True
        )
        if not ensure_res.success:
            raise RuntimeError(f"Failed to ensure doc: {ensure_res.error}")
        update_res = await self.repo.update_document(
            self.documents_collection, {LINK_KEY_FIELD: document_id},
            {"$set": {field_name: field_value}}
        )
        if not update_res.success:
            raise RuntimeError(f"Failed to set field: {update_res.error}")

    async def get_document_field(self, document_id: str, field_name: str) -> Any:
        """Retrieves the value of a single field from a document."""
        res = await self.repo.find_document(self.documents_collection, {LINK_KEY_FIELD: document_id})
        if not res.success: raise RuntimeError(res.error)
        return res.data.get(field_name) if res.data else None

    def _get_formatting_from_parents(self, element: NavigableString) -> Set[str]:
        """Walks up the tree from a text node to find formatting tags."""
        formats = set()
        for parent in element.find_parents():
            if parent.name in ['b', 'strong']:
                formats.add('B')
            if parent.name in ['i', 'em']:
                formats.add('I')
            if parent.name == 'u':
                formats.add('U')
        return formats

    def _get_capitalization_mask(self, token: str) -> str:
        """Generates the capitalization portion of the formatting mask."""
        if any(c.isalpha() for c in token):
            if token.isupper(): return 'U'
            if token.istitle(): return 'T'
            is_mixed = not token.islower()
            if is_mixed:
                upper_indices = [str(i) for i, char in enumerate(token) if char.isupper()]
                if upper_indices:
                    return "I," + ",".join(upper_indices)
        return '0'  # Default for lowercase, punctuation, whitespace

    async def encode_and_store_text(self, document_id: str, html_content: str) -> None:
        """
        Parses HTML content, tokenizes it while preserving formatting, and
        stores the result in the database.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        all_indices: List[int] = []
        all_masks: List[str] = []

        # We find all text nodes in the document
        for text_node in soup.find_all(string=True):
            if isinstance(text_node, NavigableString):
                # Get formatting (bold, italic) from parent tags
                formatting = self._get_formatting_from_parents(text_node)

                # Tokenize the text content of this node
                raw_tokens = pd.Series(str(text_node)).str.findall(self.config.token_pattern).iloc[0]

                for token in raw_tokens:
                    # Vocabulary always stores the lowercase version
                    processed_token = token.lower()

                    if self.config.remove_stopwords and processed_token in self.stopwords:
                        continue

                    index = await self.lexicon.get_or_create_token_index(processed_token)
                    if not index: continue

                    all_indices.append(index)

                    # Build the full formatting mask
                    cap_mask = self._get_capitalization_mask(token)

                    # Combine capitalization and style formats
                    full_mask_parts = [cap_mask] + sorted(list(formatting))
                    all_masks.append(",".join(full_mask_parts))

        # Store both the indices and the new formatting masks
        await self.set_document_field(document_id, ENCODED_INDICES_FIELD, ",".join(map(str, all_indices)))
        await self.set_document_field(document_id, FORMATTING_MASK_FIELD, ";".join(all_masks))

    async def get_encoded_indices(self, document_id: str) -> List[int]:
        """Retrieves the stored list of 1-based vocabulary indices."""
        csv_indices = await self.get_document_field(document_id, ENCODED_INDICES_FIELD)
        return [int(p) for p in (csv_indices or "").split(",") if p.isdigit()]

    async def get_formatting_mask(self, document_id: str) -> List[str]:
        """Retrieves the stored formatting mask as a list of strings."""
        mask_str = await self.get_document_field(document_id, FORMATTING_MASK_FIELD)
        return (mask_str or "").split(";")
