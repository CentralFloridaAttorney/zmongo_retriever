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
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
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
CAPITALIZATION_MASK_FIELD = "capitalization_mask"
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
    encode_capitalization: bool = True


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
    """A database interface for lossless text-to-index conversion."""

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

    async def encode_and_store_text(self, document_id: str, text_content: str) -> None:
        """
        Tokenizes text, maps tokens to vocabulary indices, and stores the
        result, including the advanced capitalization mask.
        """
        raw_tokens = pd.Series(str(text_content)).str.findall(self.config.token_pattern).iloc[0]

        final_indices = []
        capitalization_mask = []

        for token in raw_tokens:
            processed_token = token.lower()

            if self.config.remove_stopwords and processed_token in self.stopwords:
                continue

            index = await self.lexicon.get_or_create_token_index(processed_token)
            if not index: continue

            final_indices.append(index)

            if self.config.encode_capitalization:
                mask_value = self._get_capitalization_mask(token)
                capitalization_mask.append(mask_value)

        await self.set_document_field(document_id, ENCODED_INDICES_FIELD, ",".join(map(str, final_indices)))

        if self.config.encode_capitalization:
            await self.set_document_field(document_id, CAPITALIZATION_MASK_FIELD, ";".join(capitalization_mask))

    async def get_encoded_indices(self, document_id: str) -> List[int]:
        """Retrieves the stored list of 1-based vocabulary indices."""
        csv_indices = await self.get_document_field(document_id, ENCODED_INDICES_FIELD)
        return [int(p) for p in (csv_indices or "").split(",") if p.isdigit()]

    async def get_capitalization_mask(self, document_id: str) -> List[str]:
        """Retrieves the stored capitalization mask as a list of strings."""
        mask_str = await self.get_document_field(document_id, CAPITALIZATION_MASK_FIELD)
        return (mask_str or "").split(";")

