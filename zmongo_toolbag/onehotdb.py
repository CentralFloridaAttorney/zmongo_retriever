# onehotdb.py
# Python 3.10+
#
# OneHotDB (compat layer) — old API surface, new ZMongo backend
# -------------------------------------------------------------
# What changed:
#   • Removed all MySQL usage. All I/O now goes through ZMongo (async).
#   • Replaced external OneHotWords/TextProcessor dependencies with
#     an internal, persisted lexicon that assigns 1-based term indices.
#   • Preserved old method names (put_onehot, get_onehot_list, get_onehot, put, get, etc.)
#     while making them async to fit the new system (ZMongo is async).
#   • Results are plain Python objects like before (lists/strings/DataFrame).
#     Failures raise exceptions (same behavior the old code effectively had).
#
# Notes:
#   • Indexing remains **1-based** to match the old one-hot logic.
#   • Link keys are stored as document `_id` in Mongo.
#   • The "sentence" field stores a comma-separated list of indices (string), same as before.
#
# If you prefer SafeResult everywhere, you can wrap call sites or add SafeResult
# facades that return SafeResult.ok/fail; for now, this module stays close to the
# original call/return shapes for maximal drop-in compatibility.

from __future__ import annotations

import asyncio
import datetime as dt
import html
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

# New-system dependencies
try:
    from .zmongo import ZMongo
except Exception:
    from zmongo_toolbag.zmongo import ZMongo  # type: ignore

logger = logging.getLogger(__name__)

# --------------------------
# Defaults & simple cleaning
# --------------------------

LINK_KEY = "link_key"
SENTENCE_KEY = "sentence"  # stores CSV of indices, like the original
DEFAULT_SENTENCE_COLLECTION = "sentences"
DEFAULT_VOCAB_COLLECTION = "onehot_vocab"

DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9']+"
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "of", "to", "in", "on", "for", "at",
    "with", "by", "from", "as", "is", "are", "was", "were",
}


def _clean_word(s: str) -> str:
    """Very light normalization to mirror old behavior."""
    s = (s or "").strip()
    if not s:
        return s
    # Old code used html.escape and TextProcessor.get_clean_word.
    # We'll html.escape, but keep alnum/apostrophe words intact.
    s = html.escape(s)
    return s


@dataclass
class VocabConfig:
    """Simple tokenization config (closest to the old code’s assumptions)."""
    lowercase: bool = True
    token_pattern: str = DEFAULT_TOKEN_PATTERN
    remove_stopwords: bool = True
    include_oov_token: bool = False  # old code had no OOV; keep False to match behavior


class _Lexicon:
    """
    Minimal persisted lexicon that assigns **1-based** integer indices to tokens.

    Collection layout (Mongo):
      - meta doc:   { _id: "_meta", next_index: <int>, created_at, updated_at }
      - token doc:  { _id: <token>, idx: <int>, created_at }
    """

    def __init__(self, repo: ZMongo, collection: str = DEFAULT_VOCAB_COLLECTION):
        self.repo = repo
        self.collection = collection

    async def _ensure_meta(self) -> None:
        res = await self.repo.find_document(self.collection, {"_id": "_meta"})
        if not res.success:
            raise RuntimeError(res.error)
        if res.data:
            return
        init = await self.repo.update_document(
            self.collection,
            {"_id": "_meta"},
            {"$set": {"next_index": 1, "created_at": dt.datetime.now().isoformat()}},
            upsert=True,
        )
        if not init.success:
            raise RuntimeError(init.error)

    async def _next_index(self) -> int:
        await self._ensure_meta()
        # Atomically increment next_index
        upd = await self.repo.update_document(
            self.collection,
            {"_id": "_meta"},
            {"$inc": {"next_index": 1}},
            upsert=True,
        )
        if not upd.success:
            raise RuntimeError(upd.error)
        # Read back the value to compute assigned index
        meta = await self.repo.find_document(self.collection, {"_id": "_meta"})
        if not meta.success or not meta.data:
            raise RuntimeError(meta.error or "failed to read _meta after increment")
        return int(meta.data["next_index"]) - 1

    async def put(self, token: str) -> int:
        """Return index for token, creating if necessary (1-based)."""
        token = _clean_word(token)
        # Fast path: already exists?
        got = await self.repo.find_document(self.collection, {"_id": token})
        if not got.success:
            raise RuntimeError(got.error)
        if got.data:
            return int(got.data["idx"])

        # Create new token
        idx = await self._next_index()
        ins = await self.repo.update_document(
            self.collection,
            {"_id": token},
            {"$set": {"idx": idx, "created_at": dt.datetime.now().isoformat()}},
            upsert=True,
        )
        if not ins.success:
            raise RuntimeError(ins.error)
        return idx

    async def get_word(self, idx: int) -> Optional[str]:
        """Return token string for 1-based index (None if not found)."""
        res = await self.repo.find_documents(self.collection, {"idx": int(idx)}, limit=1)
        if not res.success:
            raise RuntimeError(res.error)
        docs = res.data or []
        return docs[0]["_id"] if docs else None

    async def get_words(self) -> List[str]:
        """Return tokens ordered by 1-based index ascending."""
        res = await self.repo.find_documents(self.collection, {"_id": {"$ne": "_meta"}}, sort=[("idx", 1)], limit=1000000)
        if not res.success:
            raise RuntimeError(res.error)
        return [d["_id"] for d in (res.data or [])]

    async def get_row_count(self) -> int:
        """Return vocabulary size (number of tokens; excludes _meta)."""
        cnt = await self.repo.count_documents(self.collection, {"_id": {"$ne": "_meta"}})
        if not cnt.success:
            raise RuntimeError(cnt.error)
        return int(cnt.data["count"])


class OneHotDB:
    """
    OneHotDB — compatibility layer with the new ZMongo backend.

    The old class stored one-hot **indices** in MySQL as a CSV string in the column
    named 'sentence'. We do the same in Mongo (field 'sentence') under the document
    whose `_id` is the old `link_key`.

    Parameters
    ----------
    _database_name : str | None
        Ignored (kept for compatibility).
    _table_name : str | None
        Mongo collection name to store sentences; default "sentences".
    _config_key : str | None
        Ignored (kept for compatibility).
    repo : ZMongo | None
        New-system repository. If omitted, a new ZMongo() is created.

    Collections
    -----------
    - Sentences: <_table_name or "sentences">, docs look like:
        { _id: <link_key>, sentence: "3,12,44", created_at, ... arbitrary fields }
    - Vocab: "onehot_vocab" (configurable internally), keeps token <-> 1-based index.
    """

    def __init__(
        self,
        _database_name: Optional[str] = None,
        _table_name: Optional[str] = None,
        _config_key: Optional[str] = None,
        *,
        repo: Optional[ZMongo] = None,
        vocab_collection: str = DEFAULT_VOCAB_COLLECTION,
        config: Optional[VocabConfig] = None,
        stopwords: Optional[Sequence[str]] = None,
    ):
        self.repo = repo or ZMongo()
        self.table_name = _table_name or DEFAULT_SENTENCE_COLLECTION
        self.config = config or VocabConfig()
        self.stopwords = set(stopwords) if stopwords is not None else set(DEFAULT_STOPWORDS)
        self.lex = _Lexicon(self.repo, vocab_collection)

    # -------------
    # Old helpers (no-ops / compat)
    # -------------
    async def open_database(self, _database_name: str, _table_name: Optional[str] = None) -> None:
        """Kept for compatibility (no-op for Mongo)."""
        if _table_name:
            self.table_name = _table_name

    async def open_table(self, _table_name: str) -> None:
        """Kept for compatibility (no-op for Mongo)."""
        self.table_name = _table_name

    async def delete_database(self, _database_name: str) -> None:
        """Compatibility shim: clears the sentence collection."""
        await self.repo.delete_documents(self.table_name, {})

    async def delete_table(self, _table_name: str) -> None:
        """Compatibility shim: clears the given collection."""
        await self.repo.delete_documents(_table_name, {})

    # -------------
    # Core compat API
    # -------------
    @staticmethod
    def get_clean_key_string(_string: Any) -> str:
        """
        Old escaping helper. We keep it very close to the original, using html.escape.
        """
        _string = str(_string)
        if _string.isdigit():
            _string = "_" + _string
        return html.escape(_string)

    async def put(self, _link_key: str, _key_value: Optional[str] = None, _value: Optional[str] = None) -> int:
        """
        Compatibility behaviors:
          1) put(link_key) -> ensure doc exists, return 1 if exists else 0
          2) put(from_link_key, to_link_key) -> shallow copy (copy values where not None)
          3) put(link_key, key, value) -> set field on the doc (upsert)
        """
        if _key_value is None:
            # case 1: ensure doc exists
            await self.repo.update_document(
                self.table_name,
                {"_id": _link_key},
                {"$setOnInsert": {"created_at": dt.datetime.now().isoformat()}},
                upsert=True,
            )
            return await self.get_id(_link_key)

        if _value is None:
            # case 2: shallow copy row from _link_key to _key_value
            src = await self.repo.find_document(self.table_name, {"_id": _link_key})
            if not src.success:
                raise RuntimeError(src.error)
            if not src.data:
                await self.put(_link_key)  # create empty
                src = await self.repo.find_document(self.table_name, {"_id": _link_key})
            doc = src.data or {}
            dst = await self.repo.find_document(self.table_name, {"_id": _key_value})
            if not dst.success:
                raise RuntimeError(dst.error)
            if not dst.data:
                await self.put(_key_value)

            # Copy all fields except _id; skip None (shallow)
            updates = {k: v for k, v in doc.items() if k != "_id" and v is not None}
            await self.repo.update_document(self.table_name, {"_id": _key_value}, {"$set": updates})
            return await self.get_id(_key_value)

        # case 3: set field on row
        await self.put(_link_key)  # ensure row
        await self.update_value(_link_key, _key_value, _value)
        return await self.get_id(_link_key)

    async def update_value(self, _link_key: str, _key: str, _value: Any) -> int:
        """Set a field value on the sentence doc."""
        upd = await self.repo.update_document(
            self.table_name,
            {"_id": _link_key},
            {"$set": {self.get_clean_key_string(_key): _clean_word(str(_value))}},
        )
        if not upd.success:
            raise RuntimeError(upd.error)
        return await self.get_id(_link_key)

    async def get(self, _link_key: str, _key: Optional[str] = None) -> Any:
        """
        Old behavior:
          - get(link_key) returns the full 'row' (dict now)
          - get(link_key, key) returns the value at that key (or None)
        """
        res = await self.repo.find_document(self.table_name, {"_id": _link_key})
        if not res.success:
            raise RuntimeError(res.error)
        doc = res.data
        if _key is None:
            return self.remove_none(doc)
        return self.remove_none(doc.get(self.get_clean_key_string(_key))) if doc else None

    async def get_values(self, _link_key: str, _exclude_2: bool = False) -> List[Any]:
        """
        Legacy helper: return list of values (optionally skipping first two keys).
        We adapt this to dictionaries: return values in stable key order.
        """
        res = await self.repo.find_document(self.table_name, {"_id": _link_key})
        if not res.success:
            raise RuntimeError(res.error)
        doc = res.data or {}
        keys = list(doc.keys())
        if _exclude_2:
            keys = [k for k in keys if k not in ("_id", LINK_KEY)]
        return [doc[k] for k in keys]

    async def get_id(self, _link_key: str, _value: Optional[str] = None) -> int:
        """
        In the old code this returned a MySQL row id. We return:
          • 1 if the document exists
          • 0 if not
        """
        res = await self.repo.find_document(self.table_name, {"_id": _link_key})
        if not res.success:
            raise RuntimeError(res.error)
        return 1 if res.data else 0

    async def get_row_count(self) -> int:
        """Return number of documents in the sentence collection."""
        c = await self.repo.count_documents(self.table_name, {})
        if not c.success:
            raise RuntimeError(c.error)
        return int(c.data["count"])

    async def get_columns(self, _exclude_2_keys: bool = False) -> List[str]:
        """
        Approximate column discovery: scan a small sample and union keys.
        Mongo is schema-less, so this best-effort approach mirrors the old code’s intent.
        """
        res = await self.repo.find_documents(self.table_name, {}, limit=1000)
        if not res.success:
            raise RuntimeError(res.error)
        cols: set[str] = set()
        for d in (res.data or []):
            cols.update(d.keys())
        cols = list(sorted(cols))
        if _exclude_2_keys:
            cols = [c for c in cols if c not in ("_id", LINK_KEY)]
        return cols

    async def get_dataframe(self) -> pd.DataFrame:
        """Return all documents as a pandas DataFrame (like before)."""
        res = await self.repo.find_documents(self.table_name, {}, limit=1_000_000)
        if not res.success:
            raise RuntimeError(res.error)
        rows = res.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df

    async def get_sentence_indices(self) -> List[str]:
        """Return list of link_keys (document `_id`s)."""
        res = await self.repo.find_documents(self.table_name, {}, projection=["_id"], limit=1_000_000)
        if not res.success:
            raise RuntimeError(res.error)
        return [d["_id"] for d in (res.data or [])]

    # ----------------
    # One-hot routines
    # ----------------
    async def put_onehot(self, _link_key: str, _string: str) -> int:
        """
        Tokenize `_string`, map tokens to **1-based** indices, store as CSV under 'sentence'.
        Returns 1 if row exists after write (compat with old get_id semantics).
        """
        tokens = self._tokenize(_string or "")
        indices: List[str] = []
        for tok in tokens:
            if tok:
                idx = await self.lex.put(tok)
                indices.append(str(idx))

        csv = ",".join(indices)
        await self.put(_link_key)  # ensure row
        await self.repo.update_document(
            self.table_name,
            {"_id": _link_key},
            {"$set": {SENTENCE_KEY: csv, "updated_at": dt.datetime.now().isoformat()}},
        )
        return await self.get_id(_link_key)

    async def get_onehot_list(self, _link_key: str) -> List[str]:
        """
        Return the stored list of indices (as strings) for `link_key`, like before.
        """
        val = await self.get(_link_key, SENTENCE_KEY)
        if not isinstance(val, str) or not val:
            return []
        return [p for p in val.split(",") if p]

    async def get_onehot(self, _link_key: str, _use_column_names: bool = True, _count_uses: bool = False) -> pd.DataFrame:
        """
        Build a one-hot DataFrame (1 row) for the stored sentence at `link_key`.

        Notes:
          • Indices are **1-based**; DataFrame columns are 0-based, so we subtract 1.
          • If `_use_column_names` is True, columns are the token strings ordered by idx.
          • If `_count_uses` is True, counts occurrences rather than binary presence.
        """
        idx_list = await self.get_onehot_list(_link_key)
        if not idx_list:
            return pd.DataFrame(np.zeros((1, max(1, await self.lex.get_row_count())) , dtype=int))

        V = await self.lex.get_row_count()
        vec = np.zeros((1, max(1, V)), dtype=int)

        for s in idx_list:
            try:
                idx = int(s)  # 1-based
            except ValueError:
                idx = 0
            if idx > 0 and idx <= V:
                col = idx - 1  # to 0-based
                if _count_uses:
                    vec[0, col] += 1
                else:
                    vec[0, col] = 1

        df = pd.DataFrame(vec)
        if _use_column_names:
            words = await self.lex.get_words()
            if len(words) == df.shape[1]:
                df.columns = words
        return df

    async def get_onehot_matrix(self, _link_key: str) -> int:
        """
        Compatibility stub: return number of words (vocabulary size),
        like the old method’s return behavior.
        """
        return await self.lex.get_row_count()

    # -------------
    # Misc helpers
    # -------------
    @staticmethod
    def remove_none(_result: Any) -> Any:
        """Replace None(s) with 'None' to mirror old convenience behavior."""
        if _result is None:
            return "None"
        if isinstance(_result, list):
            return ["None" if v is None else v for v in _result]
        return _result

    async def pickle_words(self, _filename_key: Optional[str] = None) -> pd.DataFrame:
        """
        Export the lexicon words to a pickle file (path compatible with old code).
        """
        words = await self.lex.get_words()
        df = pd.DataFrame(words).T
        path = f"../../data/words/{(_filename_key or 'default')}.onehotwords.pkl"
        os.makedirs("../../data/words/", exist_ok=True)
        df.to_pickle(path)
        logger.info("pickle_words: %s", path)
        return df

    # -------------
    # Tokenization
    # -------------
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize the input (close to old behavior)."""
        if self.config.lowercase:
            text = text.lower()
        patt = re.compile(self.config.token_pattern)
        toks = patt.findall(text)
        if self.config.remove_stopwords:
            toks = [t for t in toks if t not in self.stopwords]
        return toks


# -------------------------
# Tiny async demo (optional)
# -------------------------
async def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    async with ZMongo() as repo:
        oh = OneHotDB(_table_name="sentences", repo=repo)
        await oh.delete_table("sentences")  # clear

        await oh.put_onehot("first_key", "That's not it?")
        lst = await oh.get_onehot_list("first_key")
        print("indices:", lst)

        df = await oh.get_onehot("first_key", _use_column_names=True)
        print("one-hot df shape:", df.shape)

        count = await oh.get_row_count()
        print("rows:", count)

        cols = await oh.get_columns()
        print("columns:", cols)

        await oh.pickle_words("word_key")


if __name__ == "__main__":
    asyncio.run(_demo())
