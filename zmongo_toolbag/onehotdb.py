# onehotdb.py
# Python 3.10+
# Async/OOP helper for one-hot (and index) encoding that:
#   - Uses ZMongo's async API (find_document / update_document / aggregate)
#   - Always returns SafeResult
#   - Stores a persisted vocabulary and encoded outputs
#
# Storage modes:
#   - "index": per-token integer index (compact, practical)
#   - "bitset": per-token bit-packed one-hot (bytes)
#   - "byte_per_bit": diagnostic worst-case (1 bit as 1 byte)

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from bson import ObjectId

from zmongo_toolbag.data_processing import SafeResult, DataProcessor
from zmongo_toolbag.zmongo import ZMongo

logger = logging.getLogger(__name__)

TokenMode = Literal["word", "char"]
StorageMode = Literal["index", "bitset", "byte_per_bit"]

DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9']+"
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "of", "to", "in", "on", "for", "at",
    "with", "by", "from", "as", "is", "are", "was", "were",
}


@dataclass
class VocabConfig:
    mode: TokenMode = "word"
    lowercase: bool = True
    token_pattern: str = DEFAULT_TOKEN_PATTERN  # word-mode only
    remove_stopwords: bool = True
    min_freq: int = 1
    max_vocab_size: Optional[int] = None
    include_oov_token: bool = True  # add "<UNK>" at index 0


@dataclass
class OneHotSchema:
    vocab_size: int
    storage: StorageMode
    mode: TokenMode
    token_pattern: Optional[str]
    lowercase: bool
    remove_stopwords: bool
    min_freq: int
    max_vocab_size: Optional[int]
    include_oov_token: bool
    vocab_version: int


class OneHotDB:
    """
    End-to-end one-hot/index encoding pipeline with ZMongo persistence.

    Typical flow:
        async with ZMongo() as repo:
            oh = OneHotDB(repo, vocab_collection="onehot_vocab", vocab_key="kb:word")
            await oh.fit_from_collection("kb", text_field="text")
            res = await oh.store_encoded(
                target_collection="kb_encoded",
                doc_id="64f0...24charhex...",
                text="Hello world.",
                storage="index",
                extra_metadata={"src": "demo"},
            )
    """

    def __init__(
        self,
        repo: ZMongo,
        *,
        vocab_collection: str = "onehot_vocab",
        vocab_key: str = "default",
        config: Optional[VocabConfig] = None,
        stopwords: Optional[Sequence[str]] = None,
    ):
        self.repo = repo
        self.vocab_collection = vocab_collection
        self.vocab_key = vocab_key
        self.config = config or VocabConfig()
        self.stopwords = set(stopwords) if stopwords is not None else set(DEFAULT_STOPWORDS)

        self._token2idx: Dict[str, int] = {}
        self._idx2token: List[str] = []
        self._vocab_version: int = 0

    # ---------------------------
    # Vocabulary persistence (ZMongo)
    # ---------------------------
    async def load_vocab(self) -> SafeResult:
        """Load vocabulary doc from Mongo by `_id=vocab_key` via ZMongo.find_document()."""
        try:
            res = await self.repo.find_document(self.vocab_collection, {"_id": self.vocab_key})
            if not res.success:
                return res
            doc = res.data
            if not doc:
                return SafeResult.ok(None)

            # restore state
            self._token2idx = {k: int(v) for k, v in doc.get("token2idx", {}).items()}
            self._idx2token = list(doc.get("idx2token", []))
            self._vocab_version = int(doc.get("version", 0))

            # restore config
            cfg = doc.get("config", {})
            self.config = VocabConfig(
                mode=cfg.get("mode", self.config.mode),
                lowercase=cfg.get("lowercase", self.config.lowercase),
                token_pattern=cfg.get("token_pattern", self.config.token_pattern),
                remove_stopwords=cfg.get("remove_stopwords", self.config.remove_stopwords),
                min_freq=int(cfg.get("min_freq", self.config.min_freq)),
                max_vocab_size=cfg.get("max_vocab_size", self.config.max_vocab_size),
                include_oov_token=cfg.get("include_oov_token", self.config.include_oov_token),
            )
            return SafeResult.ok(doc)
        except Exception as e:
            return SafeResult.fail(f"load_vocab error: {e}", exc=e)

    async def save_vocab(self) -> SafeResult:
        """Persist vocabulary via ZMongo.update_document(..., upsert=True)."""
        try:
            doc = {
                "_id": self.vocab_key,
                "token2idx": self._token2idx,
                "idx2token": self._idx2token,
                "version": self._vocab_version,
                "config": asdict(self.config),
                "created_at": dt.datetime.utcnow().isoformat(),
            }
            return await self.repo.update_document(
                self.vocab_collection,
                {"_id": self.vocab_key},
                {"$set": doc},
                upsert=True,
            )
        except Exception as e:
            return SafeResult.fail(f"save_vocab error: {e}", exc=e)

    async def clear_vocab(self) -> SafeResult:
        """Delete vocabulary via ZMongo.delete_document()."""
        try:
            res = await self.repo.delete_document(self.vocab_collection, {"_id": self.vocab_key})
            if res.success:
                self._token2idx.clear()
                self._idx2token.clear()
                self._vocab_version = 0
            return res
        except Exception as e:
            return SafeResult.fail(f"clear_vocab error: {e}", exc=e)

    # ---------------------------
    # Fit vocabulary
    # ---------------------------
    async def fit_from_texts(self, texts: Iterable[str]) -> SafeResult:
        """Build the vocabulary from an iterable of strings; then save via ZMongo."""
        try:
            counter: Counter = Counter()
            for txt in texts:
                counter.update(self._tokenize(txt or ""))

            tokens = [t for t, c in counter.items() if c >= self.config.min_freq]
            tokens.sort(key=lambda t: (-counter[t], t))
            if self.config.max_vocab_size is not None:
                tokens = tokens[: self.config.max_vocab_size]

            vocab: Dict[str, int] = {}
            idx2token: List[str] = []
            if self.config.include_oov_token:
                vocab["<UNK>"] = 0
                idx2token.append("<UNK>")

            start = 1 if self.config.include_oov_token else 0
            for i, tok in enumerate(tokens, start=start):
                vocab[tok] = i
                idx2token.append(tok)

            self._token2idx = vocab
            self._idx2token = idx2token
            self._vocab_version += 1
            return await self.save_vocab()
        except Exception as e:
            return SafeResult.fail(f"fit_from_texts error: {e}", exc=e)

    async def fit_from_collection(
        self,
        source_collection: str,
        *,
        text_field: str,
        limit: int = 10000,
        batch_size: int = 1000,
    ) -> SafeResult:
        """
        Page through documents using ZMongo.aggregate() ONLY (no direct motor usage),
        counting token frequencies, then persist the vocabulary.
        """
        try:
            processed = 0
            counter: Counter = Counter()
            skip = 0

            while processed < limit:
                pipeline = [
                    {"$project": {"_id": 1, "txt": f"${text_field}"}},
                    {"$skip": skip},
                    {"$limit": min(batch_size, limit - processed)},
                ]
                batch_res = await self.repo.aggregate(source_collection, pipeline, limit=batch_size)
                if not batch_res.success:
                    return batch_res

                docs = batch_res.data or []
                if not docs:
                    break

                for d in docs:
                    txt = d.get("txt")
                    if isinstance(txt, str) and txt:
                        counter.update(self._tokenize(txt))
                        processed += 1
                        if processed >= limit:
                            break

                if len(docs) < min(batch_size, limit - (processed - len(docs))):
                    # likely exhausted
                    break

                skip += len(docs)

            tokens = [t for t, c in counter.items() if c >= self.config.min_freq]
            tokens.sort(key=lambda t: (-counter[t], t))
            if self.config.max_vocab_size is not None:
                tokens = tokens[: self.config.max_vocab_size]

            vocab: Dict[str, int] = {}
            idx2token: List[str] = []
            if self.config.include_oov_token:
                vocab["<UNK>"] = 0
                idx2token.append("<UNK>")

            start = 1 if self.config.include_oov_token else 0
            for i, tok in enumerate(tokens, start=start):
                vocab[tok] = i
                idx2token.append(tok)

            self._token2idx = vocab
            self._idx2token = idx2token
            self._vocab_version += 1

            save_res = await self.save_vocab()
            if not save_res.success:
                return save_res

            return SafeResult.ok(
                {"fitted_docs": processed, "vocab_size": len(self._idx2token), "version": self._vocab_version}
            )
        except Exception as e:
            return SafeResult.fail(f"fit_from_collection error: {e}", exc=e)

    # ---------------------------
    # Encoding
    # ---------------------------
    async def encode_and_store_one(
        self,
        *,
        source_collection: str,
        text_field: str,
        target_collection: str,
        doc_id: str,
        storage: StorageMode = "index",
        extra_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = True,
    ) -> SafeResult:
        """
        Fetch a document by its _id (string is fine), extract `text_field`,
        encode it, and store the result to `target_collection`.

        Returns SafeResult from the underlying ZMongo.update_document call.
        """
        try:
            # 1) Fetch the source record by _id (ZMongo should coerce str->ObjectId if hex)
            src = await self.repo.find_document(source_collection, {"_id": doc_id})
            if not src.success:
                return src
            if not src.data:
                return SafeResult.fail(f"encode_and_store_one: no document found for _id={doc_id}")

            # 2) Extract the starting text (supports dot-paths)
            text_val = DataProcessor.get_value(src.data, text_field)
            if not isinstance(text_val, str) or not text_val:
                return SafeResult.fail(f"encode_and_store_one: '{text_field}' missing or not a non-empty string")

            # 3) Encode & store using the same _id
            meta = {"source_collection": source_collection, "text_field": text_field, "doc_id_source": "by_id"}
            if extra_metadata:
                meta.update(extra_metadata)

            return await self.store_encoded(
                target_collection=target_collection,
                doc_id=doc_id,
                text=text_val,
                storage=storage,
                extra_metadata=meta,
                upsert=upsert,
            )
        except Exception as e:
            return SafeResult.fail(f"encode_and_store_one error: {e}", exc=e)




    async def encode_text(self, text: str, *, storage: StorageMode = "index") -> SafeResult:
        """Encode one text string under selected storage mode; returns SafeResult."""
        try:
            if not self._token2idx:
                loaded = await self.load_vocab()
                if not loaded.success:
                    return loaded

            tokens = self._tokenize(text or "")

            if storage == "index":
                indices = [self._to_index(tok) for tok in tokens]
                return SafeResult.ok(
                    {
                        "indices": indices,
                        "schema": asdict(self._schema(storage)),
                        "token_count": len(indices),
                    }
                )

            if storage in ("bitset", "byte_per_bit"):
                b = [self._one_hot_bytes(self._to_index(tok)) for tok in tokens]
                b64 = [base64.b64encode(x).decode("ascii") for x in b]
                return SafeResult.ok(
                    {
                        "bitpack_b64": b64,
                        "bytes_per_token": len(b[0]) if b else math.ceil(self.vocab_size / 8),
                        "schema": asdict(self._schema(storage)),
                        "token_count": len(b64),
                    }
                )

            return SafeResult.fail(f"Unsupported storage mode: {storage}")
        except Exception as e:
            return SafeResult.fail(f"encode_text error: {e}", exc=e)

    async def store_encoded(
        self,
        *,
        target_collection: str,
        doc_id: ObjectId | str,
        text: str,
        storage: StorageMode = "index",
        extra_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = True,
    ) -> SafeResult:
        """Encode and write to Mongo via ZMongo.update_document(..., upsert=True)."""
        try:
            enc = await self.encode_text(text, storage=storage)
            if not enc.success:
                return enc

            stats = self.estimate_compression_for_text(text, storage=storage)

            # rely on ZMongo's _normalize_ids_in_query() to coerce str -> ObjectId if hex
            q = {"_id": doc_id}
            payload = {
                "encoded": enc.data,
                "stats": stats,
                "meta": (extra_metadata or {}),
                "vocab_key": self.vocab_key,
                "vocab_version": self._vocab_version,
                "created_at": dt.datetime.utcnow().isoformat(),
            }
            return await self.repo.update_document(target_collection, q, {"$set": payload}, upsert=upsert)
        except Exception as e:
            return SafeResult.fail(f"store_encoded error: {e}", exc=e)

    async def encode_collection(
        self,
        source_collection: str,
        *,
        text_field: str,
        target_collection: str,
        storage: StorageMode = "index",
        limit: int = 1000,
        batch_size: int = 200,
        upsert: bool = True,
    ) -> SafeResult:
        """
        Page with ZMongo.aggregate() to fetch _id + text, encode each, and store via ZMongo.update_document().
        """
        try:
            processed = 0
            ok = 0
            errs: List[str] = []
            skip = 0

            while processed < limit:
                pipeline = [
                    {"$project": {"_id": 1, "txt": f"${text_field}"}},
                    {"$skip": skip},
                    {"$limit": min(batch_size, limit - processed)},
                ]
                batch_res = await self.repo.aggregate(source_collection, pipeline, limit=batch_size)
                if not batch_res.success:
                    return batch_res

                docs = batch_res.data or []
                if not docs:
                    break

                for d in docs:
                    _id = d.get("_id")  # ZMongo.stringify returns str(ObjectId); OK for its own coercion later
                    txt = d.get("txt")
                    if not isinstance(txt, str):
                        errs.append(f"{_id}: no text at '{text_field}'")
                        processed += 1
                        if processed >= limit:
                            break
                        continue

                    sres = await self.store_encoded(
                        target_collection=target_collection,
                        doc_id=_id,
                        text=txt,
                        storage=storage,
                        extra_metadata={"source_collection": source_collection, "text_field": text_field},
                        upsert=upsert,
                    )
                    ok += int(bool(sres.success))
                    if not sres.success:
                        errs.append(f"{_id}: {sres.error}")
                    processed += 1
                    if processed >= limit:
                        break

                if len(docs) < min(batch_size, limit - (processed - len(docs))):
                    break
                skip += len(docs)

            return SafeResult.ok({"processed": processed, "stored_ok": ok, "errors": errs})
        except Exception as e:
            return SafeResult.fail(f"encode_collection error: {e}", exc=e)

    # ---------------------------
    # Compression / stats (pure functions)
    # ---------------------------
    def estimate_compression_for_text(self, text: str, *, storage: StorageMode = "index") -> Dict[str, Any]:
        """
        Original bits ≈ len(utf8_bytes) * 8.
        index:     NW * max(1, ceil(log2(V))) bits
        bitset:    NW * V bits
        byte_per_bit: NW * V * 8 bits (diagnostic pessimistic)
        """
        tokens = self._tokenize(text or "")
        NW = len(tokens)
        V = max(1, self.vocab_size)
        L = self._avg_word_length(tokens)

        original_bits = len(text.encode("utf-8")) * 8

        if storage == "index":
            bits_per = max(1, math.ceil(math.log2(V)))
            compressed_bits = NW * bits_per
        elif storage == "bitset":
            compressed_bits = NW * V
        elif storage == "byte_per_bit":
            compressed_bits = NW * V * 8
        else:
            compressed_bits = float("inf")

        ratio = (original_bits / compressed_bits) if compressed_bits > 0 else float("inf")
        return {
            "original_bits": original_bits,
            "compressed_bits": int(compressed_bits) if math.isfinite(compressed_bits) else None,
            "ratio": ratio,
            "V": V,
            "L": L,
            "NW": NW,
            "storage": storage,
        }

    # ---------------------------
    # Internals
    # ---------------------------
    def _schema(self, storage: StorageMode) -> OneHotSchema:
        return OneHotSchema(
            vocab_size=self.vocab_size,
            storage=storage,
            mode=self.config.mode,
            token_pattern=self.config.token_pattern if self.config.mode == "word" else None,
            lowercase=self.config.lowercase,
            remove_stopwords=self.config.remove_stopwords,
            min_freq=self.config.min_freq,
            max_vocab_size=self.config.max_vocab_size,
            include_oov_token=self.config.include_oov_token,
            vocab_version=self._vocab_version,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._idx2token)

    def _tokenize(self, text: str) -> List[str]:
        if self.config.lowercase:
            text = text.lower()

        if self.config.mode == "char":
            # keep spaces; drop control newlines/tabs
            return [ch for ch in text if ch not in {"\n", "\r", "\t"}]

        patt = re.compile(self.config.token_pattern)
        tokens = patt.findall(text)
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def _to_index(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        return self._token2idx.get("<UNK>", 0) if self.config.include_oov_token else -1

    def _one_hot_bytes(self, idx: int) -> bytes:
        V = self.vocab_size
        nbytes = (V + 7) // 8
        buf = bytearray(nbytes)
        if 0 <= idx < V:
            byte_i = idx // 8
            bit_i = idx % 8
            buf[byte_i] |= (1 << bit_i)
        return bytes(buf)

    @staticmethod
    def _avg_word_length(tokens: Sequence[str]) -> float:
        return (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0

    # ---------------------------
    # Demo
    # ---------------------------
    @classmethod
    async def demo(cls) -> None:
        logging.basicConfig(level=logging.INFO)
        async with ZMongo() as repo:
            oh = cls(repo, vocab_key="demo:word", config=VocabConfig(mode="word"))
            await oh.clear_vocab()
            await oh.fit_from_texts(["Hello world", "Hello Orlando", "World of law"])
            res = await oh.encode_text("Hello law world", storage="index")
            logger.info("Encoded demo: %r", res.data)

if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    asyncio.run(OneHotDB.demo())
