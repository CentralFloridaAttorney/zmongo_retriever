# onehotdb_demo.py
# Python 3.10+
# Demo runner for OneHotDB:
# - fits/loads vocab from a source collection & text field
# - stores *indices* in one target collection
# - stores *one-hot (bitset)* in a separate target collection

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from zmongo_toolbag.data_processing import DataProcessor
from zmongo_toolbag.zmongo import ZMongo
from zmongo_toolbag.zonehotdb import ZOneHotDB, VocabConfig


async def run_demo(
    source_collection: str,
    text_field: str,
    *,
    vocab_collection: str,
    vocab_key: str,
    index_collection: str,
    onehot_collection: str,
    mode: str = "word",                # "word" or "char"
    lowercase: bool = True,
    remove_stopwords: bool = True,
    token_pattern: str = r"[A-Za-z0-9']+",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    include_oov_token: bool = True,
    limit: int = 10000,
    batch_size: int = 1000,
    force_refit: bool = False,
) -> int:
    """
    Returns
    -------
    int : process exit code (0 = success, 1 = failure)
    """
    # Build vocab config
    cfg = VocabConfig(
        mode="char" if mode == "char" else "word",
        lowercase=lowercase,
        token_pattern=token_pattern,
        remove_stopwords=remove_stopwords,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        include_oov_token=include_oov_token,
    )

    async with ZMongo() as repo:
        oh = ZOneHotDB(
            repo,
            vocab_collection=vocab_collection,
            vocab_key=vocab_key,
            config=cfg,
        )

        # Load existing vocab if present (unless forcing a refit)
        if not force_refit:
            load_res = await oh.load_vocab()
            if not load_res.success:
                logging.warning("load_vocab failed (%s); will attempt to fit a new vocab.", load_res.error)

        # Fit a fresh vocab if none present or --force-refit is set
        need_fit = force_refit
        if not need_fit:
            check = await oh.load_vocab()
            # If .data is None, no vocab stored yet
            need_fit = (not check.success) or (check.data is None)

        if need_fit:
            logging.info("Fitting vocabulary from %s.%s (limit=%d)...", source_collection, text_field, limit)
            fit_res = await oh.fit_from_collection(
                source_collection,
                text_field=text_field,
                limit=limit,
                batch_size=batch_size,
            )
            if not fit_res.success:
                logging.error("fit_from_collection failed: %s", fit_res.error)
                return 1
            logging.info("Vocab ready: %s", fit_res.data)

        # Encode to INDEX storage (compact): save in index_collection
        logging.info("Encoding (INDEX) -> %s", index_collection)
        idx_res = await oh.encode_collection(
            source_collection,
            text_field=text_field,
            target_collection=index_collection,
            storage="index",
            limit=limit,
            batch_size=max(200, batch_size // 5),  # smaller write batches
            upsert=True,
        )
        if not idx_res.success:
            logging.error("encode_collection (index) failed: %s", idx_res.error)
            return 1
        logging.info("INDEX stored: %s", idx_res.data)

        # Encode to ONE-HOT (bitset): save in onehot_collection
        logging.info("Encoding (BITSET one-hot) -> %s", onehot_collection)
        oh_res = await oh.encode_collection(
            source_collection,
            text_field=text_field,
            target_collection=onehot_collection,
            storage="bitset",
            limit=limit,
            batch_size=max(200, batch_size // 5),
            upsert=True,
        )
        if not oh_res.success:
            logging.error("encode_collection (bitset) failed: %s", oh_res.error)
            return 1
        logging.info("BITSET stored: %s", oh_res.data)

    return 0

async def run_single(
    source_collection: str,
    text_field: str,
    *,
    vocab_collection: str,
    vocab_key: str,
    index_collection: str,
    onehot_collection: str,
    doc_id: str,
    mode: str = "word",
    lowercase: bool = True,
    remove_stopwords: bool = True,
    token_pattern: str = r"[A-Za-z0-9']+",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    include_oov_token: bool = True,
    force_refit: bool = False,
) -> int:
    cfg = VocabConfig(
        mode="char" if mode == "char" else "word",
        lowercase=lowercase,
        token_pattern=token_pattern,
        remove_stopwords=remove_stopwords,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        include_oov_token=include_oov_token,
    )

    async with ZMongo() as repo:
        oh = ZOneHotDB(
            repo,
            vocab_collection=vocab_collection,
            vocab_key=vocab_key,
            config=cfg,
        )

        # Ensure a vocab exists (load or fit)
        need_fit = force_refit
        if not need_fit:
            chk = await oh.load_vocab()
            need_fit = (not chk.success) or (chk.data is None)

        if need_fit:
            # fit a minimal vocab from just this one doc (fallback) to guarantee indices exist
            src = await repo.find_document(source_collection, {"_id": doc_id})
            if not src.success or not src.data:
                logging.error("Could not fetch source doc for initial vocab: %s", getattr(src, "error", None))
                return 1
            text_val = DataProcessor.get_value(src.data, text_field)
            if not isinstance(text_val, str) or not text_val:
                logging.error("Text field '%s' missing or empty in single-doc fit.", text_field)
                return 1
            fit = await oh.fit_from_texts([text_val])
            if not fit.success:
                logging.error("fit_from_texts (single-doc) failed: %s", fit.error)
                return 1

        # Encode/store INDEX
        r1 = await oh.encode_and_store_one(
            source_collection=source_collection,
            text_field=text_field,
            target_collection=index_collection,
            doc_id=doc_id,
            storage="index",
            extra_metadata={"run": "single"},
            upsert=True,
        )
        if not r1.success:
            logging.error("encode_and_store_one (index) failed: %s", r1.error)
            return 1
        logging.info("INDEX stored for _id=%s -> %s", doc_id, index_collection)

        # Encode/store BITSET
        r2 = await oh.encode_and_store_one(
            source_collection=source_collection,
            text_field=text_field,
            target_collection=onehot_collection,
            doc_id=doc_id,
            storage="bitset",
            extra_metadata={"run": "single"},
            upsert=True,
        )
        if not r2.success:
            logging.error("encode_and_store_one (bitset) failed: %s", r2.error)
            return 1
        logging.info("BITSET stored for _id=%s -> %s", doc_id, onehot_collection)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo: one-hot/index encoding for a Mongo collection field using OneHotDB (ZMongo)."
    )
    p.add_argument("--doc-id", help="Process only this document (string _id); fetch text from the source collection by _id.")

    p.add_argument("--source-collection", required=True, help="Collection containing raw text.")
    p.add_argument("--text-field", required=True, help="Dot-path to the text field (e.g., 'text' or 'details.body').")

    p.add_argument("--vocab-collection", default="onehot_vocab", help="Collection to store vocabulary.")
    p.add_argument("--vocab-key", required=True, help="Key to identify this vocab entry (e.g., 'cases:word').")

    p.add_argument("--index-collection", help="Target collection to store *index* encodings.")
    p.add_argument("--onehot-collection", help="Target collection to store *bitset one-hot* encodings.")

    # Vocab config
    p.add_argument("--mode", choices=["word", "char"], default="word", help="Tokenization mode.")
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing.")
    p.add_argument("--no-stopwords", action="store_true", help="Disable stopword removal (word mode).")
    p.add_argument("--token-pattern", default=r"[A-Za-z0-9']+", help="Regex for word-mode tokenization.")
    p.add_argument("--min-freq", type=int, default=1, help="Minimum frequency for tokens to be in vocab.")
    p.add_argument("--max-vocab-size", type=int, default=None, help="Max vocab size (optional).")
    p.add_argument("--no-oov", action="store_true", help="Disable <UNK> token.")

    # Paging/limits
    p.add_argument("--limit", type=int, default=10000, help="Max docs to process when fitting/encoding.")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size for source reads.")
    p.add_argument("--force-refit", action="store_true", help="Force re-fitting the vocabulary.")

    # Logging
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Default targets if not provided
    index_collection = args.index_collection or f"{args.source_collection}_index"
    onehot_collection = args.onehot_collection or f"{args.source_collection}_onehot"

    if args.doc_id:
        rc = asyncio.run(
            run_single(
                source_collection=args.source_collection,
                text_field=args.text_field,
                vocab_collection=args.vocab_collection,
                vocab_key=args.vocab_key,
                index_collection=index_collection,
                onehot_collection=onehot_collection,
                doc_id=args.doc_id,
                mode=args.mode,
                lowercase=not args.no_lowercase,
                remove_stopwords=not args.no_stopwords,
                token_pattern=args.token_pattern,
                min_freq=args.min_freq,
                max_vocab_size=args.max_vocab_size,
                include_oov_token=not args.no_oov,
                force_refit=args.force_refit,
            )
        )
        raise SystemExit(rc)


    rc = asyncio.run(
        run_demo(
            source_collection=args.source_collection,
            text_field=args.text_field,
            vocab_collection=args.vocab_collection,
            vocab_key=args.vocab_key,
            index_collection=index_collection,
            onehot_collection=onehot_collection,
            mode=args.mode,
            lowercase=not args.no_lowercase,
            remove_stopwords=not args.no_stopwords,
            token_pattern=args.token_pattern,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            include_oov_token=not args.no_oov,
            limit=args.limit,
            batch_size=args.batch_size,
            force_refit=args.force_refit,
        )
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
