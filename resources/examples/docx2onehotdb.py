#!/usr/bin/env python3
"""
demo_onehot_from_docx.py

Read a .docx file, extract text, and store it in OneHotDB using the new ZMongo backend.

Usage:
  python demo_onehot_from_docx.py /path/to/file.docx \
      --link-key my_doc_key \
      --collection sentences \
      --vocab-collection onehot_vocab

Notes:
- Requires your updated `zonehotdb.py` and `zmongo.py` to be importable (same folder or in PYTHONPATH).
- Tries to use `python-docx`; falls back to `docx2txt` if available.
  Install one of them:
      pip install python-docx
  or: pip install docx2txt
- ZMongo connection is driven by environment variables (e.g., MONGO_URI, MONGO_DATABASE_NAME).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from zmongo_toolbag.zmongo import ZMongo
from onehotdb.zonehotdb import ZOneHotDB


# --- .docx text extraction helpers ------------------------------------------------

def _extract_text_with_python_docx(path: Path) -> str | None:
    try:
        import docx  # python-docx
    except Exception:
        return None
    doc = docx.Document(str(path))
    # Join non-empty paragraphs with newlines
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def _extract_text_with_docx2txt(path: Path) -> str | None:
    try:
        import docx2txt
    except Exception:
        return None
    txt = docx2txt.process(str(path))
    return txt if txt and txt.strip() else ""

def extract_text_from_docx(path: Path) -> str:
    """
    Try python-docx first, then docx2txt. Raise if neither is available.
    """
    text = _extract_text_with_python_docx(path)
    if text is None:
        text = _extract_text_with_docx2txt(path)
    if text is None:
        raise RuntimeError(
            "No .docx reader found. Please install one:\n"
            "  pip install python-docx\n"
            "or:\n"
            "  pip install docx2txt"
        )
    return text.strip()

# --- OneHot demo logic ------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description="Store a .docx file's text into OneHotDB.")
    parser.add_argument("filepath", type=str, help="Path to a .docx file")
    parser.add_argument("--link-key", type=str, default=None,
                        help="Logical key to store under (defaults to filename stem)")
    parser.add_argument("--collection", type=str, default="sentences",
                        help="Mongo collection to store encoded sentences (default: sentences)")
    parser.add_argument("--vocab-collection", type=str, default="onehot_vocab",
                        help="Mongo collection to store vocabulary (default: onehot_vocab)")
    parser.add_argument("--print-top", type=int, default=20,
                        help="Print up to N non-zero columns after write (default: 20)")

    args = parser.parse_args()
    path = Path(args.filepath)

    if not path.exists() or path.suffix.lower() != ".docx":
        print(f"Error: {path} does not exist or is not a .docx file.", file=sys.stderr)
        return 2



    # Extract text
    text = extract_text_from_docx(path)
    if not text:
        print("The .docx file contained no text after extraction.", file=sys.stderr)
        return 1

    link_key = args.link_key or path.stem

    # Connect and write
    async with ZMongo() as repo:
        oh = ZOneHotDB(_table_name=args.collection, repo=repo, vocab_collection=args.vocab_collection)

        # Store the one-hot indices CSV under 'sentence'
        await oh.put_onehot(link_key, text)

        # Fetch indices and show a brief summary
        idx_list = await oh.get_onehot_list(link_key)
        vocab_size = await oh.get_onehot_matrix(link_key)  # number of words in vocab
        print(f"\nStored one-hot for link_key='{link_key}' in collection='{args.collection}'.")
        print(f"Total indices stored: {len(idx_list)}")
        print(f"Current vocabulary size: {vocab_size}")

        # Optionally print up to N non-zero columns with names
        if args.print_top > 0:
            df = await oh.get_onehot(link_key, _use_column_names=True, _count_uses=False)
            nonzero = [c for c, v in zip(df.columns, df.iloc[0].tolist()) if v]
            show = nonzero[: args.print_top]
            print(f"\nFirst {len(show)} active columns:")
            for c in show:
                print(f"  - {c}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
