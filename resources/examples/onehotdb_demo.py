# onehotdb_demo.py
# Python 3.10+
#
# A modern demonstration for the ZOneHotDB lossless text encoding library.
# This script showcases the full encode-decode-verify cycle.

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from zmongo_toolbag.zmongo import ZMongo
from onehotdb.zonehotdb import ZOneHotDB, VocabConfig

# --- Demo Content ---
# A sample text with various challenges for reconstruction:
# mixedCase, ALL CAPS, punctuation, and leading/trailing whitespace.
DEMO_TEXT_CONTENT = """
Welcome to the ZOneHotDB demonstration!

This script tests the perfect, character-for-character reconstruction of text.
It handles various capitalization styles, like Title Case, mixedCase, and even ALL CAPS.
It also preserves all punctuation (like commas, periods, and exclamation marks!) and whitespace,
    including newlines and this indented line.

- Test Item 1
- Test Item 2

The process should be completely lossless.
"""


async def run_demo(
        doc_id: str,
        text_to_encode: str,
        documents_collection: str,
        vocab_collection: str,
        force_cleanup: bool,
) -> int:
    """
    Performs the encode-decode-verify cycle for ZOneHotDB.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    logging.info("--- Starting ZOneHotDB Demo ---")

    async with ZMongo() as repo:
        # 1. Configure ZOneHotDB for lossless reconstruction.
        #    - `encode_capitalization=True` is essential for perfect case matching.
        #    - `remove_stopwords=False` ensures all tokens are preserved.
        lossless_config = VocabConfig(encode_capitalization=True, remove_stopwords=False)
        db = ZOneHotDB(
            documents_collection_name=documents_collection,
            vocab_collection_name=vocab_collection,
            repo=repo,
            config=lossless_config
        )

        # 2. Clean up previous runs if requested.
        if force_cleanup:
            logging.info("Forcing cleanup of previous demo data...")
            await repo.delete_documents(documents_collection, {})
            await repo.delete_documents(vocab_collection, {})
            logging.info("Cleanup complete.")

        # 3. Encode the text and store it in the database.
        logging.info(f"Encoding and storing text for document_id: '{doc_id}'")
        await db.encode_and_store_text(doc_id, text_to_encode)
        indices = await db.get_encoded_indices(doc_id)
        mask = await db.get_capitalization_mask(doc_id)
        logging.info(f" -> Stored {len(indices)} indices and {len(mask)} mask entries.")
        if len(indices) == 0:
            logging.error("Encoding resulted in zero indices. Cannot proceed.")
            return 1

        # 4. Reconstruct the text from the stored data.
        logging.info("Reconstructing the document from the database...")
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
                    reconstructed_token = word  # Fallback on parsing error

            reconstructed_tokens.append(reconstructed_token)

        reconstructed_text = "".join(reconstructed_tokens)

        # 5. Verify that the reconstructed text matches the original.
        print("\n" + "=" * 20 + " VERIFICATION " + "=" * 20)
        if text_to_encode == reconstructed_text:
            print("\n✅ SUCCESS: Reconstructed text perfectly matches the original content.")
            return 0
        else:
            print("\n❌ FAILURE: Reconstructed text does NOT match the original.")
            print(f"Original length: {len(text_to_encode)}, Reconstructed length: {len(reconstructed_text)}")
            # Find and display the first point of failure for easier debugging.
            for i, (orig_char, recon_char) in enumerate(zip(text_to_encode, reconstructed_text)):
                if orig_char != recon_char:
                    print(f"\nMismatch found at character index {i}:")
                    context = 25
                    start = max(0, i - context)
                    end = i + context
                    print(f"  Original snippet:     ...{repr(text_to_encode[start:end])}...")
                    print(f"  Reconstructed snippet:  ...{repr(reconstructed_text[start:end])}...")
                    print(f"  Character values: Original '{repr(orig_char)}' vs Reconstructed '{repr(recon_char)}'")
                    break
            return 1


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="A demonstration of the ZOneHotDB lossless text encoding library."
    )
    parser.add_argument(
        "--doc-id",
        default="demo_document_123",
        help="A unique identifier for the document to be processed."
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional path to a UTF-8 text file to use as input. If omitted, uses internal demo text."
    )
    parser.add_argument(
        "--docs-collection",
        default="onehot_docs_demo",
        help="MongoDB collection for storing the encoded documents."
    )
    parser.add_argument(
        "--vocab-collection",
        default="onehot_vocab_demo",
        help="MongoDB collection for storing the token vocabulary."
    )
    parser.add_argument(
        "--force-cleanup",
        action="store_true",
        help="If set, completely clears the demo collections before running."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level."
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Determine the text content to use for the demo
    text_content = DEMO_TEXT_CONTENT
    if args.text_file:
        try:
            if not args.text_file.exists():
                logging.error(f"Error: The specified text file does not exist: {args.text_file}")
                raise SystemExit(1)
            text_content = args.text_file.read_text(encoding='utf-8')
            logging.info(f"Successfully loaded content from '{args.text_file}'.")
        except Exception as e:
            logging.error(f"Failed to read text file: {e}")
            raise SystemExit(1)

    # Run the asynchronous demo
    exit_code = asyncio.run(
        run_demo(
            doc_id=args.doc_id,
            text_to_encode=text_content,
            documents_collection=args.docs_collection,
            vocab_collection=args.vocab_collection,
            force_cleanup=args.force_cleanup,
        )
    )

    if exit_code == 0:
        print("\n--- Demo Complete ---")
    else:
        print("\n--- Demo Finished with Errors ---")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
