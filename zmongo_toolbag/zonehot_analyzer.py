# onehot_analyzer.py
# Python 3.10+
#
# =============================================================================
# OneHotAnalyzer: Data Analysis for ZOneHotDB
# =============================================================================
"""
This module provides the `OneHotAnalyzer` class, an object-oriented tool for
performing data analysis on text documents that have been encoded and stored
using the ZOneHotDB library.

The analyzer connects to the same MongoDB collections used by ZOneHotDB and
leverages the highly efficient numerical index representation to perform
various text analysis tasks, such as term frequency analysis, document searching,
and similarity scoring.

Key Features:
- Loads all encoded document data into a pandas DataFrame for efficient analysis.
- Provides methods for term frequency, document searching, and cosine similarity.
- Works directly with a live ZOneHotDB database.

Prerequisites:
- A running MongoDB instance with data stored by zonehotdb.py.
- A .env file configured with MONGO_URI and MONGO_DATABASE_NAME.
"""

import asyncio
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# --- Ensure correct imports for the library ---
try:
    # Correctly import constants along with the class
    from zmongo_toolbag.zonehotdb import ZOneHotDB, ENCODED_INDICES_FIELD, LINK_KEY_FIELD
except (ImportError, ModuleNotFoundError):
    # Adjust path for local testing if needed
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from zmongo_toolbag.zonehotdb import ZOneHotDB, ENCODED_INDICES_FIELD, LINK_KEY_FIELD


class OneHotAnalyzer:
    """
    An object-oriented tool to perform data analysis on collections of documents
    encoded by the ZOneHotDB library.
    """

    def __init__(
            self,
            documents_collection_name: str = "documents",
            vocab_collection_name: str = "onehot_vocabulary",
    ):
        """
        Initializes the OneHotAnalyzer.

        Args:
            documents_collection_name: The name of the MongoDB collection where
                                       the encoded documents are stored.
            vocab_collection_name: The name of the MongoDB collection for the
                                   shared vocabulary.
        """
        self.db = ZOneHotDB(
            documents_collection_name=documents_collection_name,
            vocab_collection_name=vocab_collection_name
        )
        self.documents_df: Optional[pd.DataFrame] = None
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self.vocabulary_size: int = 0

    async def load_data(self) -> None:
        """
        Loads all encoded documents and the full vocabulary from the database
        into memory for analysis. This method must be called before any other
        analysis methods.
        """
        print("Loading data from the database...")
        # 1. Fetch all documents and their indices
        all_docs_cursor = await self.db.repo.find_documents(
            self.db.documents_collection, {}, limit=1_000_000
        )
        if not all_docs_cursor.success:
            raise RuntimeError(f"Failed to load documents: {all_docs_cursor.error}")

        docs_data = []
        for doc in all_docs_cursor.data or []:
            # FIX: Use the directly imported constants, not self.db.CONSTANT
            indices_str = doc.get(ENCODED_INDICES_FIELD, "")
            indices = [int(i) for i in indices_str.split(',') if i.isdigit()]
            docs_data.append({
                "doc_id": doc.get(LINK_KEY_FIELD),
                "indices": indices
            })

        self.documents_df = pd.DataFrame(docs_data)
        if self.documents_df.empty:
            print("Warning: No documents found in the collection.")
            return

        self.documents_df.set_index("doc_id", inplace=True)
        print(f"Successfully loaded {len(self.documents_df)} documents.")

        # 2. Fetch the vocabulary and create mappings
        all_words = await self.db.lexicon.get_all_words()
        self.vocabulary_size = len(all_words)
        self.word_to_index = {word: i + 1 for i, word in enumerate(all_words)}
        self.index_to_word = {i + 1: word for i, word in enumerate(all_words)}
        print(f"Successfully loaded a vocabulary of {self.vocabulary_size} unique terms.")

    def _get_doc_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """Converts a document's list of indices into a term frequency vector."""
        if self.documents_df is None or doc_id not in self.documents_df.index:
            return None

        indices = self.documents_df.loc[doc_id, "indices"]
        vector = np.zeros(self.vocabulary_size, dtype=int)
        for index in indices:
            if 1 <= index <= self.vocabulary_size:
                vector[index - 1] += 1
        return vector

    def get_term_frequency(self, term: str) -> pd.DataFrame:
        """
        Calculates the frequency of a given term across all documents.

        Args:
            term: The word to search for.

        Returns:
            A pandas DataFrame listing the documents that contain the term and
            the frequency of the term in each document.
        """
        if self.documents_df is None:
            raise RuntimeError("Data not loaded. Please call .load_data() first.")

        term_index = self.word_to_index.get(term.lower())
        if not term_index:
            return pd.DataFrame(columns=["doc_id", "frequency"]).set_index("doc_id")

        results = []
        for doc_id, row in self.documents_df.iterrows():
            count = row["indices"].count(term_index)
            if count > 0:
                results.append({"doc_id": doc_id, "frequency": count})

        if not results:
            return pd.DataFrame(columns=["doc_id", "frequency"]).set_index("doc_id")

        return pd.DataFrame(results).set_index("doc_id")

    def find_documents_with_terms(self, terms: List[str], condition: str = 'AND') -> List[str]:
        """
        Finds documents containing a given list of terms.

        Args:
            terms: A list of words to search for.
            condition: 'AND' requires all terms to be present. 'OR' requires
                       at least one term to be present.

        Returns:
            A list of document IDs that match the search criteria.
        """
        if self.documents_df is None:
            raise RuntimeError("Data not loaded. Please call .load_data() first.")

        term_indices = {self.word_to_index.get(t.lower()) for t in terms}
        term_indices.discard(None)  # Remove terms not in vocabulary

        if not term_indices:
            return []

        matching_docs = []
        for doc_id, row in self.documents_df.iterrows():
            doc_indices = set(row["indices"])
            if condition.upper() == 'AND':
                if term_indices.issubset(doc_indices):
                    matching_docs.append(doc_id)
            elif condition.upper() == 'OR':
                if not term_indices.isdisjoint(doc_indices):
                    matching_docs.append(doc_id)
        return matching_docs

    def calculate_cosine_similarity(self, doc1_id: str, doc2_id: str) -> Optional[float]:
        """
        Calculates the cosine similarity between two documents.

        Returns:
            A float between 0 (not similar) and 1 (identical content).
        """
        if self.documents_df is None:
            raise RuntimeError("Data not loaded. Please call .load_data() first.")

        vec1 = self._get_doc_vector(doc1_id)
        vec2 = self._get_doc_vector(doc2_id)

        if vec1 is None or vec2 is None:
            print(f"Error: One or both document IDs not found.")
            return None

        # scipy.spatial.distance.cosine returns the distance (1 - similarity)
        return 1 - cosine(vec1, vec2)

    def get_most_common_terms(self, top_n: int = 10) -> List[tuple[str, int]]:
        """
        Finds the most common terms across the entire document collection.

        Returns:
            A list of (term, count) tuples.
        """
        if self.documents_df is None:
            raise RuntimeError("Data not loaded. Please call .load_data() first.")

        all_indices = [idx for index_list in self.documents_df["indices"] for idx in index_list]
        counter = Counter(all_indices)

        most_common = []
        for index, count in counter.most_common(top_n):
            word = self.index_to_word.get(index, f"UNKNOWN_INDEX_{index}")
            most_common.append((word, count))

        return most_common


async def _demo():
    """A simple demonstration of the OneHotAnalyzer class."""

    # --- Setup: Ensure there is data to analyze ---
    print("--- Setting up demo data ---")
    db = ZOneHotDB()
    await db.repo.delete_documents(db.documents_collection, {})
    await db.repo.delete_documents(db.lexicon.collection_name, {})

    doc1_content = "The quick brown fox jumps over the lazy dog."
    doc2_content = "A quick brown dog jumps over a lazy cat."
    doc3_content = "The cat and the dog are friends."


    await db.encode_and_store_text("doc1", doc1_content)
    await db.encode_and_store_text("doc2", doc2_content)
    await db.encode_and_store_text("doc3", doc3_content)
    print("Demo data has been stored in the database.")

    # --- Analysis ---
    print("\n--- Running Data Analysis Demo ---")
    analyzer = OneHotAnalyzer(
        documents_collection_name=db.documents_collection,
        vocab_collection_name=db.lexicon.collection_name
    )

    await analyzer.load_data()

    # Example 1: Get frequency of the word "dog"
    print("\n1. Frequency of the term 'dog':")
    freq_df = analyzer.get_term_frequency("dog")
    print(freq_df)

    # Example 2: Find documents containing 'quick' AND 'fox'
    print("\n2. Documents containing 'quick' AND 'fox':")
    docs_and = analyzer.find_documents_with_terms(["quick", "fox"], condition='AND')
    print(docs_and)

    # Example 3: Find documents containing 'cat' OR 'fox'
    print("\n3. Documents containing 'cat' OR 'fox':")
    docs_or = analyzer.find_documents_with_terms(["cat", "fox"], condition='OR')
    print(docs_or)

    # Example 4: Calculate similarity between doc1 and doc2
    print("\n4. Cosine Similarity between doc1 and doc2:")
    similarity = analyzer.calculate_cosine_similarity("doc1", "doc2")
    print(f"Similarity Score: {similarity:.4f}")

    # Example 5: Get the 5 most common terms in the collection
    print("\n5. Top 5 most common terms:")
    common_terms = analyzer.get_most_common_terms(top_n=5)
    print(common_terms)

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    # Ensure you have a .env file with your MONGO_URI and MONGO_DATABASE_NAME
    load_dotenv()
    if not os.getenv("MONGO_URI") or not os.getenv("MONGO_DATABASE_NAME"):
        print("FATAL: MONGO_URI and MONGO_DATABASE_NAME must be set in your .env file.")
    else:
        asyncio.run(_demo())

