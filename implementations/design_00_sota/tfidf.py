"""Simple TF-IDF keyword matching signal.

Built from review text during ingestion. At query time, computes cosine
similarity between the (expanded) query and each product's review corpus.

Pure Python — no external dependencies beyond collections and math.
"""

from __future__ import annotations

import math
import re
from collections import Counter


# Simple stop words to exclude from TF-IDF
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "was",
    "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "so", "if", "as", "then", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any", "because",
    "before", "between", "both", "each", "few", "get", "got", "here", "how",
    "i", "into", "me", "more", "most", "my", "now", "only", "other", "our",
    "out", "own", "same", "she", "he", "some", "such", "them", "there",
    "these", "they", "through", "up", "what", "when", "where", "which",
    "while", "who", "whom", "why", "you", "your", "we", "us",
    # Review-specific noise
    "ski", "skis", "skiing", "shoe", "shoes", "running", "run", "review",
    "product", "like", "really", "much", "well", "great", "good", "nice",
    "one", "two", "first", "best", "better", "worst", "worse", "think",
    "make", "made", "bit", "lot", "still", "even", "going", "went",
}

_WORD_RE = re.compile(r"[a-z_][a-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, removing stop words and very short tokens."""
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


class TFIDFIndex:
    """In-memory TF-IDF index for keyword-based product matching."""

    def __init__(self) -> None:
        # product_id -> term frequency Counter (normalized by doc length)
        self._tf: dict[str, dict[str, float]] = {}
        # term -> number of documents containing it
        self._df: Counter = Counter()
        # total number of documents
        self._n_docs: int = 0
        # cached IDF values
        self._idf: dict[str, float] = {}
        self._dirty: bool = True

    def add_document(self, product_id: str, text: str) -> None:
        """Add a product's review text to the index."""
        tokens = _tokenize(text)
        if not tokens:
            self._tf[product_id] = {}
            self._n_docs += 1
            self._dirty = True
            return

        tf_counts = Counter(tokens)
        max_freq = max(tf_counts.values())
        # Augmented TF to prevent bias toward long documents
        tf_norm = {t: 0.5 + 0.5 * (c / max_freq) for t, c in tf_counts.items()}

        self._tf[product_id] = tf_norm
        for term in tf_counts:
            self._df[term] += 1
        self._n_docs += 1
        self._dirty = True

    def _rebuild_idf(self) -> None:
        """Recompute IDF values from document frequencies."""
        if not self._dirty or self._n_docs == 0:
            return
        self._idf = {}
        for term, df in self._df.items():
            # Smooth IDF
            self._idf[term] = math.log((self._n_docs + 1) / (df + 1)) + 1.0
        self._dirty = False

    def query_similarity(self, query_text: str, product_id: str) -> float:
        """Compute TF-IDF cosine similarity between query and a product.

        Returns 0.0 if product_id not in index or no overlap.
        """
        self._rebuild_idf()

        doc_tf = self._tf.get(product_id)
        if not doc_tf:
            return 0.0

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return 0.0

        query_tf = Counter(query_tokens)
        max_q = max(query_tf.values())
        query_tfidf: dict[str, float] = {}
        for t, c in query_tf.items():
            tf = 0.5 + 0.5 * (c / max_q)
            idf = self._idf.get(t, 1.0)
            query_tfidf[t] = tf * idf

        # Build doc TF-IDF vector (only for overlapping terms)
        dot = 0.0
        for term, q_val in query_tfidf.items():
            d_tf = doc_tf.get(term, 0.0)
            if d_tf > 0:
                d_val = d_tf * self._idf.get(term, 1.0)
                dot += q_val * d_val

        if dot == 0.0:
            return 0.0

        # Norms
        q_norm = math.sqrt(sum(v * v for v in query_tfidf.values()))
        d_norm = math.sqrt(
            sum((tf * self._idf.get(t, 1.0)) ** 2 for t, tf in doc_tf.items())
        )

        if q_norm == 0 or d_norm == 0:
            return 0.0

        return dot / (q_norm * d_norm)

    def query_all_similarities(self, query_text: str) -> dict[str, float]:
        """Compute TF-IDF similarity for all indexed products.

        Returns dict of product_id -> similarity score.
        """
        self._rebuild_idf()

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return {}

        query_tf = Counter(query_tokens)
        max_q = max(query_tf.values())
        query_tfidf: dict[str, float] = {}
        for t, c in query_tf.items():
            tf = 0.5 + 0.5 * (c / max_q)
            idf = self._idf.get(t, 1.0)
            query_tfidf[t] = tf * idf

        q_norm = math.sqrt(sum(v * v for v in query_tfidf.values()))
        if q_norm == 0:
            return {}

        results: dict[str, float] = {}
        for pid, doc_tf in self._tf.items():
            dot = 0.0
            for term, q_val in query_tfidf.items():
                d_tf = doc_tf.get(term, 0.0)
                if d_tf > 0:
                    d_val = d_tf * self._idf.get(term, 1.0)
                    dot += q_val * d_val

            if dot > 0:
                d_norm = math.sqrt(
                    sum((tf * self._idf.get(t, 1.0)) ** 2
                        for t, tf in doc_tf.items())
                )
                if d_norm > 0:
                    results[pid] = dot / (q_norm * d_norm)

        return results

    def clear(self) -> None:
        """Clear the entire index."""
        self._tf.clear()
        self._df.clear()
        self._n_docs = 0
        self._idf.clear()
        self._dirty = True
