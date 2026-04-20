"""Ontology-based aspect normalization.

Maps raw aspect strings from ABSA extraction to canonical ontology terms
using exact synonym matching and embedding-based fuzzy matching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

_ONTOLOGY_DIR = Path(__file__).parent / "ontologies"


class Normalizer:
    """Loads a domain ontology and normalizes raw aspect strings."""

    def __init__(self, domain: str, llm: LLMProvider | None = None):
        self.domain = domain
        self.llm = llm
        self.ontology = self._load_ontology(domain)
        # Build reverse synonym -> canonical name map
        self.synonym_map: dict[str, str] = {}
        self.canonical_terms: list[str] = []
        self.categories: dict[str, list[str]] = self.ontology.get("categories", {})
        self.ranking_weights: dict[str, float] = self.ontology.get(
            "ranking_weights", {"vector": 0.40, "attribute": 0.35, "sentiment": 0.25}
        )
        # Flatten all terrain/category values for matching
        self._all_category_values: set[str] = set()
        for values in self.categories.values():
            for v in values:
                self._all_category_values.add(v.lower().replace("-", " ").replace("_", " "))

        for attr_name, attr_def in self.ontology.get("attributes", {}).items():
            self.canonical_terms.append(attr_name)
            # The canonical name maps to itself
            self.synonym_map[attr_name.lower()] = attr_name
            # Also map with spaces instead of underscores
            self.synonym_map[attr_name.lower().replace("_", " ")] = attr_name
            for syn in attr_def.get("synonyms", []):
                self.synonym_map[syn.lower()] = attr_name
        # Pre-computed embeddings for fuzzy matching (lazy)
        self._term_embeddings: dict[str, list[float]] | None = None

    def _load_ontology(self, domain: str) -> dict:
        path = _ONTOLOGY_DIR / f"{domain}.yaml"
        if not path.exists():
            logger.warning("No ontology file for domain %r at %s", domain, path)
            return {"attributes": {}, "categories": {}}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def normalize(self, raw_aspect: str) -> str:
        """Map a raw aspect string to a canonical ontology term.

        Returns the canonical term, or the raw aspect prefixed with
        '_unmatched:' if no match is found.
        """
        key = raw_aspect.lower().strip()

        # 1. Exact synonym lookup
        if key in self.synonym_map:
            return self.synonym_map[key]

        # 2. Substring matching — check if any synonym is contained in the raw aspect
        #    or vice versa
        for syn, canonical in self.synonym_map.items():
            if len(syn) >= 3 and (syn in key or key in syn):
                return canonical

        # 3. Check if it matches a category value (terrain type etc.)
        key_clean = key.replace("-", " ").replace("_", " ")
        if key_clean in self._all_category_values:
            # Return as a terrain/category marker
            return f"terrain:{key_clean.replace(' ', '_')}"

        # 4. Fuzzy match via embeddings if LLM provider is available
        if self.llm and self.canonical_terms:
            best = self._fuzzy_match(key)
            if best:
                return best

        return f"_unmatched:{raw_aspect}"

    def _fuzzy_match(self, raw: str, threshold: float = 0.75) -> str | None:
        """Embedding-based fuzzy matching against canonical terms."""
        if self._term_embeddings is None:
            self._term_embeddings = {}
            for term in self.canonical_terms:
                try:
                    # Embed the human-readable form
                    readable = term.replace("_", " ")
                    self._term_embeddings[term] = self.llm.embed(readable)
                except Exception:
                    logger.warning("Failed to embed ontology term %r", term)

        try:
            raw_vec = self.llm.embed(raw)
        except Exception:
            return None

        best_score = -1.0
        best_term = None
        for term, term_vec in self._term_embeddings.items():
            score = _cosine_sim(raw_vec, term_vec)
            if score > best_score:
                best_score = score
                best_term = term

        if best_term and best_score >= threshold:
            return best_term
        return None

    def get_weights(self) -> dict[str, float]:
        """Return the ranking weights from the ontology."""
        return dict(self.ranking_weights)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
