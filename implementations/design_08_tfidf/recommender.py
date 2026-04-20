"""Design #8: TF-IDF + Learned Attribute Weights Recommender.

Uses LLM for attribute extraction and query parsing, then scores
products against queries using IDF-weighted cosine similarity over
sparse feature vectors.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the shared package is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult  # noqa: E402
from shared.llm_provider import LLMProvider, get_provider  # noqa: E402

from .catalog import AttributeDef, get_catalog  # noqa: E402
from .encoder import encode_attribute  # noqa: E402
from .scorer import (  # noqa: E402
    explain_score,
    per_attribute_contributions,
    weighted_cosine,
)
from .synonyms import build_synonym_map, expand_value  # noqa: E402
from .weighting import compute_idf_weights  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output.

    Handles common issues: markdown fences, trailing commas, etc.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        # Remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
    return {}


def _build_extraction_prompt(
    review_text: str,
    catalog: list[AttributeDef],
) -> str:
    """Build the LLM prompt for extracting attributes from a review."""
    attr_specs = json.dumps(
        [a.to_prompt_dict() for a in catalog], indent=2
    )
    return f"""Extract product attributes from the following review.

Known attributes:
{attr_specs}

Review:
\"\"\"{review_text}\"\"\"

Return a JSON object with:
- Attribute names as keys, extracted values as values.
- For ordinal attributes, return an integer from 1 to 10 indicating intensity
  (e.g., stiffness: 9 for "very stiff", 5 for "medium", 2 for "quite soft").
- For categorical attributes, return the matching value(s) from the allowed list.
  If multiple values apply, return a list.
- For boolean attributes, return true or false.
- For numeric attributes, return the number.
- Only include attributes that are clearly mentioned or strongly implied.

Return ONLY valid JSON, no commentary."""


def _build_query_prompt(
    query_text: str,
    catalog: list[AttributeDef],
) -> str:
    """Build the LLM prompt for parsing a user query into attributes."""
    attr_specs = json.dumps(
        [a.to_prompt_dict() for a in catalog], indent=2
    )
    return f"""Parse this product query into desired attributes.

Known attributes:
{attr_specs}

Query: "{query_text}"

Return a JSON object with two keys:
1. "desired" — an object mapping attribute names to their desired values.
   For ordinal attributes use an integer 1-10.
   For categorical attributes use the value(s) from the allowed list.
   For boolean attributes use true/false.
2. "negations" — an object mapping attribute names to values the user
   explicitly does NOT want (e.g., "no rocker" -> {{"rocker": true}}).

Only include attributes the query mentions or strongly implies.
Return ONLY valid JSON, no commentary."""


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

class TFIDFRecommender:
    """Recommender using TF-IDF weighted attribute vectors.

    Implements the shared ``Recommender`` protocol from
    ``shared.interface``.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or get_provider()
        # Per-domain state populated by ingest()
        self._catalogs: dict[str, list[AttributeDef]] = {}
        self._products: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
        self._idf_weights: dict[str, dict[str, float]] = {}
        self._synonym_maps: dict[str, dict[str, str]] = {}
        # Numeric normalization ranges learned during ingestion
        self._numeric_ranges: dict[str, dict[str, tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> None:
        """Ingest product + review data for *domain*.

        Steps:
        1. Load/build attribute catalog and synonym map.
        2. For each product, try to use pre-computed attributes from the
           product data first (fast path).  Fall back to LLM extraction
           from reviews when attributes are not available.
        3. Compute IDF weights across all product vectors.
        """
        catalog = get_catalog(domain)
        if not catalog:
            logger.warning("No catalog for domain %r — using empty catalog", domain)
        self._catalogs[domain] = catalog
        self._synonym_maps[domain] = build_synonym_map(domain)
        syn_map = self._synonym_maps[domain]

        # Group reviews by product id
        reviews_by_product: dict[str, list[str]] = defaultdict(list)
        for review in reviews:
            pid = review.get("product_id", review.get("id", ""))
            text = review.get("review_text", review.get("text", ""))
            if pid and text:
                reviews_by_product[pid].append(text)

        product_vectors: list[tuple[str, str, dict[str, float]]] = []
        for product in products:
            pid = product.get("product_id", product.get("id", ""))
            pname = product.get("product_name", product.get("name", ""))

            # Fast path: use pre-computed attributes from product data
            precomputed = product.get("attributes", {})
            specs = product.get("specs", product.get("metadata", {}))

            if precomputed:
                fvec = self._encode_precomputed(precomputed, specs, catalog, syn_map)
            else:
                # Slow path: extract from reviews via LLM
                fvec = self._extract_from_reviews(
                    reviews_by_product.get(pid, []), catalog, syn_map
                )

            product_vectors.append((pid, pname, fvec))

        # Normalize numeric attributes to [0, 1] across the corpus
        self._normalize_numeric_features(product_vectors, catalog, domain)

        self._products[domain] = product_vectors
        all_vecs = [pv[2] for pv in product_vectors]
        self._idf_weights[domain] = compute_idf_weights(all_vecs)

        logger.info(
            "Ingested %d products for domain %r (%d feature keys)",
            len(product_vectors),
            domain,
            len(self._idf_weights[domain]),
        )

    def _normalize_numeric_features(
        self,
        product_vectors: list[tuple[str, str, dict[str, float]]],
        catalog: list[AttributeDef],
        domain: str = "",
    ) -> None:
        """Normalize numeric feature keys to [0, 1] across the corpus (in-place).

        Also stores the min/max ranges so query numeric values can be
        normalized consistently.
        """
        numeric_keys = {a.name for a in catalog if a.attr_type == "numeric"}
        if not numeric_keys:
            return

        ranges: dict[str, tuple[float, float]] = {}

        for key in numeric_keys:
            values = [
                pv[2][key] for pv in product_vectors if key in pv[2] and pv[2][key] != 0.0
            ]
            if not values:
                continue
            vmin = min(values)
            vmax = max(values)
            ranges[key] = (vmin, vmax)
            spread = vmax - vmin
            if spread == 0:
                for _, _, fvec in product_vectors:
                    if key in fvec:
                        fvec[key] = 0.5
            else:
                for _, _, fvec in product_vectors:
                    if key in fvec:
                        fvec[key] = (fvec[key] - vmin) / spread

        if domain:
            self._numeric_ranges[domain] = ranges

    def _encode_precomputed(
        self,
        attributes: dict,
        specs: dict,
        catalog: list[AttributeDef],
        syn_map: dict[str, str],
    ) -> dict[str, float]:
        """Build a feature vector from pre-computed product attributes."""
        fvec: dict[str, float] = {}
        for attr_def in catalog:
            name = attr_def.name
            # Check attributes dict first, then specs
            value = attributes.get(name, specs.get(name))
            if value is None:
                # Try alternate key formats in specs
                alt_keys = {
                    "waist_width_mm": "waist_width_mm",
                    "turn_radius": "turn_radius_m",
                    "weight": "weight_g_per_ski",
                }
                alt = alt_keys.get(name)
                if alt and alt in specs:
                    value = specs[alt]
                    # Normalize numeric specs to 1-10 scale
                    if name == "turn_radius" and attr_def.attr_type == "ordinal":
                        # turn_radius_m: ~12 (short) to ~22 (long) -> 1-10
                        value = max(1, min(10, int((float(value) - 10) / 1.3 + 1)))
                    elif name == "weight" and attr_def.attr_type == "ordinal":
                        # weight_g_per_ski: ~1200 (light) to ~2200 (heavy) -> 1-10
                        value = max(1, min(10, int((float(value) - 1100) / 120 + 1)))
            if value is None:
                # Handle rocker from rocker_profile spec
                if name == "rocker" and "rocker_profile" in specs:
                    profile = specs["rocker_profile"]
                    value = "camber" not in profile.lower() or "rocker" in profile.lower()
                    if profile.lower() == "camber":
                        value = False
                else:
                    continue

            # Expand synonyms on string values
            if isinstance(value, str):
                value = expand_value(value, syn_map)
            elif isinstance(value, list):
                value = [expand_value(v, syn_map) if isinstance(v, str) else v for v in value]

            encoded = encode_attribute(attr_def, value)
            fvec.update(encoded)

        return fvec

    def _extract_from_reviews(
        self,
        review_texts: list[str],
        catalog: list[AttributeDef],
        syn_map: dict[str, str],
    ) -> dict[str, float]:
        """Extract attributes from reviews via LLM and aggregate."""
        if not review_texts:
            return {}

        # Batch reviews to reduce LLM calls — combine up to 3
        combined_texts = []
        batch_size = 3
        for i in range(0, len(review_texts), batch_size):
            batch = review_texts[i : i + batch_size]
            combined_texts.append("\n---\n".join(batch))

        all_extractions: list[dict] = []
        for text in combined_texts:
            prompt = _build_extraction_prompt(text, catalog)
            try:
                response = self.llm.generate(prompt, json_mode=True)
                extracted = _extract_json(response)
                if extracted:
                    all_extractions.append(extracted)
            except Exception:
                logger.exception("LLM extraction failed for a review batch")

        if not all_extractions:
            return {}

        return self._aggregate_extractions(all_extractions, catalog, syn_map)

    def _aggregate_extractions(
        self,
        extractions: list[dict],
        catalog: list[AttributeDef],
        syn_map: dict[str, str],
    ) -> dict[str, float]:
        """Average multiple LLM extractions into a single feature vector."""
        accum: dict[str, list[float]] = defaultdict(list)

        for extraction in extractions:
            for attr_def in catalog:
                name = attr_def.name
                if name not in extraction:
                    continue
                value = extraction[name]
                # Expand synonyms
                if isinstance(value, str):
                    value = expand_value(value, syn_map)
                elif isinstance(value, list):
                    value = [
                        expand_value(v, syn_map) if isinstance(v, str) else v
                        for v in value
                    ]

                confidence = extraction.get(f"{name}_confidence", 0.8)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.8

                encoded = encode_attribute(attr_def, value, confidence=float(confidence))
                for k, v in encoded.items():
                    accum[k].append(v)

        # Average across extractions
        return {k: sum(vals) / len(vals) for k, vals in accum.items() if vals}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Parse *query_text* and return ranked recommendations."""
        catalog = self._catalogs.get(domain, [])
        weights = self._idf_weights.get(domain, {})
        syn_map = self._synonym_maps.get(domain, {})

        query_vec = self._parse_query(query_text, domain, catalog, syn_map)
        if not query_vec:
            logger.warning("Query parsing produced empty vector for: %s", query_text)

        results: list[RecommendationResult] = []
        for pid, pname, pvec in self._products.get(domain, []):
            score = weighted_cosine(query_vec, pvec, weights)
            matched = per_attribute_contributions(query_vec, pvec, weights)
            explanation = explain_score(query_vec, pvec, weights)

            results.append(
                RecommendationResult(
                    product_id=pid,
                    product_name=pname,
                    score=round(score, 4),
                    explanation=explanation,
                    matched_attributes=matched,
                )
            )

        # Sort by score descending; tie-break by number of matched attributes
        results.sort(key=lambda r: (r.score, len(r.matched_attributes)), reverse=True)
        return results[:top_k]

    def _parse_query(
        self,
        query_text: str,
        domain: str,
        catalog: list[AttributeDef],
        syn_map: dict[str, str],
    ) -> dict[str, float]:
        """Use LLM to parse *query_text* into a sparse query vector."""
        prompt = _build_query_prompt(query_text, catalog)
        try:
            response = self.llm.generate(prompt, json_mode=True)
            parsed = _extract_json(response)
        except Exception:
            logger.exception("LLM query parsing failed")
            parsed = {}

        if not parsed:
            return {}

        qvec: dict[str, float] = {}

        # Desired attributes
        for attr_name, value in parsed.get("desired", {}).items():
            attr_def = self._find_attr(attr_name, catalog)
            if not attr_def:
                continue
            # Expand synonyms
            if isinstance(value, str):
                value = expand_value(value, syn_map)
            elif isinstance(value, list):
                value = [
                    expand_value(v, syn_map) if isinstance(v, str) else v
                    for v in value
                ]
            encoded = encode_attribute(attr_def, value)
            qvec.update(encoded)

        # Negations — encode with negative weights
        for attr_name, value in parsed.get("negations", {}).items():
            attr_def = self._find_attr(attr_name, catalog)
            if not attr_def:
                continue
            if isinstance(value, str):
                value = expand_value(value, syn_map)
            encoded = encode_attribute(attr_def, value)
            for k, v in encoded.items():
                qvec[k] = -1.0 * v

        # Normalize numeric query values using ranges learned during ingestion
        ranges = self._numeric_ranges.get(domain, {})
        for key, (vmin, vmax) in ranges.items():
            if key in qvec and (vmax - vmin) > 0:
                qvec[key] = max(0.0, min(1.0, (qvec[key] - vmin) / (vmax - vmin)))

        return qvec

    @staticmethod
    def _find_attr(
        name: str, catalog: list[AttributeDef]
    ) -> AttributeDef | None:
        """Find an AttributeDef by name (case-insensitive, underscore-tolerant)."""
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        for a in catalog:
            if a.name.lower() == name_lower:
                return a
        return None
