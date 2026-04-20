"""Multi-signal ranking: vector similarity + attribute match + sentiment."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import StructuredStore


@dataclass
class ParsedQuery:
    """Decomposed query for structured + semantic search."""

    semantic_text: str  # full query text for embedding
    required_aspects: list[str]  # normalized aspect names the user wants
    negations: list[str]  # normalized aspect names the user does NOT want
    numeric_filters: list[dict]  # e.g., {"attr": "length", "op": ">=", "val": 180}
    domain_hint: str | None = None

    @property
    def has_structured_requirements(self) -> bool:
        return bool(self.required_aspects or self.negations or self.numeric_filters)


def parse_query(
    query_text: str,
    domain: str,
    normalizer,
) -> ParsedQuery:
    """Parse a natural language query into structured components.

    Uses regex-based heuristics for negation and numeric filter detection,
    plus ontology normalization for aspect terms.
    """
    text = query_text.strip()
    negations: list[str] = []
    required_aspects: list[str] = []
    numeric_filters: list[dict] = []

    # Detect negations: "no X", "without X", "not X", "don't want X"
    neg_patterns = [
        r"\bno\s+(\w[\w\s]*?)(?:,|$|\band\b)",
        r"\bwithout\s+(\w[\w\s]*?)(?:,|$|\band\b)",
        r"\bnot\s+(\w[\w\s]*?)(?:,|$|\band\b)",
        r"\bdon'?t\s+want\s+(\w[\w\s]*?)(?:,|$|\band\b)",
        r"\bno\s+(\w+)",
        r"\bwithout\s+(\w+)",
    ]
    negation_raw: set[str] = set()
    for pattern in neg_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            negation_raw.add(match.group(1).strip().lower())

    for raw_neg in negation_raw:
        norm = normalizer.normalize(raw_neg)
        if not norm.startswith("_unmatched:"):
            negations.append(norm)

    # Detect numeric filters: "180cm+", "100mm+", ">= 180cm", etc.
    num_patterns = [
        r"(\d+)\s*(cm|mm|g|kg)\s*\+",  # 180cm+
        r">=?\s*(\d+)\s*(cm|mm|g|kg)",  # >= 180cm
        r"(\d+)\s*(cm|mm|g|kg)\s*or\s+more",  # 180cm or more
        r"under\s+(\d+)\s*(cm|mm|g|kg)",  # under 180cm
        r"<=?\s*(\d+)\s*(cm|mm|g|kg)",  # <= 180cm
    ]
    for pattern in num_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            val = int(groups[0])
            unit = groups[1].lower()
            # Determine attribute and operator
            if "under" in match.group(0).lower() or "<" in match.group(0):
                op = "<="
            else:
                op = ">="
            # Map unit to attribute
            attr_map = {"cm": "length", "mm": "waist_width", "g": "weight", "kg": "weight"}
            attr = attr_map.get(unit, unit)
            numeric_filters.append({"attr": attr, "op": op, "val": val})

    # Extract aspect terms from the query text
    # Split on commas and common conjunctions, then normalize each fragment
    # Remove negation phrases and numeric phrases first
    clean_text = text
    for pattern in neg_patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)
    for pattern in num_patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)

    # Split remaining text into candidate aspect phrases
    fragments = re.split(r"[,;]+|\band\b|\bfor\b|\bthat\b|\bwith\b", clean_text)
    for frag in fragments:
        frag = frag.strip().lower()
        # Skip very short or generic fragments
        if len(frag) < 2:
            continue
        # Skip common filler words
        skip_words = {"a", "an", "the", "ski", "skis", "shoe", "shoes", "good", "best", "great", "i", "me", "my"}
        words = frag.split()
        meaningful_words = [w for w in words if w not in skip_words]
        if not meaningful_words:
            continue

        # Try to normalize the whole fragment
        norm = normalizer.normalize(frag)
        if not norm.startswith("_unmatched:"):
            if norm not in negations:
                required_aspects.append(norm)
            continue

        # Try individual words
        for word in meaningful_words:
            norm = normalizer.normalize(word)
            if not norm.startswith("_unmatched:") and norm not in negations:
                required_aspects.append(norm)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for a in required_aspects:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    required_aspects = deduped

    return ParsedQuery(
        semantic_text=text,
        required_aspects=required_aspects,
        negations=list(set(negations)),
        numeric_filters=numeric_filters,
        domain_hint=domain,
    )


def score_product(
    product_id: str,
    parsed: ParsedQuery,
    vector_sim: float,
    store: StructuredStore,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score a product using the three-signal ranking formula.

    Returns:
        (final_score, breakdown) where breakdown maps signal names to values.
    """
    w = weights or {"vector": 0.40, "attribute": 0.35, "sentiment": 0.25}

    # Signal 1: Vector similarity (already 0-1 from cosine)
    s_vector = max(0.0, min(1.0, vector_sim))

    # Signal 2: Attribute match rate
    product_aspects = store.get_product_aspects(product_id)
    # Filter out terrain and unmatched aspects for matching
    matchable_required = [
        a for a in parsed.required_aspects if not a.startswith("terrain:")
    ]
    matched_attrs: dict[str, float] = {}
    for a in matchable_required:
        if a in product_aspects:
            matched_attrs[a] = product_aspects[a]

    # Also check terrain matches
    terrain_matches: dict[str, float] = {}
    for a in parsed.required_aspects:
        if a.startswith("terrain:") and a in product_aspects:
            terrain_matches[a] = product_aspects[a]

    all_required = matchable_required + [
        a for a in parsed.required_aspects if a.startswith("terrain:")
    ]
    total_required = max(len(all_required), 1)
    total_matched = len(matched_attrs) + len(terrain_matches)
    s_attr_match = total_matched / total_required

    # Signal 3: Sentiment strength on matched aspects
    all_sentiments = list(matched_attrs.values()) + list(terrain_matches.values())
    if all_sentiments:
        s_sentiment = sum(all_sentiments) / len(all_sentiments)
        s_sentiment = (s_sentiment + 1.0) / 2.0  # normalize [-1,1] to [0,1]
    else:
        s_sentiment = 0.5  # neutral when no aspects matched

    # Weighted combination
    final = (
        w["vector"] * s_vector
        + w["attribute"] * s_attr_match
        + w["sentiment"] * s_sentiment
    )

    breakdown = {
        "vector_similarity": s_vector,
        "attribute_match": s_attr_match,
        "sentiment_strength": s_sentiment,
    }
    for k, v in matched_attrs.items():
        breakdown[f"aspect:{k}"] = (v + 1.0) / 2.0
    for k, v in terrain_matches.items():
        breakdown[f"aspect:{k}"] = (v + 1.0) / 2.0

    return final, breakdown


def build_explanation(
    product_name: str,
    breakdown: dict[str, float],
    parsed: ParsedQuery,
    weights: dict[str, float],
) -> str:
    """Build a human-readable explanation from the scoring breakdown."""
    lines: list[str] = []

    # Matched aspects with their normalized sentiment scores
    aspect_lines: list[str] = []
    for key, value in breakdown.items():
        if key.startswith("aspect:"):
            aspect_name = key.split(":", 1)[1].replace("_", " ")
            aspect_lines.append(f"  matched: {aspect_name} ({value:.2f})")

    if aspect_lines:
        lines.append("Attribute matches:")
        lines.extend(
            sorted(aspect_lines, key=lambda l: -float(l.split("(")[1].rstrip(")")))
        )

    # Signal summary
    lines.append(
        f"Signals: vector similarity {breakdown['vector_similarity']:.2f}, "
        f"attribute match {breakdown['attribute_match']:.2f}, "
        f"sentiment strength {breakdown['sentiment_strength']:.2f}"
    )

    # Negation filters
    if parsed.negations:
        neg_str = ", ".join(n.replace("_", " ") for n in parsed.negations)
        lines.append(f"Excluded attributes confirmed absent: {neg_str}")

    return "\n".join(lines)
