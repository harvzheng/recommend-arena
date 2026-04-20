"""Main Recommender implementation for Design #7: Bayesian / Probabilistic Scoring.

Implements the shared Recommender protocol. Uses LLM for attribute extraction
from reviews and query parsing. All scoring is pure math on conjugate-prior
posterior distributions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

# Add shared module to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.llm_provider import get_provider, LLMProvider  # noqa: E402
from shared.interface import RecommendationResult  # noqa: E402

from .beliefs import BeliefStore, ProductBelief, init_posteriors  # noqa: E402
from .schema import (  # noqa: E402
    AttributeSpec,
    get_schema,
    get_spec,
    get_category_priors,
    DOMAIN_SCHEMAS,
)
from .scorer import (  # noqa: E402
    AttributeConstraint,
    build_explanation,
    normalize_scores,
    rank_products,
    score_product,
    text_search_fallback,
)
from .updater import update_posterior  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain synonym dictionary (Enhancement #3)
# Colloquial terms -> structured attribute hints appended to query
# ---------------------------------------------------------------------------

QUERY_SYNONYMS: dict[str, list[str]] = {
    # Ski domain colloquial terms
    "ice coast": ["strong edge_grip", "high dampness", "stiff", "on-piste"],
    "east coast": ["strong edge_grip", "high dampness", "stiff"],
    "alive underfoot": ["high playfulness", "high responsiveness"],
    "lively": ["high playfulness", "high responsiveness"],
    "poppy": ["high playfulness"],
    "confidence-inspiring": ["high stability", "high dampness", "stiff"],
    "confidence inspiring": ["high stability", "high dampness", "stiff"],
    "charger": ["very_stiff stiffness", "high dampness", "exceptional edge_grip"],
    "charging": ["very_stiff stiffness", "high dampness"],
    "forgiving": ["soft stiffness", "high stability", "moderate dampness"],
    "beginner-friendly": ["soft stiffness", "beginner skill_level", "high stability"],
    "pow": ["excellent powder_float", "freeride terrain", "powder terrain"],
    "deep snow": ["excellent powder_float", "powder terrain"],
    "groomers": ["on-piste terrain", "strong edge_grip"],
    "groomed": ["on-piste terrain", "strong edge_grip"],
    "hard pack": ["exceptional edge_grip", "high dampness", "on-piste"],
    "hardpack": ["exceptional edge_grip", "high dampness"],
    "crud": ["high dampness", "all-mountain terrain"],
    "chop": ["high dampness", "all-mountain terrain"],
    "choppy": ["high dampness", "all-mountain terrain"],
    "buttery": ["soft stiffness", "high playfulness"],
    "damp": ["high dampness"],
    "chattery": ["low dampness"],
    "hooky": ["exceptional edge_grip"],
    "surfy": ["high playfulness", "excellent powder_float", "freeride"],
    "stable at speed": ["very_stiff stiffness", "high dampness"],
    "fast": ["stiff stiffness", "high dampness", "strong edge_grip"],
    "nimble": ["light weight_feel", "high playfulness"],
    "heavy": ["heavy weight_feel", "high dampness"],
    "lightweight": ["ultralight weight_feel", "light weight_feel"],
    "light": ["light weight_feel"],
    "touring": ["backcountry terrain", "ultralight weight_feel"],
    "sidecountry": ["freeride terrain", "good powder_float"],
    "slackcountry": ["freeride terrain", "moderate powder_float"],
    "all mountain": ["all-mountain terrain"],
    "all-mountain": ["all-mountain terrain"],
    "frontside": ["on-piste terrain", "strong edge_grip"],
    "big mountain": ["freeride terrain", "very_stiff stiffness", "high dampness"],

    # Running shoe colloquial terms
    "bouncy": ["high responsiveness", "high cushioning"],
    "plush": ["maximal cushioning", "high cushioning"],
    "soft landing": ["high cushioning", "maximal cushioning"],
    "snappy": ["very_high responsiveness", "high responsiveness"],
    "rocket": ["very_high responsiveness", "racing use_case"],
    "daily driver": ["daily_training use_case", "high durability"],
    "workhorse": ["daily_training use_case", "very_high durability"],
    "tempo": ["speed_work use_case", "high responsiveness"],
    "long run": ["long_run use_case", "high cushioning"],
    "easy day": ["recovery use_case", "high cushioning"],
    "recovery run": ["recovery use_case", "maximal cushioning"],
    "road runner": ["road surface"],
    "trail runner": ["trail surface", "aggressive grip"],
    "technical terrain": ["trail surface", "aggressive grip", "high stability"],
    "muddy": ["aggressive grip", "trail surface"],
    "rocky": ["high grip", "trail surface", "high durability"],
    "speed demon": ["racing use_case", "very_high responsiveness"],
    "cushioned": ["high cushioning", "maximal cushioning"],
    "firm": ["minimal cushioning", "moderate cushioning"],
    "zero drop": ["heel_drop_mm 0"],
    "low drop": ["heel_drop_mm 4"],
    "high drop": ["heel_drop_mm 12"],
    "breathable": ["very_high breathability"],
    "supportive": ["high_support stability", "moderate_support stability"],
    "neutral": ["neutral stability"],
    "overpronation": ["high_support stability"],
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _expand_query_synonyms(query_text: str) -> str:
    """Expand colloquial terms in a query using the synonym dictionary."""
    query_lower = query_text.lower()
    expansions: list[str] = []

    # Sort by length descending so longer phrases match first
    for phrase, hints in sorted(QUERY_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in query_lower:
            expansions.extend(hints)

    if expansions:
        # Append expansion hints to the original query for LLM context
        return query_text + "\n\n[Attribute hints: " + ", ".join(expansions) + "]"
    return query_text


class BayesianRecommender:
    """Bayesian probabilistic scoring recommender.

    Products are represented as posterior distributions over structured
    attributes. Reviews update these distributions via conjugate Bayesian
    updates. Queries are scored by computing P(match | posteriors).
    """

    def __init__(self) -> None:
        self.store = BeliefStore()
        self._llm: LLMProvider | None = None
        self._ingested_domains: set[str] = set()

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_provider()
        return self._llm

    # ------------------------------------------------------------------
    # ingest()
    # ------------------------------------------------------------------

    def ingest(
        self, products: list[dict], reviews: list[dict], domain: str
    ) -> None:
        """Ingest product and review data for a domain.

        1. Initialize product beliefs from schema priors (with category priors).
        2. Batch all reviews per product, extract attributes via 1 LLM call per product.
        3. Apply Bayesian updates to posterior distributions.
        4. Create embeddings for each product.
        5. Persist beliefs and reviews to SQLite.
        """
        schema = get_schema(domain)

        # Clear any prior ingestion for this domain
        self.store.clear_domain(domain)

        # Step 1: Initialize beliefs for all products
        beliefs: dict[str, ProductBelief] = {}
        product_map: dict[str, dict] = {}  # pid -> product dict
        for p in products:
            pid = p.get("id") or p.get("product_id", "")
            pname = p.get("name") or p.get("product_name", "")
            product_map[pid] = p

            belief = ProductBelief(
                product_id=pid,
                domain=domain,
                name=pname,
                posteriors=init_posteriors(schema),
                evidence_count=0,
            )

            # Enhancement #4: Apply category-based informative priors
            category = (p.get("category", "") or "").strip()
            cat_priors = get_category_priors(category) if category else None
            if cat_priors:
                self._apply_category_priors(belief, cat_priors, schema)

            beliefs[pid] = belief

            # If the product has spec metadata, use it as strong evidence
            self._ingest_product_specs(belief, p, schema)

        # Step 2 & 3: Batch reviews per product, 1 LLM call per product
        reviews_by_product: dict[str, list[dict]] = {}
        for r in reviews:
            pid = r.get("product_id", "")
            reviews_by_product.setdefault(pid, []).append(r)

        for pid, product_reviews in reviews_by_product.items():
            belief = beliefs.get(pid)
            if belief is None:
                continue

            # Collect all review texts for this product
            review_texts: list[str] = []
            review_ids: list[str] = []
            for review in product_reviews:
                review_text = review.get("text") or review.get("review_text", "")
                review_id = review.get("review_id") or review.get(
                    "id",
                    hashlib.md5(
                        f"{pid}:{review_text[:50]}".encode()
                    ).hexdigest(),
                )
                review_texts.append(review_text)
                review_ids.append(review_id)

            # Enhancement #1: Single batched LLM call for all reviews of this product
            non_empty = [t for t in review_texts if t.strip()]
            if non_empty:
                all_extractions = self._extract_attributes_batch(
                    belief.name, non_empty, schema
                )

                # Apply each extracted observation as a Bayesian update
                for extracted in all_extractions:
                    update_posterior(belief, extracted, schema)

            # Store each review individually (with empty extraction as placeholder)
            for review_id, review_text in zip(review_ids, review_texts):
                self.store.store_review(
                    review_id=review_id,
                    product_id=pid,
                    raw_text=review_text,
                    extracted={},  # extractions were batched
                )

        # Step 4: Create embeddings for each product
        for pid, belief in beliefs.items():
            p = product_map.get(pid, {})
            try:
                embedding_text = self._build_embedding_text(belief, p, schema)
                belief.embedding = self.llm.embed(embedding_text)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", pid, e)
                belief.embedding = None

        # Step 5: Persist all beliefs
        for belief in beliefs.values():
            self.store.upsert_belief(belief)

        self._ingested_domains.add(domain)
        logger.info(
            "Ingested %d products and %d reviews for domain %r",
            len(products), len(reviews), domain,
        )

    def _build_embedding_text(
        self, belief: ProductBelief, product: dict, schema: list[AttributeSpec]
    ) -> str:
        """Build rich text for embedding from product info + posterior summaries."""
        parts: list[str] = [belief.name]

        category = product.get("category", "")
        if category:
            parts.append(f"Category: {category}")

        # Summarize top posterior beliefs
        for spec in schema:
            posterior = belief.posteriors.get(spec.name)
            if posterior is None:
                continue
            if spec.attr_type in ("ordinal", "categorical") and spec.levels:
                alpha = posterior["alpha"]
                alpha_sum = sum(alpha)
                if alpha_sum > len(alpha) + 1:  # Has some evidence
                    best_idx = max(range(len(alpha)), key=lambda i: alpha[i])
                    p = alpha[best_idx] / alpha_sum
                    if p > 0.25:  # Only include if somewhat confident
                        parts.append(f"{spec.name}: {spec.levels[best_idx]}")
            elif spec.attr_type == "continuous":
                parts.append(f"{spec.name}: {posterior['mu']:.0f}")

        # Add review snippets
        review_texts = self.store.get_review_texts(belief.product_id)
        if review_texts:
            combined = " ".join(review_texts)
            parts.append(f"Reviews: {combined[:500]}")

        return ". ".join(parts)

    def _apply_category_priors(
        self, belief: ProductBelief, cat_priors: dict, schema: list[AttributeSpec]
    ) -> None:
        """Blend category-based informative priors into the belief's posteriors."""
        for attr_name, prior_overrides in cat_priors.items():
            posterior = belief.posteriors.get(attr_name)
            if posterior is None:
                continue

            if "alpha" in prior_overrides and "alpha" in posterior:
                # Replace the uniform prior with the informative one
                new_alpha = prior_overrides["alpha"]
                if len(new_alpha) == len(posterior["alpha"]):
                    posterior["alpha"] = list(new_alpha)

            if "mu" in prior_overrides and "mu" in posterior:
                posterior["mu"] = prior_overrides["mu"]
            if "sigma" in prior_overrides and "sigma" in posterior:
                posterior["sigma"] = prior_overrides["sigma"]

    def _ingest_product_specs(
        self,
        belief: ProductBelief,
        product: dict,
        schema: list[AttributeSpec],
    ) -> None:
        """Use product spec metadata to set strong priors.

        Product specs (e.g., waist_width_mm=100) are treated as highly
        reliable observations, equivalent to several review mentions.
        """
        specs = product.get("specs", {})
        attributes = product.get("attributes", {})

        # Map known spec fields to schema attributes
        spec_mappings = self._build_spec_observation(specs, attributes, schema)

        # Apply as multiple observations (specs are reliable)
        for _ in range(3):  # Weight specs as 3 observations
            update_posterior(belief, spec_mappings, schema)

    def _build_spec_observation(
        self,
        specs: dict,
        attributes: dict,
        schema: list[AttributeSpec],
    ) -> dict[str, str]:
        """Convert raw product specs/attributes into schema-mapped observations."""
        obs: dict[str, str] = {}

        for spec_attr in schema:
            attr_name = spec_attr.name

            # Direct continuous spec match
            if spec_attr.attr_type == "continuous":
                if attr_name in specs:
                    obs[attr_name] = str(specs[attr_name])
                elif attr_name.replace("_mm", "") in specs:
                    obs[attr_name] = str(specs[attr_name.replace("_mm", "")])
                # Check common field aliases
                for alias, target in [
                    ("waist_width_mm", "waist_width_mm"),
                    ("weight_g", "weight_g"),
                    ("heel_drop_mm", "heel_drop_mm"),
                    ("weight_g_per_ski", "weight_g"),
                ]:
                    if alias in specs and target == attr_name:
                        obs[attr_name] = str(specs[alias])

            # Map numeric attributes to ordinal levels
            elif spec_attr.attr_type in ("ordinal", "categorical"):
                if spec_attr.levels is None:
                    continue

                # Check for the attribute under its name and common aliases
                _ATTR_ALIASES = {
                    "dampness": ["damp", "dampness"],
                    "edge_grip": ["edge_grip", "edgeGrip"],
                    "powder_float": ["powder_float", "powderFloat"],
                    "weight_feel": ["weight_feel", "weightFeel"],
                    "skill_level": ["skill_level", "ability_level"],
                }
                lookup_keys = _ATTR_ALIASES.get(attr_name, [attr_name])
                val = None
                for lk in lookup_keys:
                    if lk in attributes:
                        val = attributes[lk]
                        break

                if val is not None:
                    mapped = self._map_numeric_to_level(
                        val, attr_name, spec_attr
                    )
                    if mapped:
                        obs[attr_name] = mapped

                # Handle terrain specially -- it can be a list
                if attr_name == "terrain" and "terrain" in attributes:
                    terrain_val = attributes["terrain"]
                    if isinstance(terrain_val, list) and terrain_val:
                        # Use first terrain as primary
                        primary = terrain_val[0]
                        if primary in spec_attr.levels:
                            obs[attr_name] = primary

        return obs

    def _map_numeric_to_level(
        self,
        value: int | float | str,
        attr_name: str,
        spec: AttributeSpec,
    ) -> str | None:
        """Map a numeric attribute value (1-10 scale) to ordinal levels."""
        if isinstance(value, str):
            # Already a string level
            if value in (spec.levels or []):
                return value
            return None

        if not isinstance(value, (int, float)):
            return None

        n_levels = len(spec.levels or [])
        if n_levels == 0:
            return None

        # Map 1-10 scale to level indices
        # value=1 -> first level, value=10 -> last level
        idx = min(n_levels - 1, max(0, int((value - 1) / 10 * n_levels)))
        return spec.levels[idx]

    # ------------------------------------------------------------------
    # query()
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
        *,
        ranking_mode: str = "expected",
        k: float = 1.0,
    ) -> list[RecommendationResult]:
        """Query the recommender with natural language.

        1. Parse query into attribute constraints via LLM.
        2. Score all products against constraints.
        3. Apply text-search fallback for unmapped terms.
        4. Blend embedding similarity signal.
        5. Normalize, rank, and return top-k results.
        """
        schema = get_schema(domain)
        beliefs = self.store.get_all_beliefs(domain)

        if not beliefs:
            return []

        # Step 1: Parse query
        constraints, unmapped_terms = self._parse_query(query_text, schema)

        # Step 2: Score all products
        scored: list[tuple[str, float, float]] = []
        belief_map: dict[str, ProductBelief] = {}
        for belief in beliefs:
            s, unc = score_product(belief, constraints, schema)
            scored.append((belief.product_id, s, unc))
            belief_map[belief.product_id] = belief

        # Step 3: Unmapped-term fallback
        if unmapped_terms:
            product_ids = [b.product_id for b in beliefs]
            review_texts = {
                pid: self.store.get_review_texts(pid) for pid in product_ids
            }
            fallback_scores = text_search_fallback(
                unmapped_terms, product_ids, review_texts
            )
            total_terms = len(constraints) + len(unmapped_terms)
            w_fallback = len(unmapped_terms) / total_terms if total_terms > 0 else 0.0
            w_fallback = min(w_fallback, 0.5)

            if w_fallback > 0:
                scored = [
                    (
                        pid,
                        (1 - w_fallback) * raw + w_fallback * fallback_scores.get(pid, 0.0),
                        unc,
                    )
                    for pid, raw, unc in scored
                ]

        # Step 4: Enhancement #2 — Embedding similarity blending
        scored = self._blend_embedding_scores(
            query_text, scored, belief_map, constraints
        )

        # Step 5: Normalize and rank
        scored = normalize_scores(scored)
        ranked = rank_products(scored, mode=ranking_mode, k=k)

        # Build results
        results: list[RecommendationResult] = []
        for pid, norm_score, unc in ranked[:top_k]:
            belief = belief_map[pid]
            explanation, matched_attrs = build_explanation(
                belief, constraints, schema
            )
            results.append(
                RecommendationResult(
                    product_id=pid,
                    product_name=belief.name,
                    score=round(norm_score, 4),
                    explanation=explanation,
                    matched_attributes=matched_attrs,
                )
            )

        return results

    def _blend_embedding_scores(
        self,
        query_text: str,
        scored: list[tuple[str, float, float]],
        belief_map: dict[str, ProductBelief],
        constraints: list[AttributeConstraint],
    ) -> list[tuple[str, float, float]]:
        """Blend embedding cosine similarity into probabilistic scores.

        The embedding weight is higher when:
        - Few structured constraints were parsed (vague query)
        - Product has low evidence count (high posterior uncertainty)
        """
        # Check if any products have embeddings
        has_embeddings = any(
            belief_map[pid].embedding is not None
            for pid, _, _ in scored
        )
        if not has_embeddings:
            return scored

        # Embed the query
        try:
            query_embedding = self.llm.embed(query_text)
        except Exception as e:
            logger.warning("Query embedding failed: %s", e)
            return scored

        # Compute cosine similarities
        similarities: dict[str, float] = {}
        for pid, _, _ in scored:
            emb = belief_map[pid].embedding
            if emb is not None:
                similarities[pid] = _cosine_similarity(query_embedding, emb)
            else:
                similarities[pid] = 0.0

        # Normalize similarities to [0, 1]
        sim_values = list(similarities.values())
        sim_min, sim_max = min(sim_values), max(sim_values)
        if sim_max - sim_min > 1e-12:
            similarities = {
                pid: (s - sim_min) / (sim_max - sim_min)
                for pid, s in similarities.items()
            }

        # Determine embedding weight based on constraint coverage
        # Fewer constraints = more reliance on embedding
        n_constraints = len(constraints)
        if n_constraints == 0:
            w_embed = 0.6  # Very vague query — embedding dominant
        elif n_constraints <= 2:
            w_embed = 0.35  # Some structure but still lean on embedding
        elif n_constraints <= 4:
            w_embed = 0.2  # Good structure, embedding as tiebreaker
        else:
            w_embed = 0.1  # Rich structure, minimal embedding influence

        # Blend per-product, adjusting for evidence level
        blended: list[tuple[str, float, float]] = []
        for pid, raw_score, unc in scored:
            evidence = belief_map[pid].evidence_count
            # Low evidence -> boost embedding weight
            evidence_factor = 1.0 if evidence < 5 else (0.7 if evidence < 15 else 0.5)
            effective_w = w_embed * evidence_factor

            new_score = (1 - effective_w) * raw_score + effective_w * similarities.get(pid, 0.0)
            blended.append((pid, new_score, unc))

        return blended

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    def _extract_attributes_batch(
        self,
        product_name: str,
        review_texts: list[str],
        schema: list[AttributeSpec],
    ) -> list[dict[str, str]]:
        """Extract attribute observations from ALL reviews for a product in one LLM call.

        Returns a list of observation dicts (one per review) suitable for
        sequential Bayesian updates.
        """
        schema_desc = self._schema_description(schema)

        # Build numbered review block
        review_block = "\n\n".join(
            f"Review {i+1}: {text}" for i, text in enumerate(review_texts) if text.strip()
        )
        n_reviews = len([t for t in review_texts if t.strip()])

        prompt = f"""Extract product attributes from these {n_reviews} reviews of "{product_name}".

For EACH review, return the attributes mentioned. Return a JSON object with a "reviews" array where each element is an object mapping attribute names to observed values.

SCHEMA (use ONLY these attribute names and their allowed values):
{schema_desc}

RULES:
- Only include attributes that are clearly mentioned or strongly implied in each review.
- For ordinal/categorical attributes, use EXACTLY one of the listed level values.
- For continuous attributes, extract the numeric value.
- If an attribute is not mentioned in a review, do NOT include it for that review.
- Return valid JSON only, no extra text.

{review_block}

JSON (format: {{"reviews": [{{...}}, {{...}}, ...]}}):"""

        try:
            response = self.llm.generate(prompt, json_mode=True)
            parsed = json.loads(response)
            valid_names = {s.name for s in schema}

            reviews_data = parsed.get("reviews", [])
            if not isinstance(reviews_data, list):
                # Maybe it returned a flat dict -- treat as single observation
                reviews_data = [parsed]

            results: list[dict[str, str]] = []
            for item in reviews_data:
                if isinstance(item, dict):
                    obs = {k: str(v) for k, v in item.items()
                           if k in valid_names}
                    if obs:
                        results.append(obs)

            # If LLM returned fewer items than reviews, that's OK.
            # If it returned a single summary, apply it once.
            return results if results else []

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning("LLM batch extraction failed for %s: %s", product_name, e)
            return []

    def _parse_query(
        self, query_text: str, schema: list[AttributeSpec]
    ) -> tuple[list[AttributeConstraint], list[str]]:
        """Parse a natural-language query into attribute constraints.

        Enhancement #3: Expands colloquial synonyms before LLM parsing.
        Enhancement #5: Uses few-shot examples for better parsing.
        """
        schema_desc = self._schema_description(schema)

        # Enhancement #3: Expand synonyms
        expanded_query = _expand_query_synonyms(query_text)

        # Enhancement #5: Few-shot examples
        few_shot = self._get_few_shot_examples(schema)

        prompt = f"""Parse this product search query into structured attribute constraints.

SCHEMA (use ONLY these attribute names and their allowed values):
{schema_desc}

RULES:
- Map the query to specific attributes and values from the schema.
- For each constraint, specify the attribute name, value, and a strength (0.0-1.0) indicating how important this constraint is in the query.
- Be aggressive about mapping terms — most queries relate to specific attributes even if they use informal language.
- If a query term truly cannot be mapped to any schema attribute, list it under "unmapped".
- Return valid JSON only.

{few_shot}

FORMAT:
{{"mapped": [{{"attribute": "name", "value": "level", "strength": 0.9}}], "unmapped": ["term1"]}}

QUERY: "{expanded_query}"

JSON:"""

        try:
            response = self.llm.generate(prompt, json_mode=True)
            parsed = json.loads(response)

            constraints: list[AttributeConstraint] = []
            valid_names = {s.name for s in schema}

            mapped = parsed.get("mapped", parsed.get("constraints", []))
            if isinstance(mapped, list):
                for item in mapped:
                    attr = item.get("attribute", "")
                    if attr in valid_names:
                        constraints.append(
                            AttributeConstraint(
                                attribute=attr,
                                value=str(item.get("value", "")),
                                strength=float(item.get("strength", 0.8)),
                            )
                        )

            unmapped = parsed.get("unmapped", [])
            if not isinstance(unmapped, list):
                unmapped = []

            # Also run deterministic synonym-based constraint extraction
            # to catch things the LLM might miss
            syn_constraints = self._synonym_constraints(query_text, schema)
            existing_attrs = {c.attribute for c in constraints}
            for sc in syn_constraints:
                if sc.attribute not in existing_attrs:
                    constraints.append(sc)
                    existing_attrs.add(sc.attribute)

            return constraints, unmapped

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning("LLM query parse failed: %s — using fallback", e)
            return self._fallback_parse(query_text, schema)

    def _synonym_constraints(
        self, query_text: str, schema: list[AttributeSpec]
    ) -> list[AttributeConstraint]:
        """Deterministic constraint extraction from synonym dictionary."""
        query_lower = query_text.lower()
        constraints: list[AttributeConstraint] = []
        valid_names = {s.name for s in schema}

        for phrase, hints in sorted(QUERY_SYNONYMS.items(), key=lambda x: -len(x[0])):
            if phrase in query_lower:
                for hint in hints:
                    parts = hint.split()
                    if len(parts) == 2:
                        val, attr = parts[0], parts[1]
                        if attr in valid_names:
                            constraints.append(
                                AttributeConstraint(
                                    attribute=attr,
                                    value=val,
                                    strength=0.7,
                                )
                            )
                    elif len(parts) == 1:
                        # Single word -- might be a level or attribute name
                        # Check if it's a valid level for any schema attribute
                        for spec in schema:
                            if spec.levels and parts[0] in spec.levels:
                                constraints.append(
                                    AttributeConstraint(
                                        attribute=spec.name,
                                        value=parts[0],
                                        strength=0.6,
                                    )
                                )

        return constraints

    def _get_few_shot_examples(self, schema: list[AttributeSpec]) -> str:
        """Return domain-appropriate few-shot examples for query parsing."""
        # Detect domain by checking for domain-specific attributes
        attr_names = {s.name for s in schema}

        if "stiffness" in attr_names:
            # Ski domain
            return """EXAMPLES:
Query: "I want something for charging hard on groomers"
Answer: {"mapped": [{"attribute": "stiffness", "value": "very_stiff", "strength": 0.9}, {"attribute": "terrain", "value": "on-piste", "strength": 0.9}, {"attribute": "edge_grip", "value": "exceptional", "strength": 0.8}, {"attribute": "dampness", "value": "high", "strength": 0.8}], "unmapped": []}

Query: "playful ski that floats in powder but can handle the whole mountain"
Answer: {"mapped": [{"attribute": "playfulness", "value": "high", "strength": 0.9}, {"attribute": "powder_float", "value": "good", "strength": 0.8}, {"attribute": "terrain", "value": "all-mountain", "strength": 0.7}], "unmapped": []}

Query: "forgiving ski for improving intermediates, not too heavy"
Answer: {"mapped": [{"attribute": "stiffness", "value": "soft", "strength": 0.8}, {"attribute": "skill_level", "value": "intermediate", "strength": 0.9}, {"attribute": "weight_feel", "value": "light", "strength": 0.7}], "unmapped": []}

Query: "ice coast ripper"
Answer: {"mapped": [{"attribute": "edge_grip", "value": "exceptional", "strength": 0.9}, {"attribute": "dampness", "value": "high", "strength": 0.8}, {"attribute": "stiffness", "value": "stiff", "strength": 0.7}, {"attribute": "terrain", "value": "on-piste", "strength": 0.7}], "unmapped": []}
"""
        elif "cushioning" in attr_names:
            # Running shoe domain
            return """EXAMPLES:
Query: "bouncy daily trainer that lasts"
Answer: {"mapped": [{"attribute": "responsiveness", "value": "high", "strength": 0.8}, {"attribute": "cushioning", "value": "high", "strength": 0.7}, {"attribute": "use_case", "value": "daily_training", "strength": 0.9}, {"attribute": "durability", "value": "high", "strength": 0.8}], "unmapped": []}

Query: "lightweight racer for 5k and 10k"
Answer: {"mapped": [{"attribute": "weight_feel", "value": "ultralight", "strength": 0.9}, {"attribute": "use_case", "value": "racing", "strength": 0.9}, {"attribute": "responsiveness", "value": "very_high", "strength": 0.8}], "unmapped": []}

Query: "plush recovery shoe with lots of cushion"
Answer: {"mapped": [{"attribute": "cushioning", "value": "maximal", "strength": 0.9}, {"attribute": "use_case", "value": "recovery", "strength": 0.8}], "unmapped": []}

Query: "trail shoe for rocky technical terrain"
Answer: {"mapped": [{"attribute": "surface", "value": "trail", "strength": 0.9}, {"attribute": "grip", "value": "aggressive", "strength": 0.8}, {"attribute": "stability", "value": "moderate_support", "strength": 0.7}, {"attribute": "durability", "value": "high", "strength": 0.7}], "unmapped": []}
"""
        else:
            return ""

    def _fallback_parse(
        self, query_text: str, schema: list[AttributeSpec]
    ) -> tuple[list[AttributeConstraint], list[str]]:
        """Simple keyword-based fallback when LLM parsing fails."""
        constraints: list[AttributeConstraint] = []
        query_lower = query_text.lower()
        matched_terms: set[str] = set()

        for spec in schema:
            if spec.levels is None:
                continue
            for level in spec.levels:
                level_lower = level.lower().replace("_", " ").replace("-", " ")
                if level_lower in query_lower:
                    constraints.append(
                        AttributeConstraint(
                            attribute=spec.name,
                            value=level,
                            strength=0.7,
                        )
                    )
                    matched_terms.add(level_lower)

        # Identify unmapped terms
        words = set(re.findall(r'\b[a-z]+\b', query_lower))
        stop_words = {
            "a", "an", "the", "for", "and", "or", "that", "this", "is",
            "in", "on", "with", "to", "of", "i", "want", "need", "looking",
            "ski", "shoe", "shoes", "something", "very", "really", "good",
            "great", "best", "like", "me",
        }
        unmapped = [
            w for w in words
            if w not in stop_words and w not in matched_terms
        ]

        return constraints, unmapped

    def _schema_description(self, schema: list[AttributeSpec]) -> str:
        """Build a readable schema description for LLM prompts."""
        lines: list[str] = []
        for s in schema:
            if s.attr_type in ("ordinal", "categorical"):
                levels_str = ", ".join(s.levels or [])
                lines.append(
                    f'- {s.name} ({s.attr_type}): [{levels_str}]'
                )
            else:
                lines.append(
                    f'- {s.name} ({s.attr_type}): numeric value'
                )
        return "\n".join(lines)
