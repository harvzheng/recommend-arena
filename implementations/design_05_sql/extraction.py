"""LLM-based attribute extraction for Design 05: SQL-First / SQLite + FTS5.

Uses the shared LLM provider to extract structured attributes from review text
during ingestion. Also extracts sentiment.
"""

from __future__ import annotations

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)

# Extraction prompt template per domain
EXTRACTION_PROMPTS = {
    "ski": """Extract structured attributes from this ski review.
Return JSON with these fields (omit if not mentioned):
- stiffness: integer 1-10 (1=soft, 10=race-stiff)
- damp: integer 1-10 (1=chattery, 10=extremely damp)
- edge_grip: integer 1-10 (1=no grip, 10=race-level grip)
- stability_at_speed: integer 1-10 (1=unstable, 10=rock solid)
- playfulness: integer 1-10 (1=serious/stiff, 10=very playful/fun)
- powder_float: integer 1-10 (1=sinks, 10=surfs on top)
- forgiveness: integer 1-10 (1=punishing, 10=very forgiving)
- terrain: list from [on-piste, off-piste, all-mountain, park, touring, race, carving, groomed, freeride, powder, big-mountain, backcountry]
- skill_level: list from [beginner, intermediate, advanced, expert]
- sentiment: float -1.0 to 1.0 (negative=dislikes, positive=likes)

Review: {review_text}

Return ONLY valid JSON, no other text.""",

    "running_shoe": """Extract structured attributes from this running shoe review.
Return JSON with these fields (omit if not mentioned):
- cushioning: integer 1-10 (1=minimal, 10=maximum cushion)
- responsiveness: integer 1-10 (1=dead, 10=extremely bouncy/propulsive)
- stability: integer 1-10 (1=wobbly, 10=very stable)
- grip: integer 1-10 (1=slippery, 10=incredible traction)
- breathability: integer 1-10 (1=hot, 10=very breathable)
- durability: integer 1-10 (1=falls apart quickly, 10=lasts forever)
- weight_feel: integer 1-10 (1=brick, 10=featherlight)
- surface: list from [road, trail, track, treadmill]
- sentiment: float -1.0 to 1.0 (negative=dislikes, positive=likes)

Review: {review_text}

Return ONLY valid JSON, no other text.""",
}


def extract_attributes_from_review(
    llm_provider,
    review_text: str,
    domain: str,
) -> dict:
    """Extract structured attributes from a single review using LLM.

    Returns a dict of extracted attribute name -> value.
    """
    prompt_template = EXTRACTION_PROMPTS.get(domain)
    if not prompt_template:
        # Generic fallback
        prompt_template = (
            "Extract structured attributes from this product review. "
            "Return JSON with a 'sentiment' field (float -1.0 to 1.0). "
            "Include any other attributes you can identify.\n\n"
            "Review: {review_text}\n\nReturn ONLY valid JSON."
        )

    prompt = prompt_template.format(review_text=review_text)

    try:
        response = llm_provider.generate(prompt, json_mode=True)
        # Strip markdown fences if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            response = "\n".join(lines)
        return json.loads(response)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract attributes: %s", e)
        return {"sentiment": 0.0}


def ingest_product_with_specs(
    db: sqlite3.Connection,
    product: dict,
    domain_id: int,
) -> int:
    """Insert a product and its spec-derived attributes.

    Returns the internal product_id.
    """
    external_id = product.get("id", product.get("product_id", ""))
    name = product.get("name", product.get("product_name", ""))
    brand = product.get("brand", "")
    category = product.get("category", "")
    raw_specs = json.dumps(product.get("specs", {}))

    # Upsert product
    db.execute(
        "INSERT OR IGNORE INTO products "
        "(domain_id, external_id, name, brand, category, raw_specs) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (domain_id, external_id, name, brand, category, raw_specs),
    )
    row = db.execute(
        "SELECT id FROM products WHERE domain_id = ? AND external_id = ?",
        (domain_id, external_id),
    ).fetchone()
    product_id = row[0]

    # Insert attributes from the product's attributes dict (ground truth)
    attributes = product.get("attributes", {})
    specs = product.get("specs", {})

    for attr_name, value in attributes.items():
        _upsert_attribute(db, product_id, domain_id, attr_name, value, "spec_sheet")

    # Also insert numeric specs as attributes
    spec_mappings = {
        "waist_width_mm": specs.get("waist_width_mm"),
        "turn_radius_m": specs.get("turn_radius_m"),
        "weight_g": specs.get("weight_g_per_ski", specs.get("weight_g")),
        "heel_drop_mm": specs.get("heel_drop_mm"),
        "stack_height_mm": specs.get("stack_height_mm"),
    }
    for attr_name, value in spec_mappings.items():
        if value is not None:
            _upsert_attribute(db, product_id, domain_id, attr_name, value, "spec_sheet")

    # Store rocker_profile and construction as text attributes
    if specs.get("rocker_profile"):
        _upsert_attribute(db, product_id, domain_id, "rocker_profile",
                          specs["rocker_profile"], "spec_sheet")
    if specs.get("construction"):
        _upsert_attribute(db, product_id, domain_id, "construction",
                          specs["construction"], "spec_sheet")

    # Store available lengths
    if specs.get("lengths_cm"):
        _upsert_attribute(db, product_id, domain_id, "lengths_available",
                          json.dumps(specs["lengths_cm"]), "spec_sheet")
    if specs.get("surface"):
        _upsert_attribute(db, product_id, domain_id, "surface",
                          specs["surface"], "spec_sheet")

    return product_id


def ingest_review(
    db: sqlite3.Connection,
    llm_provider,
    review: dict,
    domain: str,
    domain_id: int,
    product_id_map: dict[str, int],
    use_llm: bool = True,
) -> None:
    """Ingest a single review: store it and optionally extract attributes via LLM."""
    ext_product_id = review.get("product_id", "")
    internal_pid = product_id_map.get(ext_product_id)
    if internal_pid is None:
        logger.warning("Unknown product_id in review: %s", ext_product_id)
        return

    review_text = review.get("text", review.get("review_text", ""))
    author = review.get("reviewer", review.get("author", ""))
    rating = review.get("rating")
    source = review.get("source", "")

    # Default sentiment from rating if available
    sentiment = 0.0
    if rating is not None:
        # Map 1-5 rating to -1.0 to 1.0
        sentiment = (float(rating) - 3.0) / 2.0

    # Insert review
    db.execute(
        "INSERT INTO reviews (product_id, source, author, content, rating, sentiment) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (internal_pid, source, author, review_text, rating, sentiment),
    )
    review_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Insert into FTS index
    product_name = db.execute(
        "SELECT name FROM products WHERE id = ?", (internal_pid,)
    ).fetchone()[0]
    db.execute(
        "INSERT INTO reviews_fts(rowid, content, product_name) VALUES (?, ?, ?)",
        (review_id, review_text, product_name),
    )

    # Optionally extract attributes via LLM
    if use_llm and llm_provider is not None:
        try:
            extracted = extract_attributes_from_review(llm_provider, review_text, domain)
            # Update sentiment if LLM provided one
            if "sentiment" in extracted:
                db.execute(
                    "UPDATE reviews SET sentiment = ? WHERE id = ?",
                    (extracted["sentiment"], review_id),
                )
            # We don't override product attributes from reviews since we
            # already have ground-truth attributes from specs
        except Exception as e:
            logger.warning("LLM extraction failed for review: %s", e)


def _upsert_attribute(
    db: sqlite3.Connection,
    product_id: int,
    domain_id: int,
    attr_name: str,
    value,
    source: str,
) -> None:
    """Upsert a single product attribute."""
    # Get or skip if attribute_def doesn't exist
    row = db.execute(
        "SELECT id, data_type FROM attribute_defs WHERE domain_id = ? AND name = ?",
        (domain_id, attr_name),
    ).fetchone()
    if row is None:
        return

    attr_def_id, data_type = row

    value_numeric = None
    value_text = None

    if data_type in ("numeric", "scale"):
        try:
            value_numeric = float(value)
        except (TypeError, ValueError):
            value_text = str(value)
    elif data_type == "categorical":
        if isinstance(value, list):
            value_text = json.dumps(value)
        else:
            value_text = str(value)
    else:
        if isinstance(value, (list, dict)):
            value_text = json.dumps(value)
        else:
            value_text = str(value)

    db.execute(
        "INSERT OR REPLACE INTO product_attributes "
        "(product_id, attribute_def_id, value_numeric, value_text, confidence, source) "
        "VALUES (?, ?, ?, ?, 1.0, ?)",
        (product_id, attr_def_id, value_numeric, value_text, source),
    )
