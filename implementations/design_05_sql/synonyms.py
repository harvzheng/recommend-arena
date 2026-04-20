"""Synonym expansion for Design 05: SQL-First / SQLite + FTS5.

Built-in synonym tables per domain. Synonyms are used to expand FTS5
queries so that semantically related terms match.
"""

import sqlite3

# Built-in synonym definitions per domain
DOMAIN_SYNONYMS = {
    "ski": {
        "forgiving": ["easy to ski", "forgiving", "beginner-friendly", "user-friendly", "friendly",
                       "mellow", "approachable", "easy", "gentle", "tolerant"],
        "responsive": ["responsive", "quick edge to edge", "snappy", "lively", "alive",
                        "alive underfoot", "quick", "reactive", "energetic", "peppy"],
        "stable": ["stable", "composed", "planted", "confident", "confidence",
                    "confidence-inspiring", "locked in", "predictable", "trustworthy", "steady",
                    "solid", "assured", "reliable"],
        "playful": ["playful", "fun", "loose", "surfy", "buttery", "poppy",
                     "smeary", "skiddy", "whippy", "jibby", "nimble", "flickable",
                     "turny", "agile", "maneuverable"],
        "charger": ["charger", "go fast", "high speed", "aggressive", "charge",
                     "speed demon", "bomber", "ripping", "hauling", "straight-lining",
                     "full send", "sending it"],
        "stiff": ["stiff", "rigid", "firm", "demanding", "beefy", "powerful",
                   "muscular", "burly"],
        "carving": ["carving", "edge grip", "on-piste", "groomed", "hardpack", "trenches",
                     "rail", "railing", "laying trenches", "edge hold", "grippy",
                     "arcing", "arc", "carve"],
        "powder": ["powder", "float", "deep snow", "soft snow", "pow",
                    "deep days", "blower", "face shots", "waist deep", "chest deep",
                    "powder float", "surfy"],
        "versatile": ["versatile", "quiver of one", "all-mountain", "do it all", "mixed conditions",
                       "one-ski quiver", "one ski quiver", "quiver killer",
                       "everywhere", "jack of all trades", "daily driver", "everyday"],
        "lightweight": ["lightweight", "light", "feather", "ultralight", "touring weight"],
        "damp": ["damp", "smooth", "dampness", "vibration free", "composed",
                  "chatter free", "quiet", "settled", "plush", "absorbent",
                  "vibration absorption", "damping", "calm"],
        "ice": ["ice", "ice coast", "east coast", "hardpack", "bulletproof", "icy",
                 "firm snow", "boilerplate", "scraped off", "blue ice", "crud"],
        "freeride": ["freeride", "off-piste", "big mountain", "backcountry",
                      "steep", "steep and deep", "gnarly", "committing"],
        "edge_grip": ["edge grip", "edge hold", "grip on ice", "grip on hardpack",
                       "holds an edge", "bite", "rail", "carve", "ice coast",
                       "confidence-inspiring", "hardpack grip"],
        "powder_float": ["powder float", "float", "surfy", "surfing", "planes",
                          "stays on top", "flotation"],
        "stiffness_low": ["buttery", "noodly", "soft", "flexy", "forgiving",
                           "mellow", "easy flex"],
    },
    "running_shoe": {
        "cushioned": ["cushioned", "plush", "soft", "comfortable", "protective", "cloud",
                       "pillowy", "marshmallow", "padded", "luxurious", "sink in",
                       "well-cushioned", "max cushion", "thick"],
        "responsive": ["responsive", "bouncy", "springy", "snappy", "energy return", "propulsive",
                        "poppy", "lively", "peppy", "zippy", "dynamic", "toe-off",
                        "energy", "return"],
        "lightweight": ["lightweight", "light", "feather", "nimble", "barely there",
                         "disappears on foot", "ultralight", "airy", "featherweight"],
        "grippy": ["grippy", "grip", "traction", "aggressive lugs", "sticky",
                    "bite", "confident footing", "multi-surface", "wet grip",
                    "holds on", "sure-footed", "grips well"],
        "durable": ["durable", "long lasting", "high mileage", "rugged",
                     "hard wearing", "lasts", "built to last", "tank", "workhorse"],
        "stable": ["stable", "support", "secure", "planted", "locked in",
                    "supportive", "steady", "controlled", "guidance",
                    "confidence-inspiring", "reliable platform"],
        "fast": ["fast", "speed", "racing", "race day", "tempo", "interval",
                  "PR", "personal best", "quick", "speedy", "race"],
        "trail": ["trail", "off-road", "mountain", "technical terrain",
                   "rocks", "roots", "single track", "singletrack", "mud",
                   "dirt", "technical"],
        "marathon": ["marathon", "long distance", "long run", "endurance",
                      "ultra", "easy day", "recovery run", "daily trainer",
                      "mileage", "high mileage"],
        "comfort": ["comfort", "comfortable", "all-day", "plush", "cozy",
                     "smooth ride", "easy on feet", "forgiving"],
        "breathable": ["breathable", "airy", "cool", "ventilated", "mesh",
                        "well-ventilated", "doesn't overheat"],
    },
}

# Mapping of colloquial/vague phrases to structured attribute implications.
# Used to pre-expand queries before LLM parsing and to enrich BM25 search.
# Format: phrase -> list of (attribute_name, op, value) or keyword strings.
PHRASE_EXPANSIONS = {
    "ski": {
        "ice coast": [
            ("edge_grip", "gte", 7), ("damp", "gte", 6), ("stability_at_speed", "gte", 6),
        ],
        "east coast": [
            ("edge_grip", "gte", 7), ("damp", "gte", 6),
        ],
        "alive underfoot": [
            ("playfulness", "gte", 7), ("responsiveness", "gte", 7),
        ],
        "confidence-inspiring": [
            ("stability_at_speed", "gte", 7), ("damp", "gte", 6), ("edge_grip", "gte", 6),
        ],
        "confidence inspiring": [
            ("stability_at_speed", "gte", 7), ("damp", "gte", 6), ("edge_grip", "gte", 6),
        ],
        "charger": [
            ("stiffness", "gte", 7), ("damp", "gte", 7), ("stability_at_speed", "gte", 7),
        ],
        "surfy": [
            ("powder_float", "gte", 7), ("playfulness", "gte", 6),
        ],
        "buttery": [
            ("stiffness", "lte", 4), ("playfulness", "gte", 6),
        ],
        "forgiving": [
            ("stiffness", "lte", 5), ("forgiveness", "gte", 6),
        ],
        "poppy": [
            ("playfulness", "gte", 7),
        ],
        "one-ski quiver": [
            ("terrain", "contains", "all-mountain"),
        ],
        "one ski quiver": [
            ("terrain", "contains", "all-mountain"),
        ],
        "quiver of one": [
            ("terrain", "contains", "all-mountain"),
        ],
        "daily driver": [
            ("terrain", "contains", "all-mountain"), ("forgiveness", "gte", 5),
        ],
        "bomber": [
            ("stiffness", "gte", 7), ("stability_at_speed", "gte", 8), ("damp", "gte", 7),
        ],
        "nimble": [
            ("playfulness", "gte", 6), ("stiffness", "lte", 6),
        ],
        "damp": [
            ("damp", "gte", 7),
        ],
        "smooth": [
            ("damp", "gte", 6),
        ],
    },
    "running_shoe": {
        "bouncy": [
            ("responsiveness", "gte", 7), ("cushioning", "gte", 6),
        ],
        "plush": [
            ("cushioning", "gte", 7),
        ],
        "nimble": [
            ("responsiveness", "gte", 6), ("weight_feel", "gte", 7),
        ],
        "cloud-like": [
            ("cushioning", "gte", 8),
        ],
        "tank": [
            ("durability", "gte", 8),
        ],
        "workhorse": [
            ("durability", "gte", 7), ("cushioning", "gte", 6),
        ],
        "race day": [
            ("responsiveness", "gte", 7), ("weight_feel", "gte", 7),
        ],
        "daily trainer": [
            ("cushioning", "gte", 6), ("durability", "gte", 6),
        ],
        "recovery run": [
            ("cushioning", "gte", 7),
        ],
        "speed work": [
            ("responsiveness", "gte", 7), ("weight_feel", "gte", 7),
        ],
    },
}


def seed_synonyms(db: sqlite3.Connection, domain: str, domain_id: int) -> None:
    """Seed the synonyms table with built-in synonyms for a domain."""
    syns = DOMAIN_SYNONYMS.get(domain, {})
    for canonical, variants in syns.items():
        for variant in variants:
            db.execute(
                "INSERT OR IGNORE INTO synonyms (domain_id, canonical, variant) "
                "VALUES (?, ?, ?)",
                (domain_id, canonical, variant),
            )
    db.commit()


def pre_expand_query(query_text: str, domain: str) -> tuple[str, list[dict]]:
    """Pre-expand a query using phrase expansions before LLM parsing.

    Returns:
        - expanded_query: the original query with appended expansion hints
        - extra_filters: list of {attribute, op, value} dicts derived from phrases
    """
    phrase_map = PHRASE_EXPANSIONS.get(domain, {})
    if not phrase_map:
        return query_text, []

    query_lower = query_text.lower()
    extra_filters = []
    expansion_terms = []

    for phrase, expansions in phrase_map.items():
        if phrase in query_lower:
            for exp in expansions:
                attr, op, val = exp
                extra_filters.append({"attribute": attr, "op": op, "value": val})
                expansion_terms.append(attr.replace("_", " "))

    if expansion_terms:
        # Deduplicate
        unique_terms = list(dict.fromkeys(expansion_terms))
        hint = " [expansion hints: " + ", ".join(unique_terms) + "]"
        return query_text + hint, extra_filters
    return query_text, extra_filters


def expand_synonyms(db: sqlite3.Connection, keywords: str, domain_id: int) -> str:
    """Expand query keywords using the synonym table.

    For each keyword that has synonyms, builds an FTS5 OR query so
    that any synonym variant matches. Terms without synonyms are
    passed through unchanged.
    """
    if not keywords:
        return keywords
    if isinstance(keywords, list):
        keywords = " ".join(str(k) for k in keywords)
    if not keywords.strip():
        return keywords

    tokens = keywords.lower().split()
    expanded_parts = []

    for token in tokens:
        # Look up synonyms for this token (as canonical or as a variant)
        rows = db.execute(
            "SELECT DISTINCT variant FROM synonyms "
            "WHERE domain_id = ? AND canonical = ?",
            (domain_id, token),
        ).fetchall()

        if rows:
            alternatives = " OR ".join(f'"{r[0]}"' for r in rows)
            expanded_parts.append(f"({alternatives})")
        else:
            # Also check if this token is a variant, and expand via its canonical
            canonical_row = db.execute(
                "SELECT DISTINCT s2.variant FROM synonyms s1 "
                "JOIN synonyms s2 ON s1.domain_id = s2.domain_id AND s1.canonical = s2.canonical "
                "WHERE s1.domain_id = ? AND s1.variant = ?",
                (domain_id, token),
            ).fetchall()
            if canonical_row:
                alternatives = " OR ".join(f'"{r[0]}"' for r in canonical_row)
                expanded_parts.append(f"({alternatives})")
            else:
                expanded_parts.append(token)

    return " ".join(expanded_parts)
