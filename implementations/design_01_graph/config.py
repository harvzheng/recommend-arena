"""Domain configuration and ontology for the graph-based recommender."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Domain synonym dictionary — maps colloquial phrases to attribute expansions
# Each entry: phrase -> list of (attribute, polarity, weight) tuples
# ---------------------------------------------------------------------------

DOMAIN_SYNONYMS: dict[str, dict[str, list[tuple[str, float, float]]]] = {
    "ski": {
        "ice coast": [("edge_grip", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "ice": [("edge_grip", 1.0, 1.5)],
        "bulletproof": [("edge_grip", 1.0, 1.5), ("damp", 1.0, 1.0), ("stability_at_speed", 1.0, 1.0)],
        "alive underfoot": [("playfulness", 1.0, 1.5), ("responsiveness", 1.0, 1.0)],
        "lively feel": [("playfulness", 1.0, 1.5), ("responsiveness", 1.0, 1.0)],
        "confidence-inspiring": [("stability_at_speed", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "confidence inspiring": [("stability_at_speed", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "inspires confidence": [("stability_at_speed", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "charger": [("stiffness", 1.0, 1.2), ("damp", 1.0, 1.2), ("stability_at_speed", 1.0, 1.5)],
        "charge hard": [("stiffness", 1.0, 1.2), ("damp", 1.0, 1.2), ("stability_at_speed", 1.0, 1.5)],
        "charging": [("stiffness", 1.0, 1.2), ("damp", 1.0, 1.2), ("stability_at_speed", 1.0, 1.5)],
        "forgiving": [("forgiveness", 1.0, 1.5), ("stiffness", -0.5, 0.8)],
        "easy to ski": [("forgiveness", 1.0, 1.5), ("stiffness", -0.5, 0.8)],
        "user friendly": [("forgiveness", 1.0, 1.5)],
        "surfy": [("powder_float", 1.0, 1.5), ("playfulness", 1.0, 1.0)],
        "surf": [("powder_float", 1.0, 1.5), ("playfulness", 1.0, 1.0)],
        "surf the pow": [("powder_float", 1.0, 1.5), ("playfulness", 1.0, 1.0)],
        "pow slayer": [("powder_float", 1.0, 1.5)],
        "hard charger": [("stiffness", 1.0, 1.5), ("damp", 1.0, 1.2), ("stability_at_speed", 1.0, 1.5)],
        "race-inspired": [("stiffness", 1.0, 1.2), ("edge_grip", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "quick edge to edge": [("turn_initiation", 1.0, 1.5), ("playfulness", 1.0, 1.0)],
        "quick turn": [("turn_initiation", 1.0, 1.5), ("playfulness", 1.0, 1.0)],
        "nimble": [("playfulness", 1.0, 1.2), ("turn_initiation", 1.0, 1.2)],
        "poppy": [("playfulness", 1.0, 1.5)],
        "damp and stable": [("damp", 1.0, 1.5), ("stability_at_speed", 1.0, 1.5)],
        "smooth and composed": [("damp", 1.0, 1.5), ("stability_at_speed", 1.0, 1.0)],
        "crud buster": [("damp", 1.0, 1.5), ("stability_at_speed", 1.0, 1.0)],
        "chop": [("damp", 1.0, 1.5), ("stability_at_speed", 1.0, 1.0)],
        "east coast": [("edge_grip", 1.0, 1.5), ("damp", 1.0, 1.0)],
        "hardpack": [("edge_grip", 1.0, 1.5)],
        "groomer": [("edge_grip", 1.0, 1.2), ("stability_at_speed", 1.0, 1.0)],
        "one ski quiver": [("versatility", 1.0, 1.5)],
        "quiver of one": [("versatility", 1.0, 1.5)],
        "do everything": [("versatility", 1.0, 1.5)],
        "jack of all trades": [("versatility", 1.0, 1.5)],
    },
    "running_shoe": {
        "bouncy": [("responsiveness", 1.0, 1.5), ("cushioning", 1.0, 1.0)],
        "plush": [("cushioning", 1.0, 1.5), ("comfort", 1.0, 1.0)],
        "cloud-like": [("cushioning", 1.0, 1.5), ("comfort", 1.0, 1.0)],
        "race day": [("responsiveness", 1.0, 1.5), ("weight", -1.0, 1.2)],
        "daily trainer": [("durability", 1.0, 1.2), ("comfort", 1.0, 1.2), ("cushioning", 1.0, 1.0)],
        "trail": [("grip", 1.0, 1.5), ("stability", 1.0, 1.0)],
        "tank": [("durability", 1.0, 1.5)],
        "lightweight": [("weight", -1.0, 1.5)],
        "featherweight": [("weight", -1.0, 1.5)],
    },
}

# ---------------------------------------------------------------------------
# Ski domain configuration
# ---------------------------------------------------------------------------

SKI_ATTRIBUTES: dict[str, dict] = {
    "stiffness": {
        "synonyms": ["stiff", "rigid", "flex", "soft", "noodle", "stiffness", "rigidity"],
        "spectrum": ["soft", "medium", "stiff"],
        "positive_terms": ["stiff", "rigid", "powerful"],
        "negative_terms": ["soft", "noodle", "noodly", "flexible", "bendy"],
    },
    "damp": {
        "synonyms": ["damp", "dampness", "dampening", "vibration", "chatter", "smooth", "composed"],
        "spectrum": ["chattery", "moderate", "damp"],
        "positive_terms": ["damp", "smooth", "composed", "absorbs"],
        "negative_terms": ["chattery", "vibration", "harsh", "rattly"],
    },
    "edge_grip": {
        "synonyms": ["edge grip", "edge hold", "grip", "hold", "ice", "hardpack", "carving", "carve"],
        "spectrum": ["poor", "adequate", "strong"],
        "positive_terms": ["grip", "hold", "locks", "rails", "carves"],
        "negative_terms": ["slips", "slides", "washy", "no grip"],
    },
    "stability_at_speed": {
        "synonyms": ["stability", "stable", "speed", "high speed", "charging", "confidence", "composed at speed"],
        "spectrum": ["unstable", "moderate", "stable"],
        "positive_terms": ["stable", "planted", "confident", "composed"],
        "negative_terms": ["unstable", "shaky", "sketchy", "wobbly"],
    },
    "playfulness": {
        "synonyms": ["playful", "fun", "lively", "agile", "nimble", "maneuverable", "quick", "playfulness", "agility"],
        "spectrum": ["serious", "moderate", "playful"],
        "positive_terms": ["playful", "fun", "lively", "agile", "nimble", "poppy", "sprightly"],
        "negative_terms": ["boring", "serious", "lifeless", "dead"],
    },
    "powder_float": {
        "synonyms": ["powder", "float", "deep snow", "soft snow", "flotation", "powder float"],
        "spectrum": ["no float", "moderate", "surfy"],
        "positive_terms": ["floats", "surfs", "floaty"],
        "negative_terms": ["sinks", "submarines", "no float"],
    },
    "forgiveness": {
        "synonyms": ["forgiving", "forgiveness", "easy", "accessible", "friendly", "tolerant", "demanding", "punishing"],
        "spectrum": ["demanding", "moderate", "forgiving"],
        "positive_terms": ["forgiving", "easy", "friendly", "accessible", "tolerant"],
        "negative_terms": ["demanding", "punishing", "unforgiving", "harsh"],
    },
    "weight": {
        "synonyms": ["weight", "light", "heavy", "lightweight", "heft"],
        "spectrum": ["heavy", "moderate", "light"],
        "positive_terms": ["light", "lightweight"],
        "negative_terms": ["heavy", "hefty", "weighty"],
    },
    "versatility": {
        "synonyms": ["versatile", "versatility", "all-mountain", "quiver", "do-it-all", "one-ski"],
        "spectrum": ["specialist", "moderate", "versatile"],
        "positive_terms": ["versatile", "all-mountain", "do-it-all"],
        "negative_terms": ["one-dimensional", "specialist", "limited"],
    },
    "responsiveness": {
        "synonyms": ["responsive", "responsiveness", "snappy", "quick", "energy", "rebound", "pop"],
        "spectrum": ["dead", "moderate", "responsive"],
        "positive_terms": ["responsive", "snappy", "energetic", "poppy", "rebound"],
        "negative_terms": ["dead", "sluggish", "flat", "muted"],
    },
    "turn_initiation": {
        "synonyms": ["turn initiation", "turn entry", "edge-to-edge", "quick turns", "short turns", "turn initiation"],
        "spectrum": ["slow", "moderate", "quick"],
        "positive_terms": ["quick", "easy turn", "snappy", "pivots"],
        "negative_terms": ["slow", "hooky", "hard to turn"],
    },
}

# ---------------------------------------------------------------------------
# Running shoe domain configuration
# ---------------------------------------------------------------------------

RUNNING_SHOE_ATTRIBUTES: dict[str, dict] = {
    "cushioning": {
        "synonyms": ["cushion", "cushioning", "padding", "soft", "plush", "foam"],
        "spectrum": ["minimal", "moderate", "maximal"],
        "positive_terms": ["cushioned", "plush", "soft", "padded"],
        "negative_terms": ["firm", "hard", "minimal", "thin"],
    },
    "responsiveness": {
        "synonyms": ["responsive", "responsiveness", "bouncy", "springy", "snappy", "energy return"],
        "spectrum": ["dead", "moderate", "responsive"],
        "positive_terms": ["responsive", "bouncy", "springy", "snappy"],
        "negative_terms": ["dead", "flat", "sluggish"],
    },
    "stability": {
        "synonyms": ["stable", "stability", "support", "pronation"],
        "spectrum": ["neutral", "moderate", "stable"],
        "positive_terms": ["stable", "supportive"],
        "negative_terms": ["wobbly", "unstable", "tippy"],
    },
    "grip": {
        "synonyms": ["grip", "traction", "outsole", "trail"],
        "spectrum": ["poor", "moderate", "excellent"],
        "positive_terms": ["grippy", "traction", "sticky"],
        "negative_terms": ["slippery", "no grip"],
    },
    "breathability": {
        "synonyms": ["breathable", "breathability", "ventilation", "airflow"],
        "spectrum": ["poor", "moderate", "excellent"],
        "positive_terms": ["breathable", "airy", "ventilated"],
        "negative_terms": ["hot", "stuffy", "unbreathable"],
    },
    "durability": {
        "synonyms": ["durable", "durability", "longevity", "wear"],
        "spectrum": ["poor", "moderate", "excellent"],
        "positive_terms": ["durable", "long-lasting", "tough"],
        "negative_terms": ["wears out", "fragile", "short-lived"],
    },
    "weight": {
        "synonyms": ["weight", "light", "heavy", "lightweight", "heft"],
        "spectrum": ["heavy", "moderate", "light"],
        "positive_terms": ["light", "lightweight", "featherweight"],
        "negative_terms": ["heavy", "clunky", "brick"],
    },
    "comfort": {
        "synonyms": ["comfort", "comfortable", "fit", "cozy"],
        "spectrum": ["uncomfortable", "moderate", "comfortable"],
        "positive_terms": ["comfortable", "cozy", "great fit"],
        "negative_terms": ["uncomfortable", "tight", "blistering"],
    },
}

SKI_CATEGORIES: list[str] = [
    "race_slalom",
    "race_gs",
    "beginner_frontside",
    "expert_carving",
    "advanced_frontside",
    "all_mountain",
    "freeride",
    "powder",
    "park",
    "touring",
]

# Terrain tags that map to categories
SKI_TERRAIN_MAP: dict[str, list[str]] = {
    "on-piste": ["race_slalom", "race_gs", "expert_carving", "advanced_frontside", "beginner_frontside"],
    "race": ["race_slalom", "race_gs"],
    "groomed": ["beginner_frontside", "advanced_frontside", "expert_carving"],
    "all-mountain": ["all_mountain"],
    "off-piste": ["freeride", "powder"],
    "powder": ["powder"],
    "park": ["park"],
    "backcountry": ["touring", "freeride"],
}

# Category synonyms for query matching
CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "race_slalom": ["slalom", "race", "racing", "sl"],
    "race_gs": ["giant slalom", "gs", "race", "racing"],
    "beginner_frontside": ["beginner", "learning", "easy", "novice", "starter"],
    "expert_carving": ["carving", "carve", "expert carving", "frontside expert"],
    "advanced_frontside": ["frontside", "on-piste", "piste", "groomer", "carving"],
    "all_mountain": ["all-mountain", "all mountain", "versatile", "quiver of one", "one-ski", "do-it-all"],
    "freeride": ["freeride", "free ride", "big mountain", "big-mountain", "charging"],
    "powder": ["powder", "deep snow", "fat", "wide"],
    "park": ["park", "freestyle", "terrain park", "jib"],
    "touring": ["touring", "backcountry", "uphill", "skinning"],
}


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS: dict[str, dict] = {
    "ski": {
        "attributes": SKI_ATTRIBUTES,
        "categories": SKI_CATEGORIES,
        "category_synonyms": CATEGORY_SYNONYMS,
        "terrain_map": SKI_TERRAIN_MAP,
    },
    "running_shoe": {
        "attributes": RUNNING_SHOE_ATTRIBUTES,
        "categories": [],
        "category_synonyms": {},
        "terrain_map": {},
    },
}


def get_domain_config(domain: str) -> dict:
    """Get configuration for a domain, returning a generic config if unknown."""
    if domain in DOMAIN_CONFIGS:
        return DOMAIN_CONFIGS[domain]
    # Return a minimal config for unknown domains
    return {
        "attributes": {},
        "categories": [],
        "category_synonyms": {},
        "terrain_map": {},
    }
