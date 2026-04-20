"""Domain-specific synonym and colloquial term expansion dictionary.

Used to pre-expand query text before LLM parsing, and to improve
attribute matching. No LLM calls — purely deterministic lookups.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colloquial phrase → concrete attribute expansions (per domain)
# ---------------------------------------------------------------------------
PHRASE_EXPANSIONS: dict[str, dict[str, list[str]]] = {
    "ski": {
        # Vague / colloquial ski phrases
        "ice coast": ["edge_grip", "damp", "stability", "hardpack"],
        "ice coast ripper": ["edge_grip", "damp", "stability", "stiffness"],
        "east coast": ["edge_grip", "damp", "stability", "hardpack"],
        "hardpack": ["edge_grip", "stability", "damp"],
        "bulletproof": ["edge_grip", "stability", "damp", "stiffness"],
        "alive underfoot": ["playfulness", "responsiveness"],
        "lively": ["playfulness", "responsiveness"],
        "alive": ["playfulness", "responsiveness"],
        "confidence-inspiring": ["stability", "damp", "edge_grip"],
        "confidence inspiring": ["stability", "damp", "edge_grip"],
        "confident": ["stability", "damp", "edge_grip"],
        "one-ski quiver": ["versatility"],
        "one ski quiver": ["versatility"],
        "quiver of one": ["versatility"],
        "quiver killer": ["versatility"],
        "do-it-all": ["versatility"],
        "do it all": ["versatility"],
        "all-mountain": ["versatility"],
        "all mountain": ["versatility"],
        "charger": ["stiffness", "damp", "stability"],
        "charge": ["stiffness", "damp", "stability"],
        "charging": ["stiffness", "damp", "stability"],
        "rail": ["stiffness", "stability", "edge_grip"],
        "surfy": ["powder_float", "playfulness"],
        "surf": ["powder_float", "playfulness"],
        "surfing": ["powder_float", "playfulness"],
        "buttery": ["playfulness"],  # implies low stiffness
        "butter": ["playfulness"],
        "poppy": ["responsiveness", "playfulness"],
        "pop": ["responsiveness", "playfulness"],
        "snappy": ["responsiveness", "turn_initiation"],
        "forgiving": ["stability"],  # implies low stiffness
        "easy-going": ["stability", "turn_initiation"],
        "easy going": ["stability", "turn_initiation"],
        "nimble": ["turn_initiation", "playfulness"],
        "quick edge to edge": ["turn_initiation", "responsiveness"],
        "quick": ["turn_initiation", "responsiveness"],
        "damp": ["damp", "vibration_absorption"],
        "smooth": ["damp", "vibration_absorption"],
        "stable": ["stability", "damp"],
        "stable at speed": ["stability", "damp", "stiffness"],
        "high speed": ["stability", "damp", "stiffness"],
        "fast": ["stability", "stiffness", "damp"],
        "floaty": ["powder_float"],
        "float": ["powder_float"],
        "powder": ["powder_float"],
        "deep snow": ["powder_float"],
        "pow": ["powder_float"],
        "crud": ["damp", "stability", "vibration_absorption"],
        "chop": ["damp", "vibration_absorption", "stability"],
        "choppy": ["damp", "vibration_absorption", "stability"],
        "mogul": ["turn_initiation", "playfulness", "responsiveness"],
        "bump": ["turn_initiation", "playfulness", "responsiveness"],
        "bumps": ["turn_initiation", "playfulness", "responsiveness"],
        "carving": ["edge_grip", "stability", "responsiveness"],
        "carve": ["edge_grip", "stability", "responsiveness"],
        "groomer": ["edge_grip", "stability", "turn_initiation"],
        "groomers": ["edge_grip", "stability", "turn_initiation"],
        "piste": ["edge_grip", "stability"],
        "on-piste": ["edge_grip", "stability"],
        "off-piste": ["powder_float", "versatility"],
        "backcountry": ["powder_float", "versatility"],
        "touring": ["powder_float", "versatility"],
        "park": ["playfulness", "responsiveness"],
        "playful": ["playfulness"],
        "stiff": ["stiffness"],
        "soft": ["playfulness"],  # implies low stiffness
        "light": ["turn_initiation"],
        "heavy": ["stability", "damp"],
        "responsive": ["responsiveness"],
        "versatile": ["versatility"],
        "grippy": ["edge_grip"],
        "grip": ["edge_grip"],
        "edge hold": ["edge_grip"],
        "edge grip": ["edge_grip"],
    },
    "running_shoe": {
        # Colloquial running shoe phrases
        "cloud-like": ["cushioning", "comfort"],
        "cloud like": ["cushioning", "comfort"],
        "bouncy": ["responsiveness", "cushioning"],
        "springy": ["responsiveness"],
        "plush": ["cushioning", "comfort"],
        "marshmallow": ["cushioning", "comfort"],
        "pillowy": ["cushioning", "comfort"],
        "rock plate": ["grip", "support"],
        "speed": ["responsiveness", "weight"],
        "fast": ["responsiveness", "weight"],
        "racer": ["responsiveness", "weight"],
        "racing": ["responsiveness", "weight"],
        "trail": ["grip", "durability", "support"],
        "road": ["cushioning", "responsiveness"],
        "long run": ["cushioning", "comfort", "durability"],
        "marathon": ["cushioning", "responsiveness", "comfort"],
        "ultra": ["cushioning", "comfort", "durability"],
        "daily trainer": ["cushioning", "durability", "comfort"],
        "everyday": ["cushioning", "durability", "comfort"],
        "workhorse": ["durability", "comfort", "cushioning"],
        "stable": ["stability", "support"],
        "pronation": ["stability", "support"],
        "neutral": ["flexibility", "cushioning"],
        "minimalist": ["flexibility", "weight"],
        "barefoot": ["flexibility", "weight"],
        "maximal": ["cushioning", "comfort"],
        "max cushion": ["cushioning", "comfort"],
        "breathable": ["breathability"],
        "airy": ["breathability", "weight"],
        "waterproof": ["durability"],
        "wet": ["grip"],
        "mud": ["grip", "durability"],
        "rocky": ["grip", "support", "durability"],
        "technical": ["grip", "support"],
        "responsive": ["responsiveness"],
        "cushy": ["cushioning", "comfort"],
        "supportive": ["support", "stability"],
        "lightweight": ["weight"],
        "light": ["weight"],
        "heavy": ["cushioning", "durability"],
        "durable": ["durability"],
        "grippy": ["grip"],
        "flexible": ["flexibility"],
        "stiff": ["support", "stability"],
        "comfortable": ["comfort"],
        "snug": ["comfort", "support"],
    },
}

# ---------------------------------------------------------------------------
# Attribute name synonyms → canonical attribute names (per domain)
# ---------------------------------------------------------------------------
ATTRIBUTE_SYNONYMS: dict[str, dict[str, str]] = {
    "ski": {
        "stiff": "stiffness",
        "stiffness": "stiffness",
        "flex": "stiffness",
        "damp": "damp",
        "damping": "damp",
        "dampening": "damp",
        "vibration": "vibration_absorption",
        "vibration absorption": "vibration_absorption",
        "chatter": "vibration_absorption",
        "float": "powder_float",
        "powder float": "powder_float",
        "powder_float": "powder_float",
        "flotation": "powder_float",
        "grip": "edge_grip",
        "edge grip": "edge_grip",
        "edge_grip": "edge_grip",
        "edge hold": "edge_grip",
        "hold": "edge_grip",
        "stable": "stability",
        "stability": "stability",
        "playful": "playfulness",
        "playfulness": "playfulness",
        "fun": "playfulness",
        "responsive": "responsiveness",
        "responsiveness": "responsiveness",
        "response": "responsiveness",
        "rebound": "responsiveness",
        "turn initiation": "turn_initiation",
        "turn_initiation": "turn_initiation",
        "turning": "turn_initiation",
        "initiation": "turn_initiation",
        "versatile": "versatility",
        "versatility": "versatility",
        "quiver": "versatility",
    },
    "running_shoe": {
        "cushion": "cushioning",
        "cushioning": "cushioning",
        "cushy": "cushioning",
        "padding": "cushioning",
        "responsive": "responsiveness",
        "responsiveness": "responsiveness",
        "response": "responsiveness",
        "bounce": "responsiveness",
        "springy": "responsiveness",
        "stable": "stability",
        "stability": "stability",
        "traction": "grip",
        "grip": "grip",
        "grippy": "grip",
        "outsole": "grip",
        "breathable": "breathability",
        "breathability": "breathability",
        "ventilation": "breathability",
        "durable": "durability",
        "durability": "durability",
        "longevity": "durability",
        "light": "weight",
        "lightweight": "weight",
        "weight": "weight",
        "heavy": "weight",
        "comfortable": "comfort",
        "comfort": "comfort",
        "cozy": "comfort",
        "supportive": "support",
        "support": "support",
        "arch support": "support",
        "flexible": "flexibility",
        "flexibility": "flexibility",
        "flex": "flexibility",
    },
}

# ---------------------------------------------------------------------------
# "Low" direction hints — when these words appear alongside an attribute,
# the user probably wants a LOW value for it
# ---------------------------------------------------------------------------
LOW_DIRECTION_HINTS: set[str] = {
    "buttery", "butter", "soft", "forgiving", "easy", "easy-going",
    "mellow", "gentle", "relaxed", "chill", "smooth",
    "minimalist", "barefoot", "minimal", "lightweight", "light",
}


def expand_query(query_text: str, domain: str) -> str:
    """Expand a query string by appending concrete attribute terms.

    Looks up colloquial phrases and appends their mapped attributes
    so the LLM parser (and embedding) have more signal.

    Returns the original query text with expansions appended.
    """
    query_lower = query_text.lower()
    phrases = PHRASE_EXPANSIONS.get(domain, {})

    expanded_terms: list[str] = []
    seen: set[str] = set()

    # Check longest phrases first to avoid partial matches
    sorted_phrases = sorted(phrases.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in query_lower:
            for attr in phrases[phrase]:
                if attr not in seen:
                    expanded_terms.append(attr)
                    seen.add(attr)

    if not expanded_terms:
        return query_text

    expansion = " [expanded attributes: " + ", ".join(expanded_terms) + "]"
    return query_text + expansion


def get_attribute_expansions(query_text: str, domain: str) -> list[str]:
    """Return list of canonical attribute names triggered by the query.

    Useful for TF-IDF query expansion and direct attribute matching.
    """
    query_lower = query_text.lower()
    phrases = PHRASE_EXPANSIONS.get(domain, {})

    attrs: list[str] = []
    seen: set[str] = set()

    sorted_phrases = sorted(phrases.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in query_lower:
            for attr in phrases[phrase]:
                if attr not in seen:
                    attrs.append(attr)
                    seen.add(attr)

    return attrs
