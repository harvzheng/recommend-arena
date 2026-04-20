"""Synonym expansion tables for normalizing attribute vocabulary.

Each domain has a mapping of canonical terms to their synonyms.
Both query terms and extracted attributes are expanded through the
synonym table so that "playful" and "fun" map to the same canonical
form.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Built-in synonym tables  (canonical -> list of synonyms)
# ---------------------------------------------------------------------------

SKI_SYNONYMS: dict[str, list[str]] = {
    # Stiffness
    "stiff": ["rigid", "firm", "hard", "demanding", "beefy"],
    "soft": ["forgiving", "flexible", "buttery", "easy-going", "noodle", "noodly"],
    # Feel
    "playful": ["fun", "lively", "energetic", "poppy", "surfy"],
    "stable": ["planted", "composed", "confident", "locked-in", "solid"],
    "damp": ["smooth", "composed", "vibration-free", "plush"],
    "chattery": ["harsh", "buzzy", "rattly", "vibrating"],
    # Weight
    "light": ["lightweight", "featherweight", "feathery"],
    "heavy": ["beefy", "burly", "tank", "hefty", "porky"],
    # Terrain
    "on-piste": ["groomer", "corduroy", "frontside", "piste", "groomed"],
    "off-piste": ["backcountry", "powder", "freeride", "sidecountry"],
    "all-mountain": ["versatile", "do-it-all", "quiver-of-one", "one-ski-quiver"],
    "park": ["freestyle", "jib", "terrain-park"],
    "race": ["racing", "competition", "slalom", "gs", "giant-slalom"],
    "carving": ["carve", "trenching", "laying-trenches", "rail-turns"],
    "powder": ["deep-snow", "blower", "freshies", "pow"],
    "big-mountain": ["big-mtn", "steep", "gnarly"],
    "touring": ["skinning", "uphill", "ski-touring", "alpine-touring", "at"],
    # Edge grip
    "grippy": ["edgy", "hooky", "rail", "on-rails", "locked-in"],
    # Float
    "floaty": ["surfy", "effortless-float", "stays-on-top"],
    # Binding
    "plate": ["race-plate", "binding-plate"],
    "flat": ["flat-mount", "direct-mount"],
}


SYNONYM_TABLES: dict[str, dict[str, list[str]]] = {
    "ski": SKI_SYNONYMS,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_synonym_map(domain: str) -> dict[str, str]:
    """Build a reverse mapping: synonym -> canonical term.

    The canonical term also maps to itself.
    """
    table = SYNONYM_TABLES.get(domain, {})
    reverse: dict[str, str] = {}
    for canonical, syns in table.items():
        reverse[canonical.lower()] = canonical
        for syn in syns:
            reverse[syn.lower()] = canonical
    return reverse


def expand_synonyms(text: str, syn_map: dict[str, str]) -> str:
    """Replace known synonyms in *text* with their canonical form."""
    words = text.lower().split()
    expanded = []
    for word in words:
        cleaned = word.strip(".,!?;:'\"()[]")
        canonical = syn_map.get(cleaned)
        if canonical:
            expanded.append(canonical)
        else:
            expanded.append(word)
    return " ".join(expanded)


def expand_value(value: str, syn_map: dict[str, str]) -> str:
    """Map a single attribute value to its canonical form if known."""
    return syn_map.get(value.lower().strip(), value)
