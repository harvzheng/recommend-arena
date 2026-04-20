"""Domain attribute schemas for Bayesian recommender.

Each domain defines attribute specifications with types and priors.
The schema is the only domain-specific artifact required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AttributeSpec:
    """Specification for a single product attribute."""

    name: str
    attr_type: Literal["ordinal", "categorical", "continuous"]
    levels: list[str] | None  # for ordinal/categorical
    prior: dict  # prior hyperparameters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.attr_type,
            "levels": self.levels,
            "prior": self.prior,
        }


# ---------------------------------------------------------------------------
# Ski domain schema
# ---------------------------------------------------------------------------

SKI_SCHEMA = [
    AttributeSpec(
        "stiffness", "ordinal",
        ["very_soft", "soft", "medium", "stiff", "very_stiff"],
        prior={"alpha": [1, 1, 1, 1, 1]},
    ),
    AttributeSpec(
        "terrain", "categorical",
        ["on-piste", "all-mountain", "freeride", "backcountry", "park", "powder"],
        prior={"alpha": [1, 1, 1, 1, 1, 1]},
    ),
    AttributeSpec(
        "skill_level", "ordinal",
        ["beginner", "intermediate", "advanced", "expert"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "playfulness", "ordinal",
        ["low", "moderate", "high"],
        prior={"alpha": [1, 1, 1]},
    ),
    AttributeSpec(
        "dampness", "ordinal",
        ["low", "moderate", "high"],
        prior={"alpha": [1, 1, 1]},
    ),
    AttributeSpec(
        "edge_grip", "ordinal",
        ["weak", "moderate", "strong", "exceptional"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "powder_float", "ordinal",
        ["minimal", "moderate", "good", "excellent"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "weight_feel", "ordinal",
        ["ultralight", "light", "moderate", "heavy"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "waist_width_mm", "continuous", None,
        prior={"mu": 90.0, "sigma": 20.0, "obs_sigma": 5.0, "epsilon": 8.0},
    ),
]

# ---------------------------------------------------------------------------
# Running shoe domain schema
# ---------------------------------------------------------------------------

RUNNING_SHOE_SCHEMA = [
    AttributeSpec(
        "cushioning", "ordinal",
        ["minimal", "moderate", "high", "maximal"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "responsiveness", "ordinal",
        ["low", "moderate", "high", "very_high"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "stability", "ordinal",
        ["neutral", "mild_support", "moderate_support", "high_support"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "surface", "categorical",
        ["road", "trail", "track", "mixed"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "use_case", "categorical",
        ["daily_training", "long_run", "speed_work", "racing", "recovery"],
        prior={"alpha": [1, 1, 1, 1, 1]},
    ),
    AttributeSpec(
        "grip", "ordinal",
        ["low", "moderate", "high", "aggressive"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "weight_feel", "ordinal",
        ["ultralight", "light", "moderate", "heavy"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "durability", "ordinal",
        ["low", "moderate", "high", "very_high"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "breathability", "ordinal",
        ["low", "moderate", "high", "very_high"],
        prior={"alpha": [1, 1, 1, 1]},
    ),
    AttributeSpec(
        "weight_g", "continuous", None,
        prior={"mu": 270.0, "sigma": 50.0, "obs_sigma": 15.0, "epsilon": 20.0},
    ),
    AttributeSpec(
        "heel_drop_mm", "continuous", None,
        prior={"mu": 8.0, "sigma": 4.0, "obs_sigma": 2.0, "epsilon": 2.0},
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_SCHEMAS: dict[str, list[AttributeSpec]] = {
    "ski": SKI_SCHEMA,
    "running_shoe": RUNNING_SHOE_SCHEMA,
}


def get_schema(domain: str) -> list[AttributeSpec]:
    """Return the attribute schema for a domain, or raise KeyError."""
    if domain not in DOMAIN_SCHEMAS:
        raise KeyError(f"No schema defined for domain {domain!r}. "
                       f"Available: {list(DOMAIN_SCHEMAS)}")
    return DOMAIN_SCHEMAS[domain]


def get_spec(schema: list[AttributeSpec], attr_name: str) -> AttributeSpec | None:
    """Look up an AttributeSpec by name, returning None if not found."""
    for s in schema:
        if s.name == attr_name:
            return s
    return None


# ---------------------------------------------------------------------------
# Category-based informative priors (Enhancement #4)
# ---------------------------------------------------------------------------

CATEGORY_PRIORS: dict[str, dict[str, dict[str, list[float] | float]]] = {
    # Ski categories
    "freeride": {
        "stiffness": {"alpha": [0.5, 0.5, 1.5, 3, 2]},
        "terrain": {"alpha": [0.5, 1, 4, 1, 0.5, 2]},
        "skill_level": {"alpha": [0.5, 0.5, 2, 3]},
        "powder_float": {"alpha": [0.5, 1, 2, 3]},
        "dampness": {"alpha": [0.5, 1, 3]},
        "weight_feel": {"alpha": [0.5, 1, 2, 2]},
        "edge_grip": {"alpha": [0.5, 1.5, 2, 1.5]},
        "waist_width_mm": {"mu": 105.0, "sigma": 12.0},
    },
    "all-mountain": {
        "stiffness": {"alpha": [0.5, 1, 3, 2, 1]},
        "terrain": {"alpha": [1.5, 4, 1.5, 0.5, 0.5, 0.5]},
        "skill_level": {"alpha": [0.5, 2, 3, 1.5]},
        "powder_float": {"alpha": [1, 2.5, 2, 1]},
        "dampness": {"alpha": [1, 2, 2]},
        "weight_feel": {"alpha": [0.5, 1.5, 2.5, 1]},
        "edge_grip": {"alpha": [0.5, 2, 2.5, 1]},
        "waist_width_mm": {"mu": 92.0, "sigma": 10.0},
    },
    "carving": {
        "stiffness": {"alpha": [0.5, 0.5, 1.5, 3, 2]},
        "terrain": {"alpha": [4, 1.5, 0.5, 0.5, 0.5, 0.5]},
        "skill_level": {"alpha": [0.5, 1, 3, 2]},
        "edge_grip": {"alpha": [0.5, 1, 2, 3.5]},
        "dampness": {"alpha": [0.5, 1.5, 3]},
        "powder_float": {"alpha": [3, 1.5, 0.5, 0.5]},
        "waist_width_mm": {"mu": 76.0, "sigma": 8.0},
    },
    "park": {
        "stiffness": {"alpha": [1, 3, 2, 0.5, 0.5]},
        "terrain": {"alpha": [0.5, 1, 0.5, 0.5, 4, 0.5]},
        "playfulness": {"alpha": [0.5, 1, 3.5]},
        "skill_level": {"alpha": [0.5, 2, 3, 1]},
        "weight_feel": {"alpha": [1.5, 2.5, 1, 0.5]},
    },
    "backcountry": {
        "stiffness": {"alpha": [0.5, 1, 2.5, 2, 1]},
        "terrain": {"alpha": [0.5, 1, 1, 4, 0.5, 1.5]},
        "weight_feel": {"alpha": [3, 2.5, 0.5, 0.5]},
        "powder_float": {"alpha": [0.5, 1.5, 2, 2.5]},
        "waist_width_mm": {"mu": 98.0, "sigma": 12.0},
    },
    "powder": {
        "stiffness": {"alpha": [0.5, 1, 2, 2.5, 1.5]},
        "terrain": {"alpha": [0.5, 0.5, 2, 1, 0.5, 4]},
        "powder_float": {"alpha": [0.5, 0.5, 1.5, 4]},
        "waist_width_mm": {"mu": 112.0, "sigma": 10.0},
        "weight_feel": {"alpha": [0.5, 1, 2, 2.5]},
    },
    # Running shoe categories
    "racing": {
        "cushioning": {"alpha": [1.5, 2, 1.5, 0.5]},
        "responsiveness": {"alpha": [0.5, 0.5, 2, 3.5]},
        "weight_feel": {"alpha": [3, 2.5, 0.5, 0.5]},
        "use_case": {"alpha": [0.5, 0.5, 1.5, 4, 0.5]},
        "weight_g": {"mu": 210.0, "sigma": 30.0},
    },
    "daily_trainer": {
        "cushioning": {"alpha": [0.5, 2, 3, 1]},
        "durability": {"alpha": [0.5, 1, 3, 2]},
        "use_case": {"alpha": [4, 2, 1, 0.5, 0.5]},
        "stability": {"alpha": [2, 2, 1.5, 0.5]},
    },
    "trail": {
        "surface": {"alpha": [0.5, 4, 0.5, 1]},
        "grip": {"alpha": [0.5, 1, 2.5, 3]},
        "stability": {"alpha": [1, 1.5, 2, 2]},
        "durability": {"alpha": [0.5, 1, 2, 3]},
    },
    "recovery": {
        "cushioning": {"alpha": [0.5, 0.5, 2, 3.5]},
        "responsiveness": {"alpha": [3, 2, 0.5, 0.5]},
        "use_case": {"alpha": [0.5, 0.5, 0.5, 0.5, 4]},
        "stability": {"alpha": [1.5, 2, 2, 1]},
    },
    "speed_work": {
        "responsiveness": {"alpha": [0.5, 1, 2.5, 3]},
        "cushioning": {"alpha": [1, 2.5, 2, 0.5]},
        "weight_feel": {"alpha": [2, 2.5, 1, 0.5]},
        "use_case": {"alpha": [0.5, 0.5, 4, 1.5, 0.5]},
    },
}


def get_category_priors(category: str) -> dict[str, dict] | None:
    """Return informative priors for a product category, or None."""
    cat_lower = category.lower().strip()
    # Try direct match first
    if cat_lower in CATEGORY_PRIORS:
        return CATEGORY_PRIORS[cat_lower]
    # Normalize separators and retry
    cat_norm = cat_lower.replace("-", "_").replace(" ", "_")
    if cat_norm in CATEGORY_PRIORS:
        return CATEGORY_PRIORS[cat_norm]
    # Try both separator forms of each key against the input
    for key, priors in CATEGORY_PRIORS.items():
        key_norm = key.replace("-", "_").replace(" ", "_")
        if key_norm == cat_norm:
            return priors
        # Substring matching
        if key in cat_lower or cat_lower in key:
            return priors
        if key_norm in cat_norm or cat_norm in key_norm:
            return priors
    return None
