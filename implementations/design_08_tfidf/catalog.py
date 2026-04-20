"""Attribute catalog definitions for each domain.

Each domain defines a catalog of known attributes with their types and
possible values.  The catalog drives attribute extraction, encoding,
and query parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttributeDef:
    """Definition of a single attribute in the catalog."""

    name: str
    attr_type: str  # "categorical", "ordinal", "numeric", "boolean"
    values: list[str] | None = None  # allowed values (categorical/ordinal)
    description: str = ""  # short description for LLM prompts

    def to_prompt_dict(self) -> dict:
        """Return a dict suitable for inclusion in LLM prompts."""
        d: dict = {"name": self.name, "type": self.attr_type}
        if self.values:
            d["values"] = self.values
        if self.description:
            d["description"] = self.description
        return d


# ---------------------------------------------------------------------------
# Built-in catalogs
# ---------------------------------------------------------------------------

SKI_CATALOG: list[AttributeDef] = [
    AttributeDef(
        "stiffness", "ordinal",
        description="How stiff the ski is (1=very soft, 10=very stiff)",
    ),
    AttributeDef(
        "damp", "ordinal",
        description="How well the ski absorbs vibrations (1=chattery, 10=very damp)",
    ),
    AttributeDef(
        "edge_grip", "ordinal",
        description="How well the ski holds on hard snow and ice (1=poor, 10=excellent)",
    ),
    AttributeDef(
        "stability_at_speed", "ordinal",
        description="How stable at high speed (1=wobbly, 10=rock solid)",
    ),
    AttributeDef(
        "playfulness", "ordinal",
        description="How fun/playful/lively the ski feels (1=dead, 10=very playful)",
    ),
    AttributeDef(
        "powder_float", "ordinal",
        description="How well the ski floats in powder (1=sinks, 10=great float)",
    ),
    AttributeDef(
        "forgiveness", "ordinal",
        description="How forgiving of mistakes (1=punishing, 10=very forgiving)",
    ),
    AttributeDef(
        "terrain", "categorical",
        values=[
            "on-piste", "off-piste", "all-mountain", "park",
            "race", "carving", "freeride", "powder",
            "big-mountain", "backcountry", "touring", "groomed",
        ],
        description="Types of terrain the ski is suited for (can be multiple)",
    ),
    AttributeDef(
        "weight", "ordinal",
        description="Perceived weight (1=very light, 10=very heavy)",
    ),
    AttributeDef(
        "rocker", "boolean",
        description="Whether the ski has rocker (tip and/or tail rocker = true, full camber = false)",
    ),
    AttributeDef(
        "waist_width_mm", "numeric",
        description="Waist width in millimeters",
    ),
    AttributeDef(
        "turn_radius", "ordinal",
        description="Turn radius preference (1=very short, 10=very long)",
    ),
    AttributeDef(
        "binding_system", "categorical",
        values=["plate", "flat", "integrated", "touring"],
        description="Binding system type",
    ),
]


# Registry keyed by domain name
CATALOGS: dict[str, list[AttributeDef]] = {
    "ski": SKI_CATALOG,
}


def get_catalog(domain: str) -> list[AttributeDef]:
    """Return the attribute catalog for *domain*, or an empty list."""
    return CATALOGS.get(domain, [])
