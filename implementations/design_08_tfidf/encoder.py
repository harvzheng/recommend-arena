"""Attribute value to sparse feature vector encoding.

Handles the mapping from typed attribute values (ordinal, categorical,
boolean, numeric) into a flat sparse dictionary of ``{feature_key: float}``.
"""

from __future__ import annotations

from .catalog import AttributeDef


def encode_attribute(
    attr_def: AttributeDef,
    value,
    confidence: float = 1.0,
) -> dict[str, float]:
    """Encode a single attribute value into sparse feature keys.

    Returns a dict mapping feature keys to float values.

    For ordinal attributes the LLM is expected to return an integer 1-10
    which is normalized to [0, 1].  If a string label is provided instead
    it is mapped to an even spread across the ordinal range.

    For categorical attributes a one-hot-style encoding is used: each
    possible value gets a key, the matching value gets *confidence*,
    others get 0.

    For boolean attributes a single key is produced (0 or 1).

    For numeric attributes the raw float value is stored (normalized
    later during IDF computation or at query time).
    """
    name = attr_def.name
    atype = attr_def.attr_type

    if atype == "categorical":
        return _encode_categorical(name, attr_def.values or [], value, confidence)
    elif atype == "ordinal":
        return _encode_ordinal(name, attr_def.values, value, confidence)
    elif atype == "boolean":
        return _encode_boolean(name, value, confidence)
    elif atype == "numeric":
        return _encode_numeric(name, value, confidence)
    return {}


# ---------------------------------------------------------------------------
# Internal encoders
# ---------------------------------------------------------------------------

def _encode_categorical(
    name: str,
    allowed: list[str],
    value,
    confidence: float,
) -> dict[str, float]:
    """One-hot encoding for categorical attributes.

    *value* may be a single string or a list of strings (for multi-label
    attributes like terrain).
    """
    result: dict[str, float] = {}
    if isinstance(value, list):
        for v in allowed:
            result[f"{name}:{v}"] = confidence if v in value else 0.0
    else:
        for v in allowed:
            result[f"{name}:{v}"] = confidence if v == value else 0.0
    return result


def _encode_ordinal(
    name: str,
    labels: list[str] | None,
    value,
    confidence: float,
) -> dict[str, float]:
    """Encode an ordinal attribute to a single normalized key.

    The LLM should return an integer 1-10.  String labels are accepted
    as fallback and mapped to an evenly spaced scale.
    """
    if isinstance(value, (int, float)):
        normalized = (float(value) - 1.0) / 9.0  # 1->0.0, 10->1.0
    elif isinstance(value, str) and labels and value in labels:
        idx = labels.index(value)
        normalized = idx / max(len(labels) - 1, 1)
    else:
        # Unknown string — try to parse as number
        try:
            normalized = (float(value) - 1.0) / 9.0
        except (ValueError, TypeError):
            normalized = 0.5  # fallback to midpoint
    normalized = max(0.0, min(1.0, normalized)) * confidence
    return {name: normalized}


def _encode_boolean(name: str, value, confidence: float) -> dict[str, float]:
    if isinstance(value, bool):
        return {name: (1.0 if value else 0.0) * confidence}
    if isinstance(value, str):
        return {name: (1.0 if value.lower() in ("true", "yes", "1") else 0.0) * confidence}
    if isinstance(value, (int, float)):
        return {name: (1.0 if value else 0.0) * confidence}
    return {name: 0.0}


def _encode_numeric(name: str, value, confidence: float) -> dict[str, float]:
    try:
        return {name: float(value) * confidence}
    except (ValueError, TypeError):
        return {}
