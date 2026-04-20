"""Bayesian update logic using conjugate priors.

- Dirichlet-Multinomial for categorical and ordinal attributes
- Normal-Normal for continuous attributes

All updates are constant-time arithmetic -- no MCMC or sampling.
"""

from __future__ import annotations

import logging

from .beliefs import ProductBelief
from .schema import AttributeSpec, get_spec

logger = logging.getLogger(__name__)


def update_posterior(
    belief: ProductBelief,
    observation: dict[str, str],
    schema: list[AttributeSpec],
) -> None:
    """Apply a single observation (extracted from one review) to a product's posteriors.

    Args:
        belief: The product's current belief state (mutated in place).
        observation: Mapping of attribute_name -> observed_value.
        schema: The domain's attribute schema.
    """
    any_update = False

    for attr_name, observed_value in observation.items():
        spec = get_spec(schema, attr_name)
        if spec is None:
            logger.debug("Skipping unknown attribute %r", attr_name)
            continue

        posterior = belief.posteriors.get(attr_name)
        if posterior is None:
            logger.debug("No posterior for attribute %r", attr_name)
            continue

        if spec.attr_type in ("ordinal", "categorical"):
            if spec.levels is None:
                continue
            # Normalize the observed value for lookup
            val = observed_value.strip().lower()
            # Try exact match first, then case-insensitive
            idx = _level_index(spec.levels, val)
            if idx is None:
                logger.debug(
                    "Value %r not in levels %r for attribute %r",
                    observed_value, spec.levels, attr_name,
                )
                continue
            # Dirichlet update: increment the matching alpha
            posterior["alpha"][idx] += 1
            any_update = True

        elif spec.attr_type == "continuous":
            try:
                x = float(observed_value)
            except (ValueError, TypeError):
                logger.debug("Cannot parse %r as float for %r", observed_value, attr_name)
                continue

            mu_0 = posterior["mu"]
            sigma_0 = posterior["sigma"]
            sigma_obs = spec.prior.get("obs_sigma", 5.0)

            precision_0 = 1.0 / (sigma_0 ** 2)
            precision_obs = 1.0 / (sigma_obs ** 2)

            posterior["mu"] = (
                (precision_0 * mu_0 + precision_obs * x) /
                (precision_0 + precision_obs)
            )
            posterior["sigma"] = (1.0 / (precision_0 + precision_obs)) ** 0.5
            any_update = True

    if any_update:
        belief.evidence_count += 1


def _level_index(levels: list[str], value: str) -> int | None:
    """Find the index of *value* in *levels*, case-insensitive."""
    val_lower = value.lower().replace(" ", "_").replace("-", "_")
    for i, lvl in enumerate(levels):
        if lvl.lower().replace(" ", "_").replace("-", "_") == val_lower:
            return i
    return None
