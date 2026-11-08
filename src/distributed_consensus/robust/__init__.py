"""Robust information-processing utilities for adversarial MAS experiments."""

from .wmsr import (
    WMSRLocalResult,
    wmsr_local,
    wmsr_local_with_details,
    is_r_robust,
    is_rs_robust,
    assess_wmsr_topology,
    validate_adversary_set,
)

__all__ = [
    "WMSRLocalResult",
    "wmsr_local",
    "wmsr_local_with_details",
    "is_r_robust",
    "is_rs_robust",
    "assess_wmsr_topology",
    "validate_adversary_set",
]
