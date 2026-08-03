"""Pure compatibility admission for the DINO calibration replacement.

The original calibration attempt was consumed by a byte-exact comparison of
an upstream development-only adequacy result.  The authorized ROCm runtime
recomputes that result identically except for one finite, one-ULP SSIM scalar.
This module implements the replacement's complete compatibility exception.

It deliberately performs no filesystem or runtime access.  Binding checks,
the scoped evaluator substitution, and receipt publication belong to the
replacement runner.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping


TASK_RELEVANCE_SCHEMA = (
    "lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_v1"
)
TASK_RELEVANCE_PASS_STATUS = (
    "PASS_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY"
)
COMPATIBILITY_EVIDENCE_SCHEMA = (
    "lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_compatibility_admission_v1"
)
COMPATIBILITY_PASS_STATUS = "PASS_SINGLETON_SSIM_COMPATIBILITY_ADMISSION"
SSIM_PATH = (
    "measurements",
    "pixels",
    "minimum_reference_candidate_rgb_ssim",
)
SSIM_DOTTED_PATH = ".".join(SSIM_PATH)
SSIM_ABSOLUTE_TOLERANCE = 1.0e-12
SSIM_RELATIVE_TOLERANCE = 0.0
MINIMUM_SSIM_GATE = 0.99


class CompatibilityAdmissionError(ValueError):
    """Raised when a recomputation exceeds the sole frozen exception."""


def _canonical_json_bytes(value: object, *, label: str) -> bytes:
    """Encode finite JSON exactly like the frozen outer loader."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompatibilityAdmissionError(
            f"{label} is not canonical finite JSON"
        ) from exc


def _required_mapping(
    value: object, *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompatibilityAdmissionError(f"{label} is not a JSON object")
    return value


def _nested_value(
    document: Mapping[str, Any], path: tuple[str, ...], *, label: str
) -> object:
    value: object = document
    for part in path:
        panel = _required_mapping(value, label=label)
        if part not in panel:
            raise CompatibilityAdmissionError(
                f"{label} is missing {'.'.join(path)}"
            )
        value = panel[part]
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompatibilityAdmissionError(f"{label} is not a numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CompatibilityAdmissionError(f"{label} is not finite")
    return result


def _replace_nested_value(
    document: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    panel: dict[str, Any] = document
    for part in path[:-1]:
        child = panel.get(part)
        if not isinstance(child, dict):
            raise CompatibilityAdmissionError(
                f"document is missing {'.'.join(path)}"
            )
        panel = child
    panel[path[-1]] = value


def admit_task_relevance_result_v1(
    *,
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Admit an exact result or only the frozen singleton SSIM difference.

    Returns the original ``stored`` mapping, rather than a normalized or
    recomputed document, plus JSON-ready compatibility evidence.  The returned
    stored object is the same object supplied by the caller; neither input is
    mutated.
    """

    stored_document = _required_mapping(stored, label="stored result")
    recomputed_document = _required_mapping(
        recomputed, label="recomputed result"
    )
    stored_bytes = _canonical_json_bytes(stored_document, label="stored result")
    recomputed_bytes = _canonical_json_bytes(
        recomputed_document, label="recomputed result"
    )

    for label, document in (
        ("stored", stored_document),
        ("recomputed", recomputed_document),
    ):
        if document.get("schema") != TASK_RELEVANCE_SCHEMA:
            raise CompatibilityAdmissionError(
                f"{label} task-relevance schema is not the frozen schema"
            )
        if document.get("status") != TASK_RELEVANCE_PASS_STATUS:
            raise CompatibilityAdmissionError(
                f"{label} task-relevance status is not PASS"
            )
        thresholds = _required_mapping(
            document.get("thresholds"), label=f"{label} thresholds"
        )
        gate = _finite_number(
            thresholds.get("minimum_reference_candidate_rgb_ssim"),
            label=f"{label} minimum SSIM threshold",
        )
        if gate != MINIMUM_SSIM_GATE:
            raise CompatibilityAdmissionError(
                f"{label} minimum SSIM threshold changed"
            )

    stored_ssim = _finite_number(
        _nested_value(stored_document, SSIM_PATH, label="stored result"),
        label="stored minimum SSIM",
    )
    recomputed_ssim = _finite_number(
        _nested_value(
            recomputed_document, SSIM_PATH, label="recomputed result"
        ),
        label="recomputed minimum SSIM",
    )
    if stored_ssim < MINIMUM_SSIM_GATE or recomputed_ssim < MINIMUM_SSIM_GATE:
        raise CompatibilityAdmissionError(
            "stored and recomputed minimum SSIM must both retain the 0.99 gate"
        )
    if not math.isclose(
        stored_ssim,
        recomputed_ssim,
        rel_tol=SSIM_RELATIVE_TOLERANCE,
        abs_tol=SSIM_ABSOLUTE_TOLERANCE,
    ):
        raise CompatibilityAdmissionError(
            "minimum SSIM difference exceeds the frozen absolute tolerance"
        )

    canonical_exact = stored_bytes == recomputed_bytes
    normalized_recomputed = deepcopy(dict(recomputed_document))
    _replace_nested_value(
        normalized_recomputed,
        SSIM_PATH,
        _nested_value(stored_document, SSIM_PATH, label="stored result"),
    )
    if _canonical_json_bytes(
        normalized_recomputed, label="normalized recomputed result"
    ) != stored_bytes:
        raise CompatibilityAdmissionError(
            "task-relevance result changed outside the singleton SSIM path"
        )

    differing_paths = [] if canonical_exact else [SSIM_DOTTED_PATH]
    evidence = {
        "schema": COMPATIBILITY_EVIDENCE_SCHEMA,
        "status": COMPATIBILITY_PASS_STATUS,
        "stored_status": stored_document["status"],
        "recomputed_status": recomputed_document["status"],
        "allowed_differing_paths": [SSIM_DOTTED_PATH],
        "differing_paths": differing_paths,
        "canonical_exact": canonical_exact,
        "all_other_fields_canonical_exact": True,
        "stored_minimum_reference_candidate_rgb_ssim": stored_ssim,
        "recomputed_minimum_reference_candidate_rgb_ssim": recomputed_ssim,
        "absolute_difference": abs(stored_ssim - recomputed_ssim),
        "absolute_tolerance": SSIM_ABSOLUTE_TOLERANCE,
        "relative_tolerance": SSIM_RELATIVE_TOLERANCE,
        "minimum_ssim_gate": MINIMUM_SSIM_GATE,
        "both_values_finite": True,
        "both_values_at_or_above_gate": True,
        "returns_reviewed_stored_document": True,
    }
    return stored, evidence


__all__ = [
    "COMPATIBILITY_EVIDENCE_SCHEMA",
    "COMPATIBILITY_PASS_STATUS",
    "CompatibilityAdmissionError",
    "MINIMUM_SSIM_GATE",
    "SSIM_ABSOLUTE_TOLERANCE",
    "SSIM_DOTTED_PATH",
    "SSIM_PATH",
    "SSIM_RELATIVE_TOLERANCE",
    "TASK_RELEVANCE_PASS_STATUS",
    "TASK_RELEVANCE_SCHEMA",
    "admit_task_relevance_result_v1",
]
