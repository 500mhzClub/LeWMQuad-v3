"""Phase 2D counterfactual data-generation contract helpers."""
from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Mapping, Sequence

from .phase2d_readiness import LINEAGE_FIELD_ALIASES

PHASE2D_PRIMITIVE_NAMES = (
    "hold",
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "backward",
    "yaw_left",
    "yaw_right",
    "arc_left",
    "arc_right",
)
PHASE2D_HORIZON_BLOCKS = 2
PHASE2D_EXPECTED_SEQUENCE_COUNT = len(PHASE2D_PRIMITIVE_NAMES) ** PHASE2D_HORIZON_BLOCKS


def factorial_primitive_sequences(
    primitive_names: Sequence[str],
    *,
    horizon_blocks: int,
) -> tuple[tuple[str, ...], ...]:
    """Return the ordered full-factorial primitive grid for a source state."""

    names = tuple(str(name) for name in primitive_names)
    if not names:
        raise ValueError("primitive_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("primitive_names must be unique")
    if horizon_blocks < 1:
        raise ValueError("horizon_blocks must be at least 1")
    return tuple(tuple(sequence) for sequence in product(names, repeat=horizon_blocks))


def sequence_grid_audit(
    *,
    primitive_names: Sequence[str],
    horizon_blocks: int,
    sequences: Sequence[Sequence[str]] | None = None,
) -> dict:
    """Audit whether generated candidates match the registered factorial grid."""

    names = tuple(str(name) for name in primitive_names)
    expected_set = set(
        factorial_primitive_sequences(names, horizon_blocks=horizon_blocks)
    )
    observed = (
        factorial_primitive_sequences(names, horizon_blocks=horizon_blocks)
        if sequences is None
        else tuple(tuple(str(item) for item in sequence) for sequence in sequences)
    )
    observed_set = set(observed)
    first_action_counts = Counter(
        sequence[0] for sequence in observed if len(sequence) == horizon_blocks
    )
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    full_factorial_passed = (
        len(observed) == len(expected_set)
        and len(observed_set) == len(expected_set)
        and not missing
        and not unexpected
    )
    phase2d_full_81 = (
        tuple(names) == PHASE2D_PRIMITIVE_NAMES
        and horizon_blocks == PHASE2D_HORIZON_BLOCKS
        and full_factorial_passed
    )
    return {
        "schema": "jepa_phase2d_counterfactual_sequence_grid_audit_v0",
        "primitive_names": list(names),
        "primitive_count": len(names),
        "horizon_blocks": horizon_blocks,
        "expected_sequence_count": len(expected_set),
        "observed_sequence_count": len(observed),
        "unique_observed_sequence_count": len(observed_set),
        "duplicate_sequence_count": len(observed) - len(observed_set),
        "first_action_count": len(first_action_counts),
        "first_action_counts": dict(sorted(first_action_counts.items())),
        "missing_sequence_count": len(missing),
        "unexpected_sequence_count": len(unexpected),
        "missing_sequences_sample": [list(sequence) for sequence in missing[:16]],
        "unexpected_sequences_sample": [
            list(sequence) for sequence in unexpected[:16]
        ],
        "full_factorial_passed": full_factorial_passed,
        "phase2d_full_81_two_block_grid": phase2d_full_81,
    }


def _is_present(value) -> bool:
    return value is not None and value != ""


def _lookup_alias(container, aliases: Sequence[str]):
    if isinstance(container, Mapping):
        for alias in aliases:
            value = container.get(alias)
            if _is_present(value):
                return value
    else:
        for alias in aliases:
            value = getattr(container, alias, None)
            if _is_present(value):
                return value
    return None


def _lineage_sources(row: Mapping, scene_manifest=None):
    yield "row", row
    metadata = row.get("scene_metadata")
    if isinstance(metadata, Mapping):
        yield "row.scene_metadata", metadata
    manifest_metadata = row.get("scene_manifest_metadata")
    if isinstance(manifest_metadata, Mapping):
        yield "row.scene_manifest_metadata", manifest_metadata
    if scene_manifest is not None:
        yield "scene_manifest", scene_manifest


def phase2d_lineage_fields(row: Mapping, *, scene_manifest=None) -> dict:
    """Return canonical topology/visual lineage fields and an audit payload."""

    values = {}
    sources = {}
    missing = []
    for canonical, aliases in LINEAGE_FIELD_ALIASES.items():
        values[canonical] = None
        sources[canonical] = None
        for source_name, source in _lineage_sources(row, scene_manifest):
            value = _lookup_alias(source, aliases)
            if _is_present(value):
                values[canonical] = value
                sources[canonical] = source_name
                break
        if not _is_present(values[canonical]):
            missing.append(canonical)

    lineage_verified = not missing
    return {
        **values,
        "phase2d_source_state_lineage": {
            "schema": "jepa_phase2d_source_state_generation_lineage_v0",
            "scene_id": row.get("scene_id"),
            "source_index": row.get("source_index"),
            "start_frame": row.get("start_frame"),
            "required_fields": sorted(LINEAGE_FIELD_ALIASES),
            "field_aliases": {
                field: list(aliases)
                for field, aliases in sorted(LINEAGE_FIELD_ALIASES.items())
            },
            "field_sources": sources,
            "missing_fields": missing,
            "lineage_verified": lineage_verified,
        },
    }
