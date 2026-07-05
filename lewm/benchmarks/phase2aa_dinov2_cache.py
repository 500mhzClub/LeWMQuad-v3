"""Phase 2AA DINOv2 frame-cache contracts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .phase2_data import future_frame_paths, transition_validity

PHASE2AA_DINOV2_CACHE_SCHEMA = "jepa_phase2aa_dinov2_patch_feature_cache_v0"


@dataclass(frozen=True)
class Phase2AAFrameRecord:
    """One unique RGB frame requested by the DINOv2 feature cache."""

    frame_path: str
    roles: tuple[str, ...]
    row_count: int


def phase2aa_unique_frame_records(rows: Sequence[dict]) -> tuple[Phase2AAFrameRecord, ...]:
    """Return deterministic unique start/valid-future frames for DINOv2 caching."""

    roles_by_path: dict[str, set[str]] = {}
    counts = Counter()
    for row in rows:
        start = str(row["start_frame"])
        roles_by_path.setdefault(start, set()).add("start")
        counts[start] += 1
        for step, (path, valid) in enumerate(
            zip(future_frame_paths(row), transition_validity(row), strict=True)
        ):
            if path is None or not valid:
                continue
            value = str(path)
            roles_by_path.setdefault(value, set()).add(f"future_step_{step}")
            counts[value] += 1
    return tuple(
        Phase2AAFrameRecord(
            frame_path=path,
            roles=tuple(sorted(roles)),
            row_count=int(counts[path]),
        )
        for path, roles in sorted(roles_by_path.items())
    )


def phase2aa_frame_cache_audit(
    rows: Sequence[dict],
    records: Sequence[Phase2AAFrameRecord],
    *,
    split_name: str,
) -> dict:
    """Return auditable frame coverage for a Phase 2AA DINOv2 cache."""

    existing = sum(Path(record.frame_path).is_file() for record in records)
    roles = Counter(role for record in records for role in record.roles)
    transition_slots = sum(len(row.get("active_blocks", ())) for row in rows)
    valid_transitions = sum(sum(transition_validity(row)) for row in rows)
    return {
        "schema": "jepa_phase2aa_dinov2_frame_cache_audit_v0",
        "split": split_name,
        "source_rows": len(rows),
        "transition_slots": int(transition_slots),
        "valid_transitions": int(valid_transitions),
        "unique_frames": len(records),
        "existing_unique_frames": int(existing),
        "missing_unique_frames": len(records) - int(existing),
        "role_counts": dict(sorted(roles.items())),
        "all_frames_exist": existing == len(records),
    }
