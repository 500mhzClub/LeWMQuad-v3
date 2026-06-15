"""Upstream task-aligned source-index readiness for Phase 2D generation."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS
from .phase2d_generation import phase2d_lineage_fields
from .phase2d_readiness import canonical_split_name


def _target_cell(row: Mapping) -> int | None:
    oracle_next = row.get("oracle_next_cell_id")
    if oracle_next is not None:
        return int(oracle_next)
    route_target = int(row.get("route_target_id", -1))
    return route_target if route_target >= 0 else None


def _path_exists(value) -> bool:
    return value is not None and Path(str(value)).is_file()


def _source_state_key(row: Mapping, line_number: int) -> tuple[str, str]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("start_frame") or row.get("start_timestamp_ns") or line_number),
    )


def _load_manifest(path: str, cache: dict[str, dict]) -> dict | None:
    if path in cache:
        return cache[path]
    manifest_path = Path(path)
    if not manifest_path.is_file():
        cache[path] = None
        return None
    payload = json.loads(manifest_path.read_text())
    cache[path] = payload
    return payload


def _audit_one_source_index(
    split_name: str,
    path: Path,
    *,
    require_local_target_frame: bool,
) -> tuple[dict, set[str], set[tuple[str, str]], set[tuple[object, object]]]:
    requirements = CONFIRMATORY_SPLIT_REQUIREMENTS[split_name]
    rows = 0
    eligible_rows = 0
    target_rows = 0
    skipped_missing_local_target = 0
    split_label_mismatch_rows = 0
    missing_manifest_rows = 0
    lineage_verified_rows = 0
    missing_lineage_field_counts = Counter()
    raw_scene_counts = Counter()
    eligible_scene_counts = Counter()
    family_counts = Counter()
    decision_type_counts = Counter()
    scene_ids = set()
    eligible_source_keys = set()
    lineage_pairs = set()
    manifest_cache: dict[str, dict] = {}

    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            rows += 1
            scene_id = str(row.get("scene_id", ""))
            scene_ids.add(scene_id)
            raw_scene_counts[scene_id] += 1
            family_counts[str(row.get("family", ""))] += 1
            for decision_type in row.get("decision_types", ()):
                decision_type_counts[str(decision_type)] += 1
            try:
                row_split = canonical_split_name(str(row.get("split", "")))
            except ValueError:
                row_split = str(row.get("split", ""))
            if row_split != split_name:
                split_label_mismatch_rows += 1

            target_cell = _target_cell(row)
            local_target_frame = row.get("local_target_frame")
            if target_cell is not None:
                target_rows += 1
            if (
                require_local_target_frame
                and target_cell is not None
                and not _path_exists(local_target_frame)
            ):
                skipped_missing_local_target += 1
                continue

            manifest_path = str(row.get("scene_manifest", ""))
            manifest = _load_manifest(manifest_path, manifest_cache)
            if manifest is None:
                missing_manifest_rows += 1
            lineage = phase2d_lineage_fields(row, scene_manifest=manifest)
            lineage_audit = lineage["phase2d_source_state_lineage"]
            if lineage_audit["lineage_verified"]:
                lineage_verified_rows += 1
                lineage_pairs.add((lineage["topology_seed"], lineage["visual_seed"]))
            else:
                for field in lineage_audit["missing_fields"]:
                    missing_lineage_field_counts[str(field)] += 1
            eligible_rows += 1
            eligible_scene_counts[scene_id] += 1
            eligible_source_keys.add(_source_state_key(row, line_number))

    minimum_eligible_sources_per_scene = min(
        eligible_scene_counts.values(),
        default=0,
    )
    passed = (
        rows > 0
        and len(scene_ids) >= requirements["minimum_scenes"]
        and minimum_eligible_sources_per_scene
        >= requirements["minimum_source_states_per_scene"]
        and split_label_mismatch_rows == 0
        and missing_manifest_rows == 0
        and lineage_verified_rows == eligible_rows
        and eligible_rows > 0
    )
    return (
        {
            "path": str(path.resolve()),
            "present": True,
            "rows": rows,
            "eligible_source_rows": eligible_rows,
            "target_rows": target_rows,
            "skipped_missing_local_target_frame": skipped_missing_local_target,
            "scenes": len(scene_ids),
            "minimum_scenes_required": requirements["minimum_scenes"],
            "minimum_scene_count_passed": (
                len(scene_ids) >= requirements["minimum_scenes"]
            ),
            "minimum_eligible_source_rows_per_scene": (
                minimum_eligible_sources_per_scene
            ),
            "minimum_source_states_per_scene_required": requirements[
                "minimum_source_states_per_scene"
            ],
            "minimum_source_states_per_scene_passed": (
                minimum_eligible_sources_per_scene
                >= requirements["minimum_source_states_per_scene"]
            ),
            "split_label_mismatch_rows": split_label_mismatch_rows,
            "missing_manifest_rows": missing_manifest_rows,
            "lineage_verified_rows": lineage_verified_rows,
            "missing_lineage_field_counts": dict(
                sorted(missing_lineage_field_counts.items())
            ),
            "lineage_verified": lineage_verified_rows == eligible_rows
            and eligible_rows > 0,
            "raw_rows_by_scene_min": min(raw_scene_counts.values(), default=0),
            "raw_rows_by_scene_max": max(raw_scene_counts.values(), default=0),
            "eligible_rows_by_scene_min": min(
                eligible_scene_counts.values(),
                default=0,
            ),
            "eligible_rows_by_scene_max": max(
                eligible_scene_counts.values(),
                default=0,
            ),
            "family_counts": dict(sorted(family_counts.items())),
            "decision_type_counts": dict(sorted(decision_type_counts.items())),
            "passed": passed,
        },
        scene_ids,
        eligible_source_keys,
        lineage_pairs,
    )


def audit_phase2d_source_indices(
    split_paths: Mapping[str, Path],
    *,
    require_local_target_frame: bool = True,
) -> dict:
    """Audit whether source decision indices can feed Phase 2D generation."""

    canonical_paths = {
        canonical_split_name(name): Path(path) for name, path in split_paths.items()
    }
    if len(canonical_paths) != len(split_paths):
        raise ValueError("split names must be unique after canonicalization")

    missing_splits = sorted(set(CONFIRMATORY_SPLIT_REQUIREMENTS) - set(canonical_paths))
    splits = {}
    split_scenes = {}
    split_sources = {}
    split_lineage = {}
    for split_name, path in sorted(canonical_paths.items()):
        if not path.is_file():
            splits[split_name] = {
                "path": str(path.resolve()),
                "present": False,
                "passed": False,
            }
            split_scenes[split_name] = set()
            split_sources[split_name] = set()
            split_lineage[split_name] = set()
            continue
        audit, scenes, sources, lineage_pairs = _audit_one_source_index(
            split_name,
            path,
            require_local_target_frame=require_local_target_frame,
        )
        splits[split_name] = audit
        split_scenes[split_name] = scenes
        split_sources[split_name] = sources
        split_lineage[split_name] = lineage_pairs

    overlaps = {}
    names = sorted(split_scenes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                "scene_ids": sorted(split_scenes[left] & split_scenes[right]),
                "source_state_keys": [
                    list(value)
                    for value in sorted(split_sources[left] & split_sources[right])
                ],
                "topology_visual_pairs": [
                    list(value)
                    for value in sorted(split_lineage[left] & split_lineage[right])
                ],
            }

    checks = {
        "all_required_source_indices_present": not missing_splits
        and all(splits.get(name, {}).get("present", False) for name in canonical_paths),
        "all_split_source_indices_passed": bool(splits)
        and all(split.get("passed", False) for split in splits.values())
        and not missing_splits,
        "no_scene_source_or_lineage_overlap": all(
            not overlap["scene_ids"]
            and not overlap["source_state_keys"]
            and not overlap["topology_visual_pairs"]
            for overlap in overlaps.values()
        ),
    }
    return {
        "schema": "jepa_phase2d_source_index_readiness_v0",
        "require_local_target_frame": require_local_target_frame,
        "required_splits": list(CONFIRMATORY_SPLIT_REQUIREMENTS),
        "missing_splits": missing_splits,
        "splits": splits,
        "pairwise_overlap": overlaps,
        "checks": checks,
        "ready_for_counterfactual_generation": all(checks.values()),
    }
