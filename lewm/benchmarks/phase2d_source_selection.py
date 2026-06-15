"""Deterministic source-row selection for bounded Phase 2D generation."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS
from .phase2d_generation import phase2d_lineage_fields
from .phase2d_readiness import canonical_split_name
from .phase2d_source_indices import _target_cell


def _path_exists(value) -> bool:
    return value is not None and Path(str(value)).is_file()


def _eligible(row: Mapping, *, require_local_target_frame: bool) -> bool:
    target_cell = _target_cell(row)
    if not require_local_target_frame or target_cell is None:
        return True
    return _path_exists(row.get("local_target_frame"))


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in (seed, *parts))
    return int.from_bytes(text.encode("utf-8"), "little", signed=False) % (2**63)


def _selected_scene_ids(
    rows_by_scene: Mapping[str, list[dict]],
    *,
    target_scenes: int,
    seed: int,
    split_name: str,
) -> list[str]:
    scenes_by_family: dict[str, list[str]] = defaultdict(list)
    for scene_id, rows in rows_by_scene.items():
        family = str(rows[0].get("family", ""))
        scenes_by_family[family].append(scene_id)
    rng = random.Random(_stable_seed(seed, split_name, "scenes"))
    for scenes in scenes_by_family.values():
        scenes.sort()
        rng.shuffle(scenes)

    selected = []
    max_len = max((len(values) for values in scenes_by_family.values()), default=0)
    for index in range(max_len):
        for family in sorted(scenes_by_family):
            scenes = scenes_by_family[family]
            if index < len(scenes):
                selected.append(scenes[index])
                if len(selected) >= target_scenes:
                    return selected
    return selected


def select_phase2d_source_rows(
    *,
    split_name: str,
    source_path: Path,
    output_path: Path,
    scene_count: int,
    source_states_per_scene: int,
    seed: int,
    require_local_target_frame: bool = True,
) -> dict:
    """Write selected source rows and return a provenance summary."""

    canonical_split = canonical_split_name(split_name)
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    input_rows = 0
    skipped_ineligible = 0
    skipped_split_mismatch = 0
    skipped_missing_lineage = 0
    manifest_cache: dict[str, dict | None] = {}
    with source_path.open() as stream:
        for line in stream:
            input_rows += 1
            row = json.loads(line)
            try:
                row_split = canonical_split_name(str(row.get("split", "")))
            except ValueError:
                row_split = str(row.get("split", ""))
            if row_split != canonical_split:
                skipped_split_mismatch += 1
                continue
            if not _eligible(row, require_local_target_frame=require_local_target_frame):
                skipped_ineligible += 1
                continue
            manifest_path = str(row.get("scene_manifest", ""))
            if manifest_path not in manifest_cache:
                path = Path(manifest_path)
                manifest_cache[manifest_path] = (
                    json.loads(path.read_text()) if path.is_file() else None
                )
            lineage = phase2d_lineage_fields(
                row,
                scene_manifest=manifest_cache[manifest_path],
            )
            if not lineage["phase2d_source_state_lineage"]["lineage_verified"]:
                skipped_missing_lineage += 1
                continue
            rows_by_scene[str(row["scene_id"])].append(row)

    eligible_scene_counts = Counter(
        scene_id
        for scene_id, rows in rows_by_scene.items()
        if len(rows) >= source_states_per_scene
    )
    candidate_rows_by_scene = {
        scene_id: rows
        for scene_id, rows in rows_by_scene.items()
        if len(rows) >= source_states_per_scene
    }
    selected_scenes = _selected_scene_ids(
        candidate_rows_by_scene,
        target_scenes=scene_count,
        seed=seed,
        split_name=canonical_split,
    )
    if len(selected_scenes) < scene_count:
        raise ValueError(
            f"{canonical_split} has {len(selected_scenes)} eligible scenes, "
            f"requires {scene_count}"
        )

    selected_rows = []
    selected_counts = {}
    for scene_id in selected_scenes:
        rows = list(candidate_rows_by_scene[scene_id])
        random.Random(_stable_seed(seed, canonical_split, scene_id, "rows")).shuffle(rows)
        chosen = rows[:source_states_per_scene]
        selected_counts[scene_id] = len(chosen)
        selected_rows.extend(chosen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for row in selected_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    requirements = CONFIRMATORY_SPLIT_REQUIREMENTS[canonical_split]
    passed = (
        scene_count >= requirements["minimum_scenes"]
        and source_states_per_scene
        >= requirements["minimum_source_states_per_scene"]
        and len(selected_rows) == scene_count * source_states_per_scene
    )
    summary = {
        "schema": "jepa_phase2d_source_row_selection_summary_v0",
        "split": canonical_split,
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "seed": seed,
        "require_local_target_frame": require_local_target_frame,
        "input_rows": input_rows,
        "skipped_split_mismatch": skipped_split_mismatch,
        "skipped_missing_local_target_or_ineligible": skipped_ineligible,
        "skipped_missing_lineage": skipped_missing_lineage,
        "eligible_scenes": len(candidate_rows_by_scene),
        "eligible_scene_count_histogram": dict(
            sorted(Counter(len(rows) for rows in candidate_rows_by_scene.values()).items())
        ),
        "requested_scene_count": scene_count,
        "requested_source_states_per_scene": source_states_per_scene,
        "selected_scene_count": len(selected_scenes),
        "selected_source_rows": len(selected_rows),
        "selected_scene_ids": selected_scenes,
        "selected_rows_by_scene": dict(sorted(selected_counts.items())),
        "minimum_scene_count_required": requirements["minimum_scenes"],
        "minimum_source_states_per_scene_required": requirements[
            "minimum_source_states_per_scene"
        ],
        "passes_registered_minimum": passed,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
