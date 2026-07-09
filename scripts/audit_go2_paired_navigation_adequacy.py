#!/usr/bin/env python3
"""Audit G2 dataset-v2 selection/calibration adequacy without touching G2.

Enforces the preregistered floors from the execution contract
(docs/lewm_go2_generalization_execution_contract_2026-07-09.md, "G2 dataset-v2
preregistration") before any model training:

- the checkpoint-selection and probability-calibration roles each contain all
  three occupancy classes among supervised next-frame cells;
- the probability-calibration role carries at least 10,000 FREE and 1,000
  OCCUPIED supervised next-frame cells;
- at least 90% of loaded rows carry a nonempty next-observed mask.

The untouched ``g2_evaluation`` role is never opened: its shards are not read
and its labels contribute nothing to any check, so a remediation decision made
from this audit cannot leak G2 information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lewm.datasets.go2_paired_navigation import (  # noqa: E402
    DATASET_ROLES,
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
    sha256_file,
)

LOADED_ROLES = ("train", "checkpoint_selection", "probability_calibration")
UNTOUCHED_ROLE = "g2_evaluation"
CLASS_NAMES = {UNKNOWN_CLASS: "unknown", FREE_CLASS: "free", OCCUPIED_CLASS: "occupied"}


class AdequacyError(RuntimeError):
    """The dataset cannot support the preregistered training protocol."""


def audit_paired_navigation_adequacy(
    dataset_manifest_path: Path,
    *,
    minimum_calibration_free_cells: int = 10_000,
    minimum_calibration_occupied_cells: int = 1_000,
    minimum_next_observed_row_fraction: float = 0.90,
) -> dict[str, Any]:
    """Return an adequacy report; every gate is recorded, none is silent."""

    manifest_path = Path(dataset_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "lewm_go2_paired_navigation_dataset_v2":
        raise AdequacyError("unsupported paired-navigation dataset schema")
    scene_roles = manifest.get("scene_roles")
    if not isinstance(scene_roles, dict) or "assignments" not in scene_roles:
        raise AdequacyError(
            "adequacy audit requires a direct family-role dataset "
            "(manifest.scene_roles.assignments)"
        )
    assignments = {
        str(scene): str(role) for scene, role in scene_roles["assignments"].items()
    }
    unknown_roles = sorted(set(assignments.values()) - set(DATASET_ROLES))
    if unknown_roles:
        raise AdequacyError(f"unknown dataset roles: {unknown_roles}")

    shard_by_scene: dict[str, dict[str, Any]] = {}
    for record in manifest.get("shards", ()):
        shard_by_scene[str(record["scene_id"])] = record

    per_role: dict[str, dict[str, Any]] = {
        role: {
            "scene_count": 0,
            "row_count": 0,
            "supervised_next_cell_counts": {"unknown": 0, "free": 0, "occupied": 0},
            "rows_with_nonempty_next_observed": 0,
        }
        for role in LOADED_ROLES
    }
    untouched_scene_count = 0
    shards_opened = 0

    for scene_id, role in sorted(assignments.items()):
        if role == UNTOUCHED_ROLE:
            untouched_scene_count += 1
            continue
        record = shard_by_scene.get(scene_id)
        if record is None:
            raise AdequacyError(f"scene {scene_id!r} has no label shard record")
        shard_path = Path(record["path"])
        actual_hash = sha256_file(shard_path)
        if actual_hash != str(record["sha256"]):
            raise AdequacyError(
                f"label shard hash mismatch for {scene_id!r}: expected "
                f"{record['sha256']}, got {actual_hash}"
            )
        with np.load(shard_path, allow_pickle=False) as archive:
            next_labels = np.asarray(archive["next_labels"])
            next_supervision = np.asarray(archive["next_supervision_mask"], dtype=bool)
            next_observed = np.asarray(archive["next_observed_mask"], dtype=bool)
        if next_labels.shape != next_supervision.shape or (
            next_labels.shape != next_observed.shape
        ):
            raise AdequacyError(f"shard arrays disagree in shape for {scene_id!r}")
        rows = int(next_labels.shape[0])
        if rows != int(record["rows"]):
            raise AdequacyError(
                f"shard row count mismatch for {scene_id!r}: manifest says "
                f"{record['rows']}, shard has {rows}"
            )
        shards_opened += 1
        stats = per_role[role]
        stats["scene_count"] += 1
        stats["row_count"] += rows
        supervised = next_labels[next_supervision]
        for class_value, class_name in CLASS_NAMES.items():
            stats["supervised_next_cell_counts"][class_name] += int(
                np.count_nonzero(supervised == class_value)
            )
        stats["rows_with_nonempty_next_observed"] += int(
            np.count_nonzero(next_observed.reshape(rows, -1).any(axis=1))
        )

    for role in LOADED_ROLES:
        stats = per_role[role]
        rows = stats["row_count"]
        stats["next_observed_row_fraction"] = (
            stats["rows_with_nonempty_next_observed"] / rows if rows else 0.0
        )

    loaded_rows = sum(per_role[role]["row_count"] for role in LOADED_ROLES)
    loaded_observed = sum(
        per_role[role]["rows_with_nonempty_next_observed"] for role in LOADED_ROLES
    )
    combined_fraction = loaded_observed / loaded_rows if loaded_rows else 0.0

    def class_counts(role: str) -> dict[str, int]:
        return per_role[role]["supervised_next_cell_counts"]

    checks = {
        "selection_role_has_all_three_classes": all(
            count > 0 for count in class_counts("checkpoint_selection").values()
        ),
        "calibration_role_has_all_three_classes": all(
            count > 0 for count in class_counts("probability_calibration").values()
        ),
        "calibration_free_cells_at_floor": (
            class_counts("probability_calibration")["free"]
            >= minimum_calibration_free_cells
        ),
        "calibration_occupied_cells_at_floor": (
            class_counts("probability_calibration")["occupied"]
            >= minimum_calibration_occupied_cells
        ),
        "next_observed_row_fraction_at_floor": (
            combined_fraction >= minimum_next_observed_row_fraction
        ),
        "every_loaded_role_nonempty": all(
            per_role[role]["row_count"] > 0 for role in LOADED_ROLES
        ),
    }

    report = {
        "schema": "lewm_go2_paired_navigation_adequacy_v1",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "floors": {
            "minimum_calibration_free_cells": int(minimum_calibration_free_cells),
            "minimum_calibration_occupied_cells": int(
                minimum_calibration_occupied_cells
            ),
            "minimum_next_observed_row_fraction": float(
                minimum_next_observed_row_fraction
            ),
        },
        "cell_semantics": (
            "supervised next-frame cells (next_supervision_mask); the row "
            "coverage gate uses nonempty next_observed_mask per row"
        ),
        "per_role": per_role,
        "combined_loaded_rows": loaded_rows,
        "combined_next_observed_row_fraction": combined_fraction,
        "untouched_g2_scene_count": untouched_scene_count,
        "untouched_g2_shards_opened": 0,
        "loaded_shards_opened": shards_opened,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-calibration-free-cells", type=int, default=10_000)
    parser.add_argument("--min-calibration-occupied-cells", type=int, default=1_000)
    parser.add_argument(
        "--min-next-observed-row-fraction", type=float, default=0.90
    )
    args = parser.parse_args()
    report = audit_paired_navigation_adequacy(
        args.dataset_manifest,
        minimum_calibration_free_cells=args.min_calibration_free_cells,
        minimum_calibration_occupied_cells=args.min_calibration_occupied_cells,
        minimum_next_observed_row_fraction=args.min_next_observed_row_fraction,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"passed": report["passed"], "checks": report["checks"]}, sort_keys=True
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
