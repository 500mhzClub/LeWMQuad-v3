#!/usr/bin/env python3
"""Audit spatial grounding of a frozen physical-evidence checkpoint.

This is a development-only diagnostic.  It evaluates exactly the ``train`` and
``checkpoint_selection`` roles and fails closed before any artifact access when
the checkpoint or its training report contains a G2 result.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lewm.benchmarks.go2_physical_spatial_grounding import (  # noqa: E402
    ALLOWED_ROLES,
    alignment_accumulators_for_batch,
    canonical_json_sha256,
    deterministic_maximum_mismatch_permutation,
    distance_bin_masks,
    empty_loss_accumulator,
    empty_physical_accumulator,
    finalize_loss_accumulator,
    finalize_physical_accumulator,
    grounding_contrast,
    loss_accumulator_for_batch,
    merge_accumulator,
    physical_accumulator_for_batch,
    visibility_regions,
)
from lewm.benchmarks.traversability_metrics import (  # noqa: E402
    TraversabilityThresholds,
    evaluate_traversability,
)
from lewm.hierarchical_probability_calibration import (  # noqa: E402
    CALIBRATION_METHOD,
    hierarchical_calibrated_probabilities,
    validate_hierarchical_probability_calibration,
)
from lewm.models.egomotion_bev_jepa import (  # noqa: E402
    EgomotionBevJepa,
    PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    PROJECTIVE_COLUMN_ATTENTION_LIFT,
    build_projective_query_support_contract,
    validate_projective_query_support_binding,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SCHEMA = "lewm_go2_egomotion_bev_jepa_checkpoint_v4"
REPORT_SCHEMA = "lewm_go2_egomotion_bev_jepa_training_report_v4"
DATASET_SCHEMA = "lewm_go2_paired_navigation_dataset_v3"
LABEL_CONTRACT = "observable_physical_occupancy_v3"
TARGET_SPACE = "observable_physical_occupancy"
OBJECTIVE_MODE = "hierarchical_equal_capacity_v1"
CONDITIONS = (
    "correct_rgb",
    "role_global_shuffled_rgb",
    "per_channel_mean_rgb",
)
TRAINING_CRITICAL_SOURCE_PATHS = {
    "trainer_source": REPOSITORY_ROOT / "scripts/train_go2_egomotion_bev_jepa.py",
    "model_source": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
    "traversability_metrics_source": (
        REPOSITORY_ROOT / "lewm/benchmarks/traversability_metrics.py"
    ),
    "calibration_source": (
        REPOSITORY_ROOT / "lewm/hierarchical_probability_calibration.py"
    ),
    "dataset_contract_source": (
        REPOSITORY_ROOT / "lewm/datasets/go2_paired_navigation.py"
    ),
}
ENCODER_SOURCE_PATH = REPOSITORY_ROOT / "lewm/models/encoders.py"
ENCODER_SOURCE_GIT_PATH = "lewm/models/encoders.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row index entry is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--expected-training-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument(
        "--frozen-cell-square-counterfactual",
        action="store_true",
        help=(
            "Evaluate frozen center-projective weights with the physical-v3 "
            "cell-square geometry only; emits no checkpoint and is never eligible "
            "for promotion."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Fail on PyTorch operations without deterministic implementations.",
    )
    args = parser.parse_args(argv)
    if not args.development_only:
        parser.error("--development-only is required")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.workers != 0:
        parser.error("workers must be zero so artifact-open accounting stays exact")
    if args.output.exists():
        parser.error("output already exists; diagnostic artifacts are immutable")
    for name in (
        "expected_dataset_manifest_sha256",
        "expected_checkpoint_sha256",
        "expected_training_report_sha256",
    ):
        value = str(getattr(args, name))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"--{name.replace('_', '-')} must be lowercase 64-hex")
    return args


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("torch.load(..., weights_only=True) is required") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be an object")
    return payload


def _verified_expected_file_sha256(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> str:
    actual = _sha256_file(path)
    if actual != str(expected_sha256):
        raise ValueError(
            f"{name} SHA-256 differs from the precommitted value: "
            f"expected={expected_sha256} actual={actual}"
        )
    return actual


def _g2_contact(payload: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    ledger = payload.get("dataset_access_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError(f"{name} lacks a dataset access ledger")
    contact = ledger.get("g2_contact")
    if not isinstance(contact, Mapping):
        raise ValueError(f"{name} lacks a G2 contact ledger")
    for key in ("label_shard_byte_opens", "image_byte_opens", "model_output_rows"):
        if int(contact.get(key, -1)) != 0:
            raise ValueError(f"{name} records forbidden G2 access: {key}")
    return contact


def _require_no_g2_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("physical diagnostic requires checkpoint schema v4")
    if (
        checkpoint.get("g2_evaluation") is not None
        or bool(checkpoint.get("g2_passes", False))
        or checkpoint.get("head_g2_evaluation") is not None
        or bool(checkpoint.get("head_g2_passes", False))
    ):
        raise ValueError("checkpoint already contains a G2 evaluation")
    if bool(checkpoint.get("runtime_ready", False)):
        raise ValueError("development checkpoint unexpectedly claims runtime readiness")
    _g2_contact(checkpoint, name="checkpoint")


def _require_no_g2_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("physical diagnostic requires training report schema v4")
    promotion = report.get("promotion")
    if not isinstance(promotion, Mapping):
        raise ValueError("training report lacks promotion record")
    if (
        report.get("final_g2_evaluation") is not None
        or report.get("final_head_g2_evaluation") is not None
        or bool(promotion.get("g2_evaluated", False))
        or bool(promotion.get("head_g2_evaluated", False))
        or bool(promotion.get("head_g2_passes", False))
    ):
        raise ValueError("training report already contains a G2 evaluation")
    row_counts = report.get("row_counts")
    if not isinstance(row_counts, Mapping) or int(
        row_counts.get("g2_evaluation", -1)
    ) != 0:
        raise ValueError("training report did not preserve an empty G2 row subset")
    _g2_contact(report, name="training report")


def _validate_dataset_manifest(manifest: Mapping[str, Any]) -> Mapping[str, str]:
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("physical diagnostic requires paired-navigation dataset v3")
    semantics = manifest.get("label_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("dataset lacks physical label semantics")
    if (
        semantics.get("label_contract") != LABEL_CONTRACT
        or semantics.get("target_occupancy_space") != TARGET_SPACE
        or semantics.get("per_frame_configuration_classes_supervised") is not False
        or semantics.get("post_memory_configuration_derivation_is_evaluation_only")
        is not True
    ):
        raise ValueError("dataset is not observable physical occupancy v3")
    roles = manifest.get("scene_roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("assignments"), Mapping):
        raise ValueError("dataset lacks direct scene-role assignments")
    assignments = {str(key): str(value) for key, value in roles["assignments"].items()}
    if canonical_json_sha256(assignments) != str(roles.get("assignments_sha256")):
        raise ValueError("dataset scene-role assignment hash mismatch")
    return assignments


def _validated_geometry_contract(
    manifest: Mapping[str, Any],
) -> tuple[Path, str]:
    record = manifest.get("geometry_contract")
    if not isinstance(record, Mapping):
        raise ValueError("dataset lacks a geometry contract")
    path = Path(str(record.get("path", ""))).resolve()
    actual = _sha256_file(path)
    if actual != str(record.get("file_sha256", "")):
        raise ValueError("dataset geometry-contract file SHA-256 mismatch")
    return path, actual


def _validated_traversability_thresholds(
    checkpoint: Mapping[str, Any],
) -> TraversabilityThresholds:
    record = checkpoint.get("traversability_thresholds")
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint lacks traversability thresholds")
    thresholds = TraversabilityThresholds(**dict(record))
    thresholds.validate()
    if not (
        thresholds.occupied_detection_min
        > thresholds.occupied_probability_max
    ):
        raise ValueError("FREE admission and OCCUPIED detection intervals overlap")
    return thresholds


def _validate_checkpoint_report_dataset(
    checkpoint: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
) -> None:
    if str(checkpoint.get("dataset_manifest_sha256")) != dataset_manifest_sha256:
        raise ValueError("checkpoint targets a different dataset manifest")
    output_contract = checkpoint.get("occupancy_output_contract")
    if not isinstance(output_contract, Mapping) or output_contract.get(
        "target_occupancy_space"
    ) != TARGET_SPACE:
        raise ValueError("checkpoint does not target observable physical occupancy")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get(
        "bev_lift_type"
    ) not in {
        PROJECTIVE_COLUMN_ATTENTION_LIFT,
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    }:
        raise ValueError("diagnostic requires a registered projective lift")
    support_value = checkpoint.get("projective_query_support")
    if support_value is not None and not isinstance(support_value, Mapping):
        raise ValueError("checkpoint projective query support is malformed")
    support = validate_projective_query_support_binding(
        model_config=model_config,
        projective_query_support=support_value,
        dataset_manifest=manifest,
        occupancy_output_contract=output_contract,
    )
    provenance = checkpoint.get("training_run_provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "projective_query_support"
    ) != support:
        raise ValueError("training provenance projective query support differs")
    if report.get("projective_query_support") != support:
        raise ValueError("training report projective query support differs")
    objective = checkpoint.get("occupancy_training_objective")
    if not isinstance(objective, Mapping) or objective.get("mode") != OBJECTIVE_MODE:
        raise ValueError("checkpoint used the wrong occupancy objective")
    report_checkpoint = report.get("checkpoint")
    report_dataset = report.get("dataset_manifest")
    if (
        not isinstance(report_checkpoint, Mapping)
        or str(report_checkpoint.get("sha256")) != checkpoint_sha256
        or not isinstance(report_dataset, Mapping)
        or str(report_dataset.get("sha256")) != dataset_manifest_sha256
    ):
        raise ValueError("training report input hashes do not match")
    if report.get("training_run_provenance") != checkpoint.get(
        "training_run_provenance"
    ):
        raise ValueError("checkpoint and training report provenance differ")


def _git_blob_sha256(head: str, repository_path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{head}:{repository_path}"),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _status_mentions_path(status_short: str, repository_path: str) -> bool:
    return any(
        repository_path in line[3:].split(" -> ")
        for line in str(status_short).splitlines()
        if len(line) >= 4
    )


def _validate_training_source_provenance(
    checkpoint: Mapping[str, Any],
    *,
    allowed_counterfactual_source_changes: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Bind the diagnostic to the exact sources that produced the checkpoint."""

    provenance = checkpoint.get("training_run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint lacks embedded training-run provenance")
    provenance_core = dict(provenance)
    declared = str(provenance_core.pop("content_sha256", ""))
    if declared != canonical_json_sha256(provenance_core):
        raise ValueError("training-run provenance content hash mismatch")
    if provenance.get("schema") != "lewm_go2_training_run_provenance_v1":
        raise ValueError("unsupported training-run provenance schema")
    if provenance.get("checkpoint_artifact_included") is not False:
        raise ValueError("training-run provenance is circular")
    critical = provenance.get("critical_inputs")
    if not isinstance(critical, Mapping):
        raise ValueError("training-run provenance lacks critical inputs")

    verified: dict[str, dict[str, str]] = {}
    for name, path in sorted(TRAINING_CRITICAL_SOURCE_PATHS.items()):
        checkpoint_record = critical.get(name)
        if not isinstance(checkpoint_record, Mapping):
            raise ValueError(f"training-run provenance lacks {name}")
        current_sha256 = _sha256_file(path)
        recorded_sha256 = str(checkpoint_record.get("sha256", ""))
        matches_checkpoint = current_sha256 == recorded_sha256
        if not matches_checkpoint and name not in allowed_counterfactual_source_changes:
            raise ValueError(f"training critical source changed: {name}")
        verified[name] = {
            "path": str(path.resolve()),
            "current_sha256": current_sha256,
            "checkpoint_sha256": recorded_sha256,
            "matches_checkpoint": matches_checkpoint,
            "counterfactual_source_transition_allowed": (
                not matches_checkpoint
                and name in allowed_counterfactual_source_changes
            ),
        }

    git_record = provenance.get("git")
    if not isinstance(git_record, Mapping):
        raise ValueError("training-run provenance lacks git state")
    training_head = str(git_record.get("head", ""))
    if len(training_head) != 40 or any(
        character not in "0123456789abcdef" for character in training_head
    ):
        raise ValueError("training-run provenance has an invalid git HEAD")
    status_short = str(git_record.get("status_short", ""))

    encoder_record = critical.get("encoder_source")
    current_encoder_sha256 = _sha256_file(ENCODER_SOURCE_PATH)
    if encoder_record is not None:
        if not isinstance(encoder_record, Mapping) or current_encoder_sha256 != str(
            encoder_record.get("sha256", "")
        ):
            raise ValueError("training critical source changed: encoder_source")
        encoder_verification = {
            "mode": "explicit_critical_input",
            "current_sha256": current_encoder_sha256,
            "checkpoint_sha256": str(encoder_record["sha256"]),
        }
    else:
        if _status_mentions_path(status_short, ENCODER_SOURCE_GIT_PATH):
            raise ValueError("encoder source was dirty in the recorded training status")
        try:
            training_blob_sha256 = _git_blob_sha256(
                training_head, ENCODER_SOURCE_GIT_PATH
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError("cannot recover encoder source at training HEAD") from exc
        if current_encoder_sha256 != training_blob_sha256:
            raise ValueError("encoder source differs from the recorded training HEAD")
        encoder_verification = {
            "mode": "clean_git_blob_fallback_for_legacy_critical_input_list",
            "repository_path": ENCODER_SOURCE_GIT_PATH,
            "training_head": training_head,
            "current_sha256": current_encoder_sha256,
            "training_git_blob_sha256": training_blob_sha256,
            "recorded_training_status_dirty": False,
        }
    return {
        "schema": "lewm_go2_training_source_provenance_verification_v1",
        "training_run_provenance_content_sha256": declared,
        "critical_sources": verified,
        "encoder_source": encoder_verification,
        "allowed_counterfactual_source_changes": sorted(
            allowed_counterfactual_source_changes
        ),
    }


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _frozen_cell_square_counterfactual(
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_config = checkpoint.get("model_config")
    if not isinstance(parent_config, Mapping) or parent_config.get(
        "bev_lift_type"
    ) != PROJECTIVE_COLUMN_ATTENTION_LIFT:
        raise ValueError(
            "cell-square counterfactual requires a center-projective parent"
        )
    if checkpoint.get("projective_query_support") is not None:
        raise ValueError("counterfactual parent already declares query support")
    support = build_projective_query_support_contract(manifest)
    evaluated_config = dict(parent_config)
    evaluated_config["bev_lift_type"] = PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
    evaluated_config["projective_output_cell_size_m"] = float(
        support["output_cell_size_m"]
    )
    counterfactual_output_contract = dict(checkpoint["occupancy_output_contract"])
    counterfactual_output_contract[
        "projective_query_support_contract_sha256"
    ] = support["contract_sha256"]
    validate_projective_query_support_binding(
        model_config=evaluated_config,
        projective_query_support=support,
        dataset_manifest=manifest,
        occupancy_output_contract=counterfactual_output_contract,
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("counterfactual parent model state is malformed")
    core = {
        "schema": "lewm_go2_frozen_projective_geometry_counterfactual_v1",
        "classification": "development_diagnostic_only",
        "parent_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT,
        "evaluated_lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        "projective_query_support": support,
        "parent_model_state_sha256": _state_dict_sha256(state),
        "learned_state_unchanged": True,
        "probability_calibration_unchanged": True,
        "traversability_thresholds_unchanged": True,
        "checkpoint_emitted_or_mutated": False,
        "training_performed": False,
        "g2_eligible": False,
        "runtime_promotion_eligible": False,
    }
    return (
        evaluated_config,
        support,
        {**core, "content_sha256": canonical_json_sha256(core)},
    )


def _frame_records(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    selected = {role: [] for role in sorted(ALLOWED_ROLES)}
    metadata_counts: dict[str, int] = {}
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if scene_id not in assignments:
            raise ValueError(f"row scene has no role assignment: {scene_id!r}")
        role = str(assignments[scene_id])
        direct_role = str(row.get("dataset_role", ""))
        if direct_role != role:
            raise ValueError(f"row role disagrees with scene role: {scene_id}")
        metadata_counts[role] = metadata_counts.get(role, 0) + 1
        if role not in ALLOWED_ROLES:
            continue
        for side in ("current", "next"):
            selected[role].append(
                {
                    "role": role,
                    "family": str(row["family"]),
                    "scene_id": scene_id,
                    "global_row": int(row["global_row"]),
                    "side": side,
                    "image_path": str(row[f"{side}_image_path"]),
                    "image_sha256": str(row[f"{side}_image_sha256"]),
                    "label_shard_path": str(row["label_shard_path"]),
                    "label_shard_sha256": str(row["label_shard_sha256"]),
                    "label_shard_row": int(row["label_shard_row"]),
                }
            )
    for role, records in selected.items():
        records.sort(key=lambda item: (item["global_row"], item["side"]))
        if not records:
            raise ValueError(f"diagnostic role has no frame records: {role}")
    return selected, dict(sorted(metadata_counts.items()))


def _maximum_mismatch_count(keys: Sequence[str]) -> tuple[int, bool, int]:
    values = tuple(str(value) for value in keys)
    if not values:
        raise ValueError("control permutation requires at least one record")
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    maximum_multiplicity = max(counts.values())
    unavoidable_matches = max(0, 2 * maximum_multiplicity - len(values))
    return len(values) - unavoidable_matches, unavoidable_matches == 0, maximum_multiplicity


def _joint_zero_match_permutation(
    image_keys: Sequence[str],
    scene_keys: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[np.ndarray, str]:
    """Find a deterministic permutation with no image or scene equality.

    The fast scene-multiset rotation handles the benchmark.  The exact
    Hopcroft-Karp fallback prevents a corpus with cross-scene duplicate images
    from being rejected merely because that first optimal rotation picked the
    wrong cross-scene source.
    """

    images = tuple(str(value) for value in image_keys)
    scenes = tuple(str(value) for value in scene_keys)
    if len(images) != len(scenes):
        raise ValueError("image and scene control keys differ in length")
    count = len(images)
    primary = deterministic_maximum_mismatch_permutation(
        scenes,
        seed=int(seed),
        namespace=f"{namespace}:scene-multiset-rotation",
    )
    if all(
        images[index] != images[int(primary[index])]
        and scenes[index] != scenes[int(primary[index])]
        for index in range(count)
    ):
        return primary, "scene_multiset_rotation"

    source_order = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"{int(seed)}\0{namespace}\0matching-source\0{index}".encode("utf-8")
        ).hexdigest(),
    )
    pair_left = np.full(count, -1, dtype=np.int64)
    pair_right = np.full(count, -1, dtype=np.int64)
    distance = np.full(count, -1, dtype=np.int64)

    def allowed(target: int, source: int) -> bool:
        return images[target] != images[source] and scenes[target] != scenes[source]

    def bfs() -> bool:
        queue = [index for index in range(count) if pair_left[index] < 0]
        for index in range(count):
            distance[index] = 0 if pair_left[index] < 0 else -1
        found_free_source = False
        cursor = 0
        while cursor < len(queue):
            target = queue[cursor]
            cursor += 1
            for source in source_order:
                if not allowed(target, source):
                    continue
                paired_target = int(pair_right[source])
                if paired_target < 0:
                    found_free_source = True
                elif distance[paired_target] < 0:
                    distance[paired_target] = distance[target] + 1
                    queue.append(paired_target)
        return found_free_source

    def dfs(target: int) -> bool:
        for source in source_order:
            if not allowed(target, source):
                continue
            paired_target = int(pair_right[source])
            if paired_target < 0 or (
                distance[paired_target] == distance[target] + 1
                and dfs(paired_target)
            ):
                pair_left[target] = source
                pair_right[source] = target
                return True
        distance[target] = -1
        return False

    matched = 0
    while bfs():
        for target in range(count):
            if pair_left[target] < 0 and dfs(target):
                matched += 1
    if matched != count:
        raise ValueError(
            "dataset cannot achieve a joint zero-same-image/zero-same-scene "
            f"role-global control: matched={matched}/{count}"
        )
    return pair_left, "exact_joint_bipartite_matching_fallback"


def _attach_role_global_controls(
    records: list[dict[str, Any]],
    *,
    role: str,
    seed: int,
) -> dict[str, Any]:
    """Attach a cross-scene control image to every target frame in one role."""

    image_keys = tuple(str(record["image_sha256"]) for record in records)
    scene_keys = tuple(str(record["scene_id"]) for record in records)
    transition_keys = tuple(
        f"{record['scene_id']}\0{int(record['global_row'])}" for record in records
    )
    image_theoretical, image_zero_feasible, image_maximum = _maximum_mismatch_count(
        image_keys
    )
    scene_theoretical, scene_zero_feasible, scene_maximum = _maximum_mismatch_count(
        scene_keys
    )
    transition_theoretical, transition_zero_feasible, transition_maximum = (
        _maximum_mismatch_count(transition_keys)
    )
    if not (image_zero_feasible and scene_zero_feasible and transition_zero_feasible):
        raise ValueError(
            f"{role} cannot support zero-pair role-global controls: "
            f"image={image_zero_feasible} scene={scene_zero_feasible} "
            f"transition={transition_zero_feasible}"
        )

    permutation, assignment_method = _joint_zero_match_permutation(
        image_keys,
        scene_keys,
        seed=int(seed),
        namespace=f"{role}:role-global-control",
    )
    same_image = 0
    same_scene = 0
    same_transition = 0
    for target_index, source_index_value in enumerate(permutation):
        source_index = int(source_index_value)
        target = records[target_index]
        source = records[source_index]
        target["control_image_path"] = str(source["image_path"])
        target["control_image_sha256"] = str(source["image_sha256"])
        target["control_scene_id"] = str(source["scene_id"])
        target["control_global_row"] = int(source["global_row"])
        target["control_side"] = str(source["side"])
        same_image += int(image_keys[target_index] == image_keys[source_index])
        same_scene += int(scene_keys[target_index] == scene_keys[source_index])
        same_transition += int(
            transition_keys[target_index] == transition_keys[source_index]
        )
    achieved_image = len(records) - same_image
    achieved_scene = len(records) - same_scene
    achieved_transition = len(records) - same_transition
    if achieved_scene != scene_theoretical:
        raise AssertionError("scene control assignment missed its multiset optimum")
    if same_image or same_scene or same_transition:
        raise ValueError(
            f"{role} role-global control is not disjoint: same_image={same_image} "
            f"same_scene={same_scene} same_transition={same_transition}"
        )
    if achieved_image != image_theoretical or achieved_transition != transition_theoretical:
        raise ValueError(f"{role} role-global control missed a feasible zero-match optimum")
    return {
        "schema": "lewm_go2_role_global_control_permutation_v1",
        "role": role,
        "algorithm": (
            "deterministic role-global joint zero-match assignment; image, scene, "
            "and transition maximum-mismatch optima verified"
        ),
        "assignment_method": assignment_method,
        "seed": int(seed),
        "record_count": len(records),
        "permutation_sha256": canonical_json_sha256(permutation.tolist()),
        "image": {
            "maximum_multiplicity": image_maximum,
            "theoretical_maximum_mismatches": image_theoretical,
            "achieved_mismatches": achieved_image,
            "same_hash_pair_count": same_image,
            "zero_match_feasible": image_zero_feasible,
        },
        "scene": {
            "maximum_multiplicity": scene_maximum,
            "theoretical_maximum_mismatches": scene_theoretical,
            "achieved_mismatches": achieved_scene,
            "same_scene_pair_count": same_scene,
            "zero_match_feasible": scene_zero_feasible,
        },
        "transition": {
            "maximum_multiplicity": transition_maximum,
            "theoretical_maximum_mismatches": transition_theoretical,
            "achieved_mismatches": achieved_transition,
            "same_transition_pair_count": same_transition,
            "zero_match_feasible": transition_zero_feasible,
        },
    }


def _verify_selected_artifacts(
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for role, records in records_by_role.items():
        if role not in ALLOWED_ROLES:
            raise ValueError(f"forbidden role reached artifact verifier: {role}")
        shards: dict[str, str] = {}
        images: dict[str, str] = {}
        for record in records:
            shard_path = str(record["label_shard_path"])
            shard_hash = str(record["label_shard_sha256"])
            if shards.setdefault(shard_path, shard_hash) != shard_hash:
                raise ValueError("inconsistent selected shard hash")
            for prefix in ("", "control_"):
                image_path = str(record[f"{prefix}image_path"])
                image_hash = str(record[f"{prefix}image_sha256"])
                if images.setdefault(image_path, image_hash) != image_hash:
                    raise ValueError("inconsistent selected image hash")
        for path, expected in sorted(shards.items()):
            if _sha256_file(Path(path)) != expected:
                raise ValueError(f"selected shard hash mismatch: {path}")
        for path, expected in sorted(images.items()):
            if _sha256_file(Path(path)) != expected:
                raise ValueError(f"selected image hash mismatch: {path}")
        result[role] = {
            "frame_records": len(records),
            "distinct_label_shard_files_hashed": len(shards),
            "distinct_image_files_hashed": len(images),
        }
    return result


class PhysicalFrameDataset(Dataset[dict[str, Any]]):
    """Single-frame reader with exact open-event accounting for workers=0."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        image_size: int,
        crop_fraction_xy: tuple[float, float],
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
    ) -> None:
        self.records = [dict(value) for value in records]
        self.image_size = int(image_size)
        self.crop_fraction_xy = tuple(float(value) for value in crop_fraction_xy)
        self.mean = torch.tensor(tuple(normalization_mean), dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(tuple(normalization_std), dtype=torch.float32)[:, None, None]
        self._shards: OrderedDict[str, Any] = OrderedDict()
        self.image_decode_events = 0
        self.label_access_events = 0
        self.label_shard_npz_open_events = 0
        self.opened_shard_paths: set[str] = set()
        self.opened_image_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _shard(self, path: str) -> Any:
        if path in self._shards:
            value = self._shards.pop(path)
            self._shards[path] = value
            return value
        value = np.load(path, allow_pickle=False)
        self._shards[path] = value
        self.label_shard_npz_open_events += 1
        self.opened_shard_paths.add(path)
        while len(self._shards) > 2:
            _old_path, old_value = self._shards.popitem(last=False)
            old_value.close()
        return value

    def close(self) -> None:
        for value in self._shards.values():
            value.close()
        self._shards.clear()

    def _image(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop_x, crop_y = self.crop_fraction_xy
            if crop_x < 1.0 or crop_y < 1.0:
                width = max(1, int(round(image.width * crop_x)))
                height = max(1, int(round(image.height * crop_y)))
                left = (image.width - width) // 2
                top = (image.height - height) // 2
                image = image.crop((left, top, left + width, top + height))
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        self.image_decode_events += 1
        self.opened_image_paths.add(image_path)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = str(record["image_path"])
        tensor = self._image(image_path)
        control_tensor = self._image(str(record["control_image_path"]))

        shard = self._shard(str(record["label_shard_path"]))
        shard_row = int(record["label_shard_row"])
        side = str(record["side"])
        labels = np.asarray(shard[f"{side}_labels"][shard_row], dtype=np.int64)
        mask = np.asarray(
            shard[f"{side}_supervision_mask"][shard_row], dtype=bool
        )
        observed = np.asarray(
            shard[f"{side}_observed_mask"][shard_row], dtype=bool
        )
        if np.any(observed != (mask & (labels != 0))):
            raise ValueError("physical observed mask disagrees with labels")
        self.label_access_events += 1
        return {
            "image": tensor,
            "control_image": control_tensor,
            "labels": torch.from_numpy(labels.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "image_sha256": str(record["image_sha256"]),
            "control_image_sha256": str(record["control_image_sha256"]),
            "control_scene_id": str(record["control_scene_id"]),
            "control_global_row": int(record["control_global_row"]),
            "family": str(record["family"]),
            "scene_id": str(record["scene_id"]),
            "global_row": int(record["global_row"]),
            "side": side,
        }


def _new_scope_accumulators() -> dict[str, dict[str, Any]]:
    return {
        condition: {
            "loss": empty_loss_accumulator(),
            "physical": empty_physical_accumulator(),
        }
        for condition in CONDITIONS
    }


def _merge_scope(
    scope: dict[str, dict[str, Any]],
    *,
    condition: str,
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    unknown_known_weights: Sequence[float],
    free_occupied_weights: Sequence[float],
    thresholds: TraversabilityThresholds,
) -> None:
    merge_accumulator(
        scope[condition]["loss"],
        loss_accumulator_for_batch(
            logits,
            labels,
            mask,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
        ),
    )
    merge_accumulator(
        scope[condition]["physical"],
        physical_accumulator_for_batch(
            probabilities,
            labels,
            mask,
            free_probability_min=thresholds.free_probability_min,
            occupied_probability_max=thresholds.occupied_probability_max,
            unknown_probability_max=thresholds.unknown_probability_max,
            occupied_detection_min=thresholds.occupied_detection_min,
        ),
    )


def _finalize_scope(scope: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        condition: {
            **finalize_loss_accumulator(values["loss"]),
            "frozen_calibrated_physical": finalize_physical_accumulator(
                values["physical"]
            ),
        }
        for condition, values in scope.items()
    }


def _finite_required_metric(
    metrics: Mapping[str, Any], name: str, *, context: str
) -> float:
    value = metrics.get(name)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"{context} lacks finite required metric {name}")
    return float(value)


def _validate_paired_condition_support(
    condition_metrics: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    """Fail closed unless every RGB condition uses identical nonempty targets."""

    if set(condition_metrics) != set(CONDITIONS):
        raise ValueError(f"{context} has incomplete paired conditions")
    supports = []
    for condition in CONDITIONS:
        metrics = condition_metrics[condition]
        for name in (
            "raw_joint_nll",
            "raw_hierarchical_balanced_nll",
            "raw_unknown_known_weighted_nll",
            "raw_known_free_occupied_weighted_nll",
            "raw_known_free_occupied_nll",
        ):
            _finite_required_metric(metrics, name, context=f"{context}/{condition}")
        physical = metrics.get("frozen_calibrated_physical")
        if not isinstance(physical, Mapping):
            raise ValueError(f"{context}/{condition} lacks physical metrics")
        support = {
            "cell_count": int(metrics.get("cell_count", 0)),
            "known_cell_count": int(metrics.get("known_cell_count", 0)),
            "true_free": int(physical.get("true_free", 0)),
            "true_occupied": int(physical.get("true_occupied", 0)),
            "true_unknown": int(physical.get("true_unknown", 0)),
        }
        if support["cell_count"] <= 0 or support["known_cell_count"] <= 0:
            raise ValueError(f"{context}/{condition} has empty required support")
        if support["known_cell_count"] != (
            support["true_free"] + support["true_occupied"]
        ):
            raise ValueError(f"{context}/{condition} known support is inconsistent")
        if support["cell_count"] != (
            support["known_cell_count"] + support["true_unknown"]
        ):
            raise ValueError(f"{context}/{condition} total support is inconsistent")
        supports.append(support)
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError(f"{context} paired condition supports differ")
    core = {
        "schema": "lewm_go2_paired_condition_support_v1",
        "context": context,
        "conditions": list(CONDITIONS),
        "supports_equal": True,
        **supports[0],
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _validate_alignment_support(
    transform_metrics: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    if not transform_metrics or "identity" not in transform_metrics:
        raise ValueError(f"{context} lacks required alignment transforms")
    supports = []
    for name, metrics in sorted(transform_metrics.items()):
        for metric_name in (
            "raw_joint_nll",
            "raw_hierarchical_balanced_nll",
            "raw_known_free_occupied_nll",
        ):
            _finite_required_metric(
                metrics, metric_name, context=f"{context}/{name}"
            )
        support = (
            int(metrics.get("cell_count", 0)),
            int(metrics.get("known_cell_count", 0)),
        )
        if support[0] <= 0 or support[1] <= 0:
            raise ValueError(f"{context}/{name} has empty alignment support")
        supports.append(support)
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError(f"{context} alignment transform supports differ")
    core = {
        "schema": "lewm_go2_alignment_support_v1",
        "context": context,
        "transform_count": len(transform_metrics),
        "supports_equal": True,
        "cell_count": supports[0][0],
        "known_cell_count": supports[0][1],
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _projective_geometry_buffer_record(
    attention_bias: torch.Tensor | None,
    query_visibility: torch.Tensor | None,
) -> dict[str, Any]:
    """Canonical byte hashes for non-persistent geometry regenerated from config."""

    if attention_bias is None or query_visibility is None:
        raise ValueError("projective model lacks regenerated geometry buffers")
    bias = (
        attention_bias.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.dtype("<f4"), copy=False)
    )
    visibility = (
        query_visibility.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    if not np.isfinite(bias).all() or not np.isin(visibility, (0, 1)).all():
        raise ValueError("regenerated projective geometry buffers are invalid")
    core = {
        "schema": "lewm_go2_regenerated_projective_geometry_buffers_v1",
        "attention_bias": {
            "canonical_dtype": "little_endian_float32",
            "shape": list(bias.shape),
            "sha256": hashlib.sha256(bias.tobytes(order="C")).hexdigest(),
        },
        "query_visibility": {
            "canonical_dtype": "uint8_boolean",
            "shape": list(visibility.shape),
            "sha256": hashlib.sha256(visibility.tobytes(order="C")).hexdigest(),
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _model_source_records() -> dict[str, dict[str, str]]:
    paths = {
        "diagnostic_script": Path(__file__).resolve(),
        "diagnostic_metrics": REPOSITORY_ROOT
        / "lewm/benchmarks/go2_physical_spatial_grounding.py",
        "model": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
        "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
        "calibration": REPOSITORY_ROOT / "lewm/hierarchical_probability_calibration.py",
        "traversability_metrics": REPOSITORY_ROOT
        / "lewm/benchmarks/traversability_metrics.py",
        "dataset_contract": REPOSITORY_ROOT / "lewm/datasets/go2_paired_navigation.py",
    }
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _git_snapshot() -> dict[str, Any]:
    def run_text(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--"),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "head": run_text("git", "rev-parse", "HEAD"),
        "status_short": run_text("git", "status", "--short"),
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty_diff_bytes": len(diff),
        "untracked_files_are_bound_by_explicit_source_hashes": True,
    }


def _determinism(enabled: bool) -> dict[str, Any]:
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return {
        "requested": bool(enabled),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def main(argv: Sequence[str] | None = None) -> int:
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    args = _parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    source_hashes_start = _model_source_records()
    git_start = _git_snapshot()
    deterministic_execution = _determinism(bool(args.deterministic))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = args.dataset_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    report_path = args.training_report.resolve()
    output_path = args.output.resolve()
    manifest_sha256 = _verified_expected_file_sha256(
        manifest_path,
        args.expected_dataset_manifest_sha256,
        name="dataset manifest",
    )
    checkpoint_sha256 = _verified_expected_file_sha256(
        checkpoint_path,
        args.expected_checkpoint_sha256,
        name="checkpoint",
    )
    report_sha256 = _verified_expected_file_sha256(
        report_path,
        args.expected_training_report_sha256,
        name="training report",
    )
    manifest = _read_json(manifest_path)
    assignments = _validate_dataset_manifest(manifest)
    checkpoint = _load_checkpoint(checkpoint_path)
    report = _read_json(report_path)
    _require_no_g2_checkpoint(checkpoint)
    _require_no_g2_report(report)
    _validate_checkpoint_report_dataset(
        checkpoint,
        report,
        manifest,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256=manifest_sha256,
    )
    counterfactual_record = None
    if args.frozen_cell_square_counterfactual:
        model_config, evaluated_projective_query_support, counterfactual_record = (
            _frozen_cell_square_counterfactual(checkpoint, manifest)
        )
        training_source_verification = _validate_training_source_provenance(
            checkpoint,
            allowed_counterfactual_source_changes=frozenset(
                {"model_source", "trainer_source"}
            ),
        )
    else:
        model_config = dict(checkpoint["model_config"])
        evaluated_projective_query_support = checkpoint.get(
            "projective_query_support"
        )
        training_source_verification = _validate_training_source_provenance(
            checkpoint
        )

    index_path = Path(str(manifest["index"]["path"])).resolve()
    index_sha256 = _sha256_file(index_path)
    if index_sha256 != str(manifest["index"]["sha256"]):
        raise ValueError("dataset row-index SHA-256 mismatch")
    rows = _read_rows(index_path)
    records_by_role, metadata_counts = _frame_records(rows, assignments)
    control_permutations = {
        role: _attach_role_global_controls(
            records_by_role[role], role=role, seed=int(args.seed)
        )
        for role in sorted(ALLOWED_ROLES)
    }
    artifact_verification = _verify_selected_artifacts(records_by_role)
    report_counts = report["row_counts"]
    for role in ALLOWED_ROLES:
        if int(report_counts.get(role, -1)) * 2 != len(records_by_role[role]):
            raise ValueError("diagnostic role does not match the trainer's full row role")

    source_camera = checkpoint.get("source_camera_contract")
    if not isinstance(source_camera, Mapping):
        raise ValueError("checkpoint lacks source camera contract")
    source_camera_path = Path(str(source_camera["path"])).resolve()
    if _sha256_file(source_camera_path) != str(source_camera["sha256"]):
        raise ValueError("checkpoint source camera contract hash mismatch")
    geometry_path, geometry_sha256 = _validated_geometry_contract(manifest)

    device = _resolve_device(args.device)
    model = EgomotionBevJepa(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    regenerated_geometry = _projective_geometry_buffer_record(
        model.bev_decoder.projective_attention_bias,
        model.bev_decoder.projective_query_visibility,
    )
    calibration = checkpoint.get("probability_calibration")
    if not isinstance(calibration, Mapping) or calibration.get(
        "method"
    ) != CALIBRATION_METHOD:
        raise ValueError("checkpoint lacks hierarchical probability calibration")
    calibration_parameters = validate_hierarchical_probability_calibration(calibration)
    thresholds = _validated_traversability_thresholds(checkpoint)
    objective = checkpoint["occupancy_training_objective"]
    unknown_known_weights = tuple(
        float(value) for value in objective["terms"]["unknown_vs_known"]["weights"]
    )
    free_occupied_weights = tuple(
        float(value)
        for value in objective["terms"]["free_vs_occupied_given_known"]["weights"]
    )
    normalization = checkpoint.get("image_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("checkpoint lacks image normalization")
    rectification = checkpoint.get("source_fov_rectification")
    if not isinstance(rectification, Mapping):
        raise ValueError("checkpoint lacks source FOV rectification")
    crop = tuple(float(value) for value in rectification["center_crop_fraction_xy"])
    if crop != (1.0, 1.0):
        raise ValueError("physical v04 diagnostic forbids source cropping")

    query_visibility = model.bev_decoder.projective_query_visibility
    if query_visibility is None:
        raise ValueError("projective model lacks query visibility")
    visible = query_visibility.detach().cpu().numpy().reshape(model.bev_size)
    visibility_masks = visibility_regions(visible)
    forward = np.linspace(*model.forward_range_m, model.bev_size[0], dtype=np.float64)
    left = np.linspace(*model.left_range_m, model.bev_size[1], dtype=np.float64)
    distances = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)
    distance_masks = distance_bin_masks(distances)
    near_mask = distances <= 2.0

    role_reports: dict[str, Any] = {}
    role_access: dict[str, Any] = {}
    for role in sorted(ALLOWED_ROLES):
        dataset = PhysicalFrameDataset(
            records_by_role[role],
            image_size=int(model_config["image_size"]),
            crop_fraction_xy=crop,
            normalization_mean=normalization["mean"],
            normalization_std=normalization["std"],
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        overall = _new_scope_accumulators()
        by_distance = {name: _new_scope_accumulators() for name in distance_masks}
        by_family: dict[str, dict[str, dict[str, Any]]] = {}
        by_scene: dict[str, dict[str, dict[str, Any]]] = {}
        by_visibility = {
            condition: {
                name: empty_physical_accumulator() for name in visibility_masks
            }
            for condition in CONDITIONS
        }
        alignment = {
            name: empty_loss_accumulator()
            for name in alignment_accumulators_for_batch(
                np.zeros((1, 3, *model.bev_size), dtype=np.float32),
                np.zeros((1, *model.bev_size), dtype=np.int64),
                np.ones((1, *model.bev_size), dtype=bool),
                unknown_known_weights=unknown_known_weights,
                free_occupied_weights=free_occupied_weights,
                max_shift=3,
            )
        }
        calibrated_contract_validation_frames = {condition: 0 for condition in CONDITIONS}
        model_output_frames = {condition: 0 for condition in CONDITIONS}
        for batch in loader:
            images = batch["image"].to(device)
            control_images = batch["control_image"].to(device)
            labels = batch["labels"].numpy()
            mask = batch["mask"].numpy().astype(bool, copy=False)
            mean_images = images.mean(dim=(2, 3), keepdim=True).expand_as(images)
            combined = torch.cat((images, control_images, mean_images), dim=0)
            with torch.no_grad():
                combined_logits = model.occupancy_logits(combined).float()
            logits_by_condition = dict(
                zip(CONDITIONS, combined_logits.chunk(len(CONDITIONS), dim=0), strict=True)
            )
            families = tuple(str(value) for value in batch["family"])
            unique_families = sorted(set(families))
            scenes = tuple(str(value) for value in batch["scene_id"])
            unique_scenes = sorted(set(scenes))
            for condition, logits_tensor in logits_by_condition.items():
                with torch.no_grad():
                    probabilities_tensor = hierarchical_calibrated_probabilities(
                        logits_tensor,
                        calibration_parameters,
                        class_dim=1,
                    )
                logits = logits_tensor.cpu().numpy()
                probabilities = probabilities_tensor.cpu().numpy()
                evaluate_traversability(
                    probabilities,
                    labels,
                    np.broadcast_to(distances, labels.shape),
                    thresholds=thresholds,
                    evaluation_mask=mask,
                    obstacle_range_m=2.0,
                )
                calibrated_contract_validation_frames[condition] += labels.shape[0]
                model_output_frames[condition] += labels.shape[0]
                _merge_scope(
                    overall,
                    condition=condition,
                    logits=logits,
                    probabilities=probabilities,
                    labels=labels,
                    mask=mask,
                    unknown_known_weights=unknown_known_weights,
                    free_occupied_weights=free_occupied_weights,
                    thresholds=thresholds,
                )
                for bin_name, bin_mask in distance_masks.items():
                    _merge_scope(
                        by_distance[bin_name],
                        condition=condition,
                        logits=logits,
                        probabilities=probabilities,
                        labels=labels,
                        mask=mask & bin_mask[None],
                        unknown_known_weights=unknown_known_weights,
                        free_occupied_weights=free_occupied_weights,
                        thresholds=thresholds,
                    )
                for scene in unique_scenes:
                    scene_scope = by_scene.setdefault(scene, _new_scope_accumulators())
                    sample_mask = np.asarray(
                        [value == scene for value in scenes], dtype=bool
                    )[:, None, None]
                    _merge_scope(
                        scene_scope,
                        condition=condition,
                        logits=logits,
                        probabilities=probabilities,
                        labels=labels,
                        mask=mask & sample_mask,
                        unknown_known_weights=unknown_known_weights,
                        free_occupied_weights=free_occupied_weights,
                        thresholds=thresholds,
                    )
                for family in unique_families:
                    family_scope = by_family.setdefault(family, _new_scope_accumulators())
                    sample_mask = np.asarray(
                        [value == family for value in families], dtype=bool
                    )[:, None, None]
                    _merge_scope(
                        family_scope,
                        condition=condition,
                        logits=logits,
                        probabilities=probabilities,
                        labels=labels,
                        mask=mask & sample_mask,
                        unknown_known_weights=unknown_known_weights,
                        free_occupied_weights=free_occupied_weights,
                        thresholds=thresholds,
                    )
                for region, region_mask in visibility_masks.items():
                    merge_accumulator(
                        by_visibility[condition][region],
                        physical_accumulator_for_batch(
                            probabilities,
                            labels,
                            mask & near_mask[None] & region_mask[None],
                            free_probability_min=thresholds.free_probability_min,
                            occupied_probability_max=thresholds.occupied_probability_max,
                            unknown_probability_max=thresholds.unknown_probability_max,
                            occupied_detection_min=thresholds.occupied_detection_min,
                        ),
                    )
            alignment_batch = alignment_accumulators_for_batch(
                logits_by_condition["correct_rgb"].cpu().numpy(),
                labels,
                mask,
                unknown_known_weights=unknown_known_weights,
                free_occupied_weights=free_occupied_weights,
                max_shift=3,
            )
            for name, values in alignment_batch.items():
                merge_accumulator(alignment[name], values)

        dataset.close()
        finalized_overall = _finalize_scope(overall)
        overall_support = _validate_paired_condition_support(
            finalized_overall, context=f"{role}/overall"
        )
        finalized_scenes = {
            name: _finalize_scope(scope) for name, scope in sorted(by_scene.items())
        }
        scene_reports = {
            name: {
                "condition_summaries": metrics,
                "paired_condition_support": _validate_paired_condition_support(
                    metrics, context=f"{role}/scene/{name}"
                ),
            }
            for name, metrics in finalized_scenes.items()
        }
        finalized_alignment = {
            name: finalize_loss_accumulator(values)
            for name, values in sorted(alignment.items())
        }
        alignment_support = _validate_alignment_support(
            finalized_alignment, context=f"{role}/raw_spatial_alignment"
        )
        ranked_alignment = sorted(
            finalized_alignment,
            key=lambda name: (
                float("inf")
                if finalized_alignment[name]["raw_hierarchical_balanced_nll"] is None
                else finalized_alignment[name]["raw_hierarchical_balanced_nll"],
                name,
            ),
        )
        role_reports[role] = {
            "frame_count": len(records_by_role[role]),
            "condition_summaries": finalized_overall,
            "paired_condition_support": overall_support,
            "grounding_contrast": grounding_contrast(finalized_overall),
            "distance_bins_m": {
                name: _finalize_scope(scope) for name, scope in by_distance.items()
            },
            "family_bins": {
                name: _finalize_scope(scope) for name, scope in sorted(by_family.items())
            },
            "scene_reports": scene_reports,
            "near_occupied_recall_by_center_visibility": {
                condition: {
                    name: finalize_physical_accumulator(values)
                    for name, values in regions.items()
                }
                for condition, regions in by_visibility.items()
            },
            "raw_spatial_alignment": {
                "common_core_margin_cells": 3,
                "shift_definition": (
                    "prediction source at target+(row_shift,col_shift); all shifts "
                    "use the same three-cell-trimmed target support"
                ),
                "transforms": finalized_alignment,
                "paired_transform_support": alignment_support,
                "ranked_by_raw_hierarchical_balanced_nll": ranked_alignment,
                "best_transform": ranked_alignment[0],
                "identity_transform": "identity",
            },
            "role_global_control": control_permutations[role],
            "calibrated_probability_contract_validation": {
                "contract": "lewm.benchmarks.traversability_metrics.evaluate_traversability",
                "finite_unit_interval_simplex_and_label_validation": True,
                "validated_frames_by_condition": calibrated_contract_validation_frames,
            },
        }
        expected_frames = len(records_by_role[role])
        if dataset.image_decode_events != 2 * expected_frames:
            raise RuntimeError(f"{role} did not decode exactly one target/control pair")
        if dataset.label_access_events != expected_frames:
            raise RuntimeError(f"{role} did not read exactly one label per target")
        if any(
            count != expected_frames
            for count in (
                *model_output_frames.values(),
                *calibrated_contract_validation_frames.values(),
            )
        ):
            raise RuntimeError(f"{role} condition output/support counts differ")
        role_access[role] = {
            **artifact_verification[role],
            "image_decode_events": dataset.image_decode_events,
            "target_image_decode_events": expected_frames,
            "role_global_control_image_decode_events": expected_frames,
            "label_access_events": dataset.label_access_events,
            "label_shard_npz_open_events": dataset.label_shard_npz_open_events,
            "distinct_image_files_decoded": len(dataset.opened_image_paths),
            "distinct_label_shard_files_opened": len(dataset.opened_shard_paths),
            "model_output_frames_by_condition": model_output_frames,
        }

    source_hashes_end = _model_source_records()
    if source_hashes_end != source_hashes_start:
        raise RuntimeError("diagnostic source files changed during evaluation")
    git_end = _git_snapshot()
    metadata_roles = {
        role: {
            "row_metadata_count": int(metadata_counts.get(role, 0)),
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_frames": 0,
        }
        for role in sorted(set(metadata_counts) - ALLOWED_ROLES)
    }
    access_ledger = {
        "schema": "lewm_go2_physical_spatial_grounding_access_ledger_v1",
        "row_index_metadata": {
            "path": str(index_path),
            "sha256": index_sha256,
            "row_count": len(rows),
            "all_role_metadata_unavoidably_read": True,
        },
        "evaluated_roles": role_access,
        "metadata_only_forbidden_roles": metadata_roles,
        "g2_contact": {
            "row_metadata_count": int(metadata_counts.get("g2_evaluation", 0)),
            "row_metadata_read": True,
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_frames": 0,
        },
        "probability_calibration_artifact_contact": {
            "row_metadata_count": int(
                metadata_counts.get("probability_calibration", 0)
            ),
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_frames": 0,
            "checkpoint_fitted_parameters_consumed": True,
        },
    }
    command = " ".join(shlex.quote(value) for value in invocation_argv)
    core = {
        "schema": "lewm_go2_physical_spatial_grounding_diagnostic_v1",
        "created_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "classification": (
            "frozen_weight_projective_geometry_counterfactual"
            if counterfactual_record is not None
            else "frozen_checkpoint_diagnostic"
        ),
        **(
            {"counterfactual": counterfactual_record}
            if counterfactual_record is not None
            else {}
        ),
        "invocation": {
            "argv": invocation_argv,
            "command": command,
            "resolved": {
                "dataset_manifest": str(manifest_path),
                "expected_dataset_manifest_sha256": str(
                    args.expected_dataset_manifest_sha256
                ),
                "checkpoint": str(checkpoint_path),
                "expected_checkpoint_sha256": str(args.expected_checkpoint_sha256),
                "training_report": str(report_path),
                "expected_training_report_sha256": str(
                    args.expected_training_report_sha256
                ),
                "output": str(output_path),
                "batch_size": int(args.batch_size),
                "workers": int(args.workers),
                "device": str(device),
                "seed": int(args.seed),
                "deterministic": bool(args.deterministic),
                "frozen_cell_square_counterfactual": bool(
                    args.frozen_cell_square_counterfactual
                ),
            },
        },
        "git": {"start": git_start, "end": git_end},
        "deterministic_execution": deterministic_execution,
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "expected_sha256": str(args.expected_checkpoint_sha256),
                "pre_deserialization_hash_match": True,
                "schema": CHECKPOINT_SCHEMA,
                "best_epoch": int(checkpoint.get("best_epoch", 0)),
            },
            "training_report": {
                "path": str(report_path),
                "sha256": report_sha256,
                "expected_sha256": str(args.expected_training_report_sha256),
                "pre_deserialization_hash_match": True,
                "schema": REPORT_SCHEMA,
            },
            "dataset_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "expected_sha256": str(args.expected_dataset_manifest_sha256),
                "pre_deserialization_hash_match": True,
                "schema": DATASET_SCHEMA,
            },
            "dataset_index": {"path": str(index_path), "sha256": index_sha256},
            "geometry_contract": {
                "path": str(geometry_path),
                "sha256": geometry_sha256,
            },
            "source_camera_contract": {
                "path": str(source_camera_path),
                "sha256": str(source_camera["sha256"]),
                "content_sha256": str(source_camera.get("content_sha256", "")),
            },
            "checkpoint_training_source_hashes": checkpoint.get(
                "training_run_provenance", {}
            ).get("critical_inputs"),
            "training_source_verification": training_source_verification,
            "checkpoint_projective_query_support": checkpoint.get(
                "projective_query_support"
            ),
            "evaluated_projective_query_support": (
                evaluated_projective_query_support
            ),
        },
        "diagnostic_source_hashes": source_hashes_end,
        "evaluation_contract": {
            "roles_evaluated": sorted(ALLOWED_ROLES),
            "roles_forbidden": ["probability_calibration", "g2_evaluation"],
            "checkpoint_is_frozen": True,
            "bev_lift_type": str(model_config["bev_lift_type"]),
            "frozen_weight_geometry_counterfactual": (
                counterfactual_record is not None
            ),
            "training_or_parameter_updates": False,
            "probability_calibration": "frozen_checkpoint_parameters",
            "thresholds": dict(checkpoint["traversability_thresholds"]),
            "raw_alignment_max_shift_cells": 3,
            "raw_alignment_common_core_margin_cells": 3,
            "visibility_rings": "eight_connected_exterior_rings_matching_3x3_convolutions",
            "near_obstacle_range_m": 2.0,
            "mean_rgb_control": "each image replaced by its spatial per-channel mean",
            "shuffled_rgb_control": (
                "deterministic role-global maximum-mismatch image control; zero "
                "same-image, same-scene, and same-transition pairs required"
            ),
            "g2_evaluated": False,
            "g2_shard_or_image_bytes_opened": False,
            "g2_model_outputs_computed": False,
            "promotion_eligible": False,
        },
        "access_ledger": access_ledger,
        "geometry_diagnostic": {
            "regenerated_projective_geometry_buffers": regenerated_geometry,
            "center_visible_query_count": int(visible.sum()),
            "bev_query_count": int(visible.size),
            "visibility_region_cell_counts": {
                name: int(mask.sum()) for name, mask in visibility_masks.items()
            },
            "distance_bin_cell_counts": {
                name: int(mask.sum()) for name, mask in distance_masks.items()
            },
        },
        "role_reports": role_reports,
    }
    output = {**core, "content_sha256": canonical_json_sha256(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "content_sha256": output["content_sha256"],
                "g2_evaluated": False,
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
