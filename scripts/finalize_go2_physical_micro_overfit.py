#!/usr/bin/env python3
"""Finalize two immutable, authoritative Go2 micro-overfit seed results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    AUTHORITATIVE_EXECUTION,
    FAMILIES,
    GATED_DISTANCE_BIN_NAMES,
    PANELS,
    RESULT_SCHEMA,
    ROWS_PER_FAMILY_PANEL,
    TRAIN_SCENES_PER_FAMILY,
    aggregate_two_seed_result_artifacts,
    canonical_json_sha256,
    classify_cross_arm_decision,
    fit_gate,
)


SOURCE_PATHS = {
    "contract": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py",
    "execution_contract": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_generalization_execution_contract_2026-07-09.md"
    ),
    "finalizer": Path(__file__).resolve(),
    "protocol": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md"
    ),
}
ARMS = ("patch14_8x8", "patch7_16x16")
CONDITIONS = (
    "correct_rgb",
    "role_global_shuffled_rgb",
    "same_scene_wrong_view_rgb",
)
CLASS_NAMES = ("free", "occupied", "unknown")
MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN = 1000
MIN_FAMILY_FREE_CELLS_PER_GATED_BIN = 100
EXPECTED_TRAIN_IMAGES = 2 * len(PANELS) * len(FAMILIES) * ROWS_PER_FAMILY_PANEL
EXPECTED_TRAIN_LABEL_SHARDS = len(FAMILIES) * TRAIN_SCENES_PER_FAMILY
RUNNER_SOURCE_SUFFIXES = {
    "contract": "lewm/benchmarks/go2_physical_micro_overfit.py",
    "dataset": "lewm/datasets/go2_paired_navigation.py",
    "diagnostic_dataset": "scripts/diagnose_go2_physical_spatial_grounding.py",
    "encoder": "lewm/models/encoders.py",
    "generalization_execution_contract": (
        "docs/lewm_go2_generalization_execution_contract_2026-07-09.md"
    ),
    "micro_overfit_protocol": (
        "docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md"
    ),
    "model": "lewm/models/egomotion_bev_jepa.py",
    "panel_preparer": "scripts/prepare_go2_physical_micro_overfit.py",
    "runner": "scripts/run_go2_physical_micro_overfit.py",
    "spatial_metrics": "lewm/benchmarks/go2_physical_spatial_grounding.py",
    "trainer": "scripts/train_go2_egomotion_bev_jepa.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _current_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _load_expected_json(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash before deserialization and prove the file stayed unchanged after it."""

    if not _is_sha256(expected_sha256):
        raise ValueError(f"expected result SHA-256 is malformed: {path}")
    pre_deserialization_sha256 = _sha256_file(path)
    if pre_deserialization_sha256 != expected_sha256:
        raise ValueError(f"result differs from its precommitted SHA-256: {path}")
    serialized = path.read_bytes()
    bytes_read_sha256 = hashlib.sha256(serialized).hexdigest()
    if bytes_read_sha256 != pre_deserialization_sha256:
        raise RuntimeError(f"result changed between pre-hash and read: {path}")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    post_read_sha256 = _sha256_file(path)
    if post_read_sha256 != pre_deserialization_sha256:
        raise RuntimeError(f"result changed during deserialization: {path}")
    return value, {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "pre_deserialization_sha256": pre_deserialization_sha256,
        "bytes_read_sha256": bytes_read_sha256,
        "post_read_sha256": post_read_sha256,
        "pre_deserialization_hash_match": True,
        "post_read_unchanged": True,
    }


def _validate_panel_report(report: object, *, panel_name: str) -> bool:
    record = _mapping(report, context=f"{panel_name} panel report")
    expected_frames = 2 * len(FAMILIES) * ROWS_PER_FAMILY_PANEL
    if record.get("panel") != panel_name or int(record.get("frame_count", -1)) != (
        expected_frames
    ):
        raise ValueError(f"{panel_name} panel report has the wrong identity or size")
    conditions = _mapping(record.get("conditions"), context=f"{panel_name} conditions")
    if set(conditions) != set(CONDITIONS):
        raise ValueError(f"{panel_name} panel conditions are incomplete")
    for condition in CONDITIONS:
        metrics = _mapping(
            conditions[condition], context=f"{panel_name}/{condition} metrics"
        )
        if metrics.get("raw_hierarchical_balanced_nll") is None:
            raise ValueError(f"{panel_name}/{condition} lacks its primary metric")
    families = _mapping(record.get("families"), context=f"{panel_name} families")
    if set(families) != set(FAMILIES):
        raise ValueError(f"{panel_name} does not contain all five family reports")
    recomputed_family_passes = []
    for family in FAMILIES:
        family_record = _mapping(
            families[family], context=f"{panel_name}/{family} report"
        )
        family_conditions = _mapping(
            family_record.get("conditions"),
            context=f"{panel_name}/{family} conditions",
        )
        if set(family_conditions) != set(CONDITIONS):
            raise ValueError(f"{panel_name}/{family} conditions are incomplete")
        family_correct = _mapping(
            family_conditions["correct_rgb"],
            context=f"{panel_name}/{family} correct metrics",
        )
        recomputed_family_gate = fit_gate(
            family_correct,
            cross_scene_shuffled_nll=float(
                _mapping(
                    family_conditions["role_global_shuffled_rgb"],
                    context=f"{panel_name}/{family} cross control metrics",
                )["raw_hierarchical_balanced_nll"]
            ),
            same_scene_shuffled_nll=float(
                _mapping(
                    family_conditions["same_scene_wrong_view_rgb"],
                    context=f"{panel_name}/{family} same control metrics",
                )["raw_hierarchical_balanced_nll"]
            ),
        )
        stored_family_gate = _mapping(
            family_record.get("fit_gate"), context=f"{panel_name}/{family} fit gate"
        )
        if _canonical_json(stored_family_gate) != _canonical_json(
            recomputed_family_gate
        ):
            raise ValueError(f"{panel_name}/{family} fit gate is inconsistent")
        recomputed_family_passes.append(bool(recomputed_family_gate["passes"]))
    recomputed_aggregate_gate = fit_gate(
        _mapping(conditions["correct_rgb"], context=f"{panel_name} correct metrics"),
        cross_scene_shuffled_nll=float(
            _mapping(
                conditions["role_global_shuffled_rgb"],
                context=f"{panel_name} cross control metrics",
            )["raw_hierarchical_balanced_nll"]
        ),
        same_scene_shuffled_nll=float(
            _mapping(
                conditions["same_scene_wrong_view_rgb"],
                context=f"{panel_name} same control metrics",
            )["raw_hierarchical_balanced_nll"]
        ),
    )
    stored_aggregate_gate = _mapping(
        record.get("fit_gate"), context=f"{panel_name} aggregate fit gate"
    )
    if _canonical_json(stored_aggregate_gate) != _canonical_json(
        recomputed_aggregate_gate
    ):
        raise ValueError(f"{panel_name} aggregate fit gate is inconsistent")
    access = _mapping(record.get("access"), context=f"{panel_name} access ledger")
    if int(access.get("non_train_image_opens", -1)) != 0 or int(
        access.get("non_train_label_shard_opens", -1)
    ) != 0:
        raise ValueError(f"{panel_name} records forbidden non-train access")
    return bool(recomputed_aggregate_gate["passes"]) and all(
        recomputed_family_passes
    )


def _validate_stage(
    stage: object,
    *,
    stage_name: str,
    maximum_steps: int,
    learning_rate: float,
    weight_decay: float,
) -> Mapping[str, Any]:
    payload = _mapping(stage, context=f"{stage_name} stage")
    if set(payload) != set(ARMS):
        raise ValueError(f"{stage_name} stage must contain exactly both arms")
    expected_curve_steps = list(
        range(
            AUTHORITATIVE_EXECUTION["evaluation_interval"],
            maximum_steps + 1,
            AUTHORITATIVE_EXECUTION["evaluation_interval"],
        )
    )
    for arm in ARMS:
        result = _mapping(payload[arm], context=f"{stage_name}/{arm}")
        if (
            result.get("schema") != "lewm_go2_physical_micro_overfit_stage_v1"
            or result.get("stage") != stage_name
            or result.get("arm") != arm
        ):
            raise ValueError(f"{stage_name}/{arm} has invalid stage identity")
        exact_numbers = {
            "maximum_steps": maximum_steps,
            "completed_steps": maximum_steps,
            "batch_size": AUTHORITATIVE_EXECUTION["batch_size"],
            "evaluation_interval": AUTHORITATIVE_EXECUTION["evaluation_interval"],
        }
        if any(int(result.get(name, -1)) != value for name, value in exact_numbers.items()):
            raise ValueError(f"{stage_name}/{arm} used a non-authoritative budget")
        if result.get("fixed_update_budget_consumed") is not True:
            raise ValueError(f"{stage_name}/{arm} did not consume its fixed budget")
        optimizer = _mapping(result.get("optimizer"), context=f"{stage_name}/{arm} optimizer")
        if (
            optimizer.get("name") != "AdamW"
            or float(optimizer.get("learning_rate", float("nan"))) != learning_rate
            or float(optimizer.get("weight_decay", float("nan"))) != weight_decay
            or float(optimizer.get("gradient_clip", float("nan"))) != 1.0
        ):
            raise ValueError(f"{stage_name}/{arm} optimizer differs from the protocol")
        curve = result.get("learning_curve")
        if not isinstance(curve, list):
            raise ValueError(f"{stage_name}/{arm} learning curve is incomplete")
        curve_records = [
            _mapping(
                item,
                context=f"{stage_name}/{arm} learning curve evaluation",
            )
            for item in curve
        ]
        if [int(item.get("step", -1)) for item in curve_records] != expected_curve_steps:
            raise ValueError(f"{stage_name}/{arm} learning curve is incomplete")
        recomputed_passes = []
        consecutive_passes = 0
        recomputed_consecutive_passes = []
        for item in curve_records:
            passed = _validate_panel_report(item.get("fit"), panel_name="fit")
            recomputed_passes.append(passed)
            consecutive_passes = consecutive_passes + 1 if passed else 0
            recomputed_consecutive_passes.append(consecutive_passes)
            if item.get("all_family_and_aggregate_fit_gate_pass") is not passed:
                raise ValueError(
                    f"{stage_name}/{arm} learning-curve gate flag is inconsistent"
                )
            if int(item.get("consecutive_fit_gate_passes", -1)) != consecutive_passes:
                raise ValueError(
                    f"{stage_name}/{arm} consecutive gate count is inconsistent"
                )
        first_single_fit_gate_step = next(
            (
                step
                for step, passed in zip(expected_curve_steps, recomputed_passes)
                if passed
            ),
            None,
        )
        first_three_consecutive_fit_gate_step = next(
            (
                step
                for step, count in zip(
                    expected_curve_steps, recomputed_consecutive_passes
                )
                if count >= 3
            ),
            None,
        )
        if (
            result.get("first_single_fit_gate_step")
            != first_single_fit_gate_step
            or result.get("first_three_consecutive_fit_gate_step")
            != first_three_consecutive_fit_gate_step
        ):
            raise ValueError(f"{stage_name}/{arm} first gate step is inconsistent")
        terminal = _mapping(
            result.get("terminal_fit_gate"), context=f"{stage_name}/{arm} terminal gate"
        )
        expected_terminal = expected_curve_steps[-3:]
        if terminal.get("terminal_evaluation_steps") != expected_terminal:
            raise ValueError(f"{stage_name}/{arm} terminal gate uses the wrong steps")
        terminal_passes = terminal.get("terminal_evaluation_passes")
        recomputed_terminal_passes = recomputed_passes[-3:]
        if terminal_passes != recomputed_terminal_passes:
            raise ValueError(f"{stage_name}/{arm} terminal gate is incomplete")
        terminal_value = all(recomputed_terminal_passes)
        if (
            terminal.get("passes") is not terminal_value
            or result.get("fit_gate_passed_terminal_three_evaluations")
            is not terminal_value
            or terminal.get("requires_aggregate_and_all_five_family_gates") is not True
        ):
            raise ValueError(f"{stage_name}/{arm} terminal gate is inconsistent")
        panels = _mapping(
            result.get("final_panels"), context=f"{stage_name}/{arm} final panels"
        )
        if set(panels) != set(PANELS):
            raise ValueError(f"{stage_name}/{arm} final panels are incomplete")
        for panel_name in PANELS:
            _validate_panel_report(panels[panel_name], panel_name=panel_name)
        model_config = _mapping(
            result.get("model_config"), context=f"{stage_name}/{arm} model config"
        )
        expected_patch = 14 if arm == "patch14_8x8" else 7
        if (
            int(model_config.get("image_size", -1)) != 112
            or int(model_config.get("patch_size", -1)) != expected_patch
            or model_config.get("bev_lift_type") != "projective_column_attention_v1"
        ):
            raise ValueError(f"{stage_name}/{arm} model configuration is invalid")
        if not _is_sha256(result.get("final_state_sha256")):
            raise ValueError(f"{stage_name}/{arm} lacks its final-state hash")
        transition_access = _mapping(
            result.get("transition_dataset_access"),
            context=f"{stage_name}/{arm} train access",
        )
        if int(transition_access.get("non_train_image_opens", -1)) != 0 or int(
            transition_access.get("non_train_label_shard_opens", -1)
        ) != 0:
            raise ValueError(f"{stage_name}/{arm} records forbidden non-train access")
    return payload


def _validate_runner_source_hashes(value: object) -> Mapping[str, Any]:
    hashes = _mapping(value, context="runner source hashes")
    if set(hashes) != set(RUNNER_SOURCE_SUFFIXES):
        raise ValueError("runner source provenance is incomplete")
    for name, suffix in RUNNER_SOURCE_SUFFIXES.items():
        record = _mapping(hashes[name], context=f"runner source hash {name}")
        if not str(record.get("path", "")).replace("\\", "/").endswith(suffix):
            raise ValueError(f"runner source path is invalid: {name}")
        if not _is_sha256(record.get("sha256")):
            raise ValueError(f"runner source hash is invalid: {name}")
    return hashes


def _validate_zero_contact_ledger(value: object, *, context: str) -> None:
    record = _mapping(value, context=context)
    for name in ("image_byte_opens", "label_shard_byte_opens", "model_outputs"):
        if int(record.get(name, -1)) != 0:
            raise ValueError(f"{context} records forbidden contact")


def _validate_class_counts(value: object, *, context: str) -> None:
    counts = _mapping(value, context=context)
    if set(counts) != set(CLASS_NAMES) or any(
        int(counts.get(name, 0)) <= 0 for name in CLASS_NAMES
    ):
        raise ValueError(f"{context} must contain nonzero support for every class")


def _validate_gated_distance_support(
    value: object, *, minimum: int, context: str
) -> None:
    counts = _mapping(value, context=context)
    if any(
        int(counts.get(name, -1)) < minimum for name in GATED_DISTANCE_BIN_NAMES
    ):
        raise ValueError(f"{context} is below the precommitted gated-bin threshold")


def _require_recomputed_decision(
    faithful: Mapping[str, Any],
    ceiling: Mapping[str, Any] | None,
    stored: object,
    *,
    seed: int,
) -> dict[str, Any]:
    recomputed = classify_cross_arm_decision(faithful, ceiling, seed=seed)
    stored_mapping = _mapping(stored, context="stored decision")
    if _canonical_json(stored_mapping) != _canonical_json(recomputed):
        raise ValueError("stored cross-arm decision differs from recomputed decision")
    return recomputed


def _validate_authoritative_result(
    artifact: Mapping[str, Any], *, expected_seed: int
) -> dict[str, Any]:
    if artifact.get("schema") != RESULT_SCHEMA:
        raise ValueError("finalizer accepts only authoritative micro-overfit result v1")
    if artifact.get("authoritative") is not True or artifact.get("promotion_eligible") is not True:
        raise ValueError("non-authoritative or non-promotable micro-overfit result")
    core = dict(artifact)
    declared_content_sha256 = str(core.pop("content_sha256", ""))
    if not _is_sha256(declared_content_sha256) or canonical_json_sha256(core) != declared_content_sha256:
        raise ValueError("micro-overfit result content hash mismatch")

    execution = _mapping(artifact.get("execution"), context="execution provenance")
    expected_execution = {
        "batch_size": AUTHORITATIVE_EXECUTION["batch_size"],
        "faithful_steps": AUTHORITATIVE_EXECUTION["faithful_steps"],
        "ceiling_steps": AUTHORITATIVE_EXECUTION["ceiling_steps"],
        "evaluation_interval": AUTHORITATIVE_EXECUTION["evaluation_interval"],
    }
    if (
        execution.get("authoritative") is not True
        or execution.get("promotion_eligible") is not True
        or execution.get("non_authoritative_smoke") is not False
        or any(int(execution.get(name, -1)) != value for name, value in expected_execution.items())
    ):
        raise ValueError("result execution is not the exact authoritative protocol")
    determinism = _mapping(execution.get("determinism"), context="determinism provenance")
    if int(determinism.get("seed", -1)) != expected_seed:
        raise ValueError("result seed does not match its finalizer input slot")

    invocation = _mapping(artifact.get("invocation"), context="invocation provenance")
    resolved = _mapping(invocation.get("resolved"), context="resolved invocation")
    if (
        int(resolved.get("seed", -1)) != expected_seed
        or resolved.get("non_authoritative_smoke") is not False
        or any(int(resolved.get(name, -1)) != value for name, value in expected_execution.items())
    ):
        raise ValueError("resolved invocation differs from the authoritative protocol")

    inputs = _mapping(artifact.get("inputs"), context="result inputs")
    panel = _mapping(inputs.get("panel_manifest"), context="panel provenance")
    for name in ("sha256", "expected_sha256", "content_sha256"):
        if not _is_sha256(panel.get(name)):
            raise ValueError(f"panel provenance lacks a valid {name}")
    if (
        panel.get("sha256") != panel.get("expected_sha256")
        or panel.get("pre_deserialization_hash_match") is not True
        or not str(panel.get("path", ""))
    ):
        raise ValueError("panel provenance is not content-addressed")

    contract = _mapping(artifact.get("contract"), context="runner contract")
    if (
        contract.get("authoritative") is not True
        or contract.get("promotion_eligible") is not True
        or contract.get("calibration_fitted_or_applied") is not False
        or contract.get("threshold_search_performed") is not False
        or contract.get("equal_samples_and_fixed_updates_between_arms") is not True
        or set(_mapping(contract.get("arms"), context="runner arms")) != set(ARMS)
        or tuple(contract.get("families", ())) != tuple(FAMILIES)
        or tuple(contract.get("panels", ())) != tuple(PANELS)
    ):
        raise ValueError("runner contract is incomplete or non-authoritative")

    initialization = _mapping(artifact.get("initialization"), context="initialization")
    if (
        initialization.get("schema") != "lewm_go2_micro_overfit_shared_initialization_v1"
        or int(initialization.get("seed", -1)) != expected_seed
        or initialization.get("query_visibility_equal_before_shared_initialization_copy") is not True
        or initialization.get("input_image_size_equal") is not True
        or initialization.get("normalized_attention_sigma_equal") is not True
    ):
        raise ValueError("matched initialization provenance is incomplete")

    support = _mapping(
        artifact.get("post_selection_support_audit"), context="support audit"
    )
    if set(support) != set(PANELS):
        raise ValueError("post-selection support audit is incomplete")
    for panel_name in PANELS:
        panel_support = _mapping(support[panel_name], context=f"{panel_name} support")
        if (
            panel_support.get("asserted_after_label_independent_selection") is not True
            or panel_support.get("failure_policy") != "abort_without_reselection"
            or tuple(panel_support.get("distance_bins_gated", ()))
            != tuple(GATED_DISTANCE_BIN_NAMES)
            or int(
                panel_support.get(
                    "minimum_aggregate_free_cells_per_gated_bin", -1
                )
            )
            != MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN
            or int(
                panel_support.get(
                    "minimum_per_family_free_cells_per_gated_bin", -1
                )
            )
            != MIN_FAMILY_FREE_CELLS_PER_GATED_BIN
            or panel_support.get("optimizer_indexes_only_selected_fit_rows")
            is not True
        ):
            raise ValueError(f"{panel_name} support audit is incomplete")
        _validate_class_counts(
            panel_support.get("class_counts"),
            context=f"{panel_name} aggregate class support",
        )
        _validate_gated_distance_support(
            panel_support.get("distance_free_support"),
            minimum=MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN,
            context=f"{panel_name} aggregate distance support",
        )
        family_support = _mapping(
            panel_support.get("family_support"),
            context=f"{panel_name} family support",
        )
        if set(family_support) != set(FAMILIES):
            raise ValueError(f"{panel_name} family support is incomplete")
        for family in FAMILIES:
            family_record = _mapping(
                family_support[family],
                context=f"{panel_name}/{family} support",
            )
            _validate_class_counts(
                family_record.get("class_counts"),
                context=f"{panel_name}/{family} class support",
            )
            _validate_gated_distance_support(
                family_record.get("distance_free_support"),
                minimum=MIN_FAMILY_FREE_CELLS_PER_GATED_BIN,
                context=f"{panel_name}/{family} distance support",
            )

    stages = _mapping(artifact.get("stages"), context="result stages")
    if set(stages) != {"production_faithful", "ceiling_optimizer"}:
        raise ValueError("result stages are incomplete")
    faithful = _validate_stage(
        stages["production_faithful"],
        stage_name="production_faithful",
        maximum_steps=AUTHORITATIVE_EXECUTION["faithful_steps"],
        learning_rate=2e-4,
        weight_decay=1e-4,
    )
    ceiling_raw = stages["ceiling_optimizer"]
    ceiling = (
        None
        if ceiling_raw is None
        else _validate_stage(
            ceiling_raw,
            stage_name="ceiling_optimizer",
            maximum_steps=AUTHORITATIVE_EXECUTION["ceiling_steps"],
            learning_rate=1e-3,
            weight_decay=0.0,
        )
    )
    recomputed = _require_recomputed_decision(
        faithful,
        ceiling,
        artifact.get("cross_arm_decision"),
        seed=expected_seed,
    )

    verification = _mapping(
        artifact.get("artifact_verification"), context="artifact verification"
    )
    if (
        int(verification.get("distinct_train_images_hashed", -1))
        != EXPECTED_TRAIN_IMAGES
        or int(verification.get("distinct_train_label_shards_hashed", -1))
        != EXPECTED_TRAIN_LABEL_SHARDS
        or int(verification.get("non_train_images_hashed", -1)) != 0
        or int(verification.get("non_train_label_shards_hashed", -1)) != 0
    ):
        raise ValueError("artifact verification differs from the frozen train panel")
    access = _mapping(artifact.get("access_ledger"), context="access ledger")
    if (
        access.get("runner_input_contains_only_train_rows") is not True
        or int(access.get("train_image_paths_available", -1))
        != EXPECTED_TRAIN_IMAGES
        or int(access.get("train_label_shard_paths_available", -1))
        != EXPECTED_TRAIN_LABEL_SHARDS
    ):
        raise ValueError("runner input isolation is not established")
    for role in ("checkpoint_selection", "probability_calibration", "g2_evaluation"):
        _validate_zero_contact_ledger(access.get(role), context=f"{role} ledger")
    reconciliation = _mapping(
        access.get("train_role_event_reconciliation"),
        context="train access reconciliation",
    )
    if (
        reconciliation.get("schema")
        != "lewm_go2_physical_micro_overfit_train_access_reconciliation_v1"
        or reconciliation.get("events_reconciled") is not True
        or int(reconciliation.get("non_train_image_byte_open_events", -1)) != 0
        or int(reconciliation.get("non_train_label_shard_byte_open_events", -1))
        != 0
        or int(reconciliation.get("non_train_model_output_frames", -1)) != 0
    ):
        raise ValueError("train access events are not reconciled")
    _mapping(artifact.get("git"), context="git provenance")
    source_hashes = _validate_runner_source_hashes(artifact.get("source_hashes"))
    return {
        "content_sha256": declared_content_sha256,
        "panel_manifest": panel,
        "contract": contract,
        "source_hashes": source_hashes,
        "recomputed_decision": recomputed,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-20260710-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260710-result-sha256", required=True)
    parser.add_argument("--seed-20260711-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260711-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; finalization artifacts are immutable")
    if args.seed_20260710_result.resolve() == args.seed_20260711_result.resolve():
        parser.error("the two seed inputs must be distinct files")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_start = _current_source_hashes()
    paths = {
        20260710: args.seed_20260710_result.resolve(),
        20260711: args.seed_20260711_result.resolve(),
    }
    expected_hashes = {
        20260710: str(args.expected_seed_20260710_result_sha256),
        20260711: str(args.expected_seed_20260711_result_sha256),
    }
    payloads: dict[int, dict[str, Any]] = {}
    input_ledgers: dict[int, dict[str, Any]] = {}
    validations: dict[int, dict[str, Any]] = {}
    for seed in (20260710, 20260711):
        payloads[seed], input_ledgers[seed] = _load_expected_json(
            paths[seed], expected_sha256=expected_hashes[seed]
        )
        validations[seed] = _validate_authoritative_result(
            payloads[seed], expected_seed=seed
        )

    for field in ("panel_manifest", "contract", "source_hashes"):
        if _canonical_json(validations[20260710][field]) != _canonical_json(
            validations[20260711][field]
        ):
            raise ValueError(f"two seed results disagree on common {field} provenance")

    aggregation = aggregate_two_seed_result_artifacts(
        payloads[20260710], payloads[20260711]
    )
    for seed in (20260710, 20260711):
        final_sha256 = _sha256_file(paths[seed])
        if final_sha256 != input_ledgers[seed]["pre_deserialization_sha256"]:
            raise RuntimeError(f"seed {seed} result changed during finalization")
        input_ledgers[seed]["post_validation_sha256"] = final_sha256
        input_ledgers[seed]["post_validation_unchanged"] = True
        input_ledgers[seed]["content_sha256"] = validations[seed]["content_sha256"]
        input_ledgers[seed]["decision_recomputed_exactly"] = True

    source_end = _current_source_hashes()
    if source_end != source_start:
        raise RuntimeError("finalizer source or protocol changed during finalization")
    core = {
        "schema": "lewm_go2_physical_micro_overfit_finalization_artifact_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_inputs_only": True,
        "input_hash_verification": [input_ledgers[seed] for seed in (20260710, 20260711)],
        "common_provenance_validated": {
            "panel_manifest": True,
            "runner_contract": True,
            "runner_source_hashes": True,
            "micro_overfit_protocol": True,
            "generalization_execution_contract": True,
        },
        "stored_decisions_recomputed_from_stages": True,
        "aggregation": aggregation,
        "source_hashes": source_end,
        "patch7_full_train_candidate_licensed": aggregation[
            "patch7_full_train_candidate_licensed"
        ],
    }
    result = {**core, "content_sha256": canonical_json_sha256(core)}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": _sha256_file(output),
                "content_sha256": result["content_sha256"],
                "patch7_full_train_candidate_licensed": result[
                    "patch7_full_train_candidate_licensed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
