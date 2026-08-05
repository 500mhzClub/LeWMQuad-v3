from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.benchmarks.go2_physical_micro_overfit import (
    AUTHORITATIVE_EXECUTION,
    FAMILIES,
    PANELS,
    ROWS_PER_FAMILY_PANEL,
    SMOKE_EXECUTION,
    SMOKE_RESULT_SCHEMA,
    canonical_json_sha256,
    classify_cross_arm_decision,
    fit_gate,
)
from scripts import finalize_go2_physical_micro_overfit as finalizer
from scripts import run_go2_physical_micro_overfit as runner


def _runner_args(tmp_path: Path) -> list[str]:
    return [
        "--panel-manifest",
        str(tmp_path / "panel.json"),
        "--expected-panel-sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "result.json"),
    ]


def _with_content_hash(core: dict) -> dict:
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _non_expressive_stage() -> dict:
    return {
        arm: {
            "fit_gate_passed_terminal_three_evaluations": False,
            "final_panels": {"fit": {"conditions": {}}},
        }
        for arm in ("patch14_8x8", "patch7_16x16")
    }


def _gate_metrics(*, passes: bool) -> dict:
    score = 0.995 if passes else 0.5
    return {
        "raw_hierarchical_balanced_nll": 0.01 if passes else 0.2,
        "unknown_known_balanced_accuracy": score,
        "free_occupied_balanced_accuracy": score,
        "class_recall": {
            "unknown": score,
            "free": score,
            "occupied": score,
        },
        "distance_free_recall": {
            "1.0_to_2.0": score,
            "2.0_to_3.0": score,
            "3.0_plus": score,
        },
    }


def _gate_panel(*, panel_name: str, passes: bool) -> dict:
    correct = _gate_metrics(passes=passes)
    control = {"raw_hierarchical_balanced_nll": 0.5}
    gate = fit_gate(
        correct,
        cross_scene_shuffled_nll=0.5,
        same_scene_shuffled_nll=0.5,
    )
    assert gate["passes"] is passes
    conditions = {
        "correct_rgb": copy.deepcopy(correct),
        "role_global_shuffled_rgb": copy.deepcopy(control),
        "same_scene_wrong_view_rgb": copy.deepcopy(control),
    }
    return {
        "panel": panel_name,
        "frame_count": 2 * len(FAMILIES) * ROWS_PER_FAMILY_PANEL,
        "conditions": copy.deepcopy(conditions),
        "families": {
            family: {
                "conditions": copy.deepcopy(conditions),
                "fit_gate": copy.deepcopy(gate),
            }
            for family in FAMILIES
        },
        "fit_gate": copy.deepcopy(gate),
        "access": {
            "non_train_image_opens": 0,
            "non_train_label_shard_opens": 0,
        },
    }


def _authoritative_stage(*, curve_passes: list[bool]) -> dict:
    interval = AUTHORITATIVE_EXECUTION["evaluation_interval"]
    maximum_steps = AUTHORITATIVE_EXECUTION["faithful_steps"]
    assert len(curve_passes) == maximum_steps // interval
    terminal_passes = curve_passes[-3:]
    terminal_value = all(terminal_passes)
    consecutive = 0
    consecutive_passes = []
    for passes in curve_passes:
        consecutive = consecutive + 1 if passes else 0
        consecutive_passes.append(consecutive)
    evaluation_steps = list(range(interval, maximum_steps + 1, interval))
    first_single = next(
        (step for step, passes in zip(evaluation_steps, curve_passes) if passes),
        None,
    )
    first_three = next(
        (step for step, count in zip(evaluation_steps, consecutive_passes) if count >= 3),
        None,
    )
    stage = {}
    for arm, patch_size in (("patch14_8x8", 14), ("patch7_16x16", 7)):
        stage[arm] = {
            "schema": "lewm_go2_physical_micro_overfit_stage_v1",
            "stage": "production_faithful",
            "arm": arm,
            "model_config": {
                "image_size": 112,
                "patch_size": patch_size,
                "bev_lift_type": "projective_column_attention_v1",
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 2e-4,
                "weight_decay": 1e-4,
                "gradient_clip": 1.0,
            },
            "maximum_steps": maximum_steps,
            "completed_steps": maximum_steps,
            "batch_size": AUTHORITATIVE_EXECUTION["batch_size"],
            "evaluation_interval": interval,
            "fixed_update_budget_consumed": True,
            "learning_curve": [
                {
                    "step": step,
                    "fit": _gate_panel(panel_name="fit", passes=passes),
                    "all_family_and_aggregate_fit_gate_pass": passes,
                    "consecutive_fit_gate_passes": consecutive_count,
                }
                for step, passes, consecutive_count in zip(
                    evaluation_steps,
                    curve_passes,
                    consecutive_passes,
                )
            ],
            "first_single_fit_gate_step": first_single,
            "first_three_consecutive_fit_gate_step": first_three,
            "terminal_fit_gate": {
                "terminal_evaluation_steps": [
                    maximum_steps - 2 * interval,
                    maximum_steps - interval,
                    maximum_steps,
                ],
                "terminal_evaluation_passes": list(terminal_passes),
                "passes": terminal_value,
                "requires_aggregate_and_all_five_family_gates": True,
            },
            "fit_gate_passed_terminal_three_evaluations": terminal_value,
            "final_panels": {
                panel_name: _gate_panel(
                    panel_name=panel_name, passes=curve_passes[-1]
                )
                for panel_name in PANELS
            },
            "final_state_sha256": "a" * 64,
            "transition_dataset_access": {
                "non_train_image_opens": 0,
                "non_train_label_shard_opens": 0,
            },
        }
    return stage


def _support_audit() -> dict:
    distance_support = {
        "0.0_to_0.5": 1000,
        "0.5_to_1.0": 1000,
        "1.0_to_2.0": 1000,
        "2.0_to_3.0": 1000,
        "3.0_plus": 1000,
    }
    family_support = {
        family: {
            "class_counts": {
                "unknown": 100,
                "free": 100,
                "occupied": 100,
            },
            "distance_free_support": {
                name: 100 for name in distance_support
            },
        }
        for family in FAMILIES
    }
    return {
        panel_name: {
            "class_counts": {
                "unknown": 1000,
                "free": 1000,
                "occupied": 1000,
            },
            "distance_free_support": copy.deepcopy(distance_support),
            "family_support": copy.deepcopy(family_support),
            "distance_bins_gated": ["1.0_to_2.0", "2.0_to_3.0", "3.0_plus"],
            "minimum_aggregate_free_cells_per_gated_bin": 1000,
            "minimum_per_family_free_cells_per_gated_bin": 100,
            "asserted_after_label_independent_selection": True,
            "failure_policy": "abort_without_reselection",
            "optimizer_indexes_only_selected_fit_rows": True,
        }
        for panel_name in PANELS
    }


def _authoritative_artifact(*, seed: int = 20260710) -> dict:
    faithful = _authoritative_stage(curve_passes=[True] * 20)
    decision = classify_cross_arm_decision(faithful, None, seed=seed)
    source_hashes = {
        name: {"path": f"/{suffix}", "sha256": "c" * 64}
        for name, suffix in finalizer.RUNNER_SOURCE_SUFFIXES.items()
    }
    core = {
        "schema": "lewm_go2_physical_micro_overfit_result_v1",
        "authoritative": True,
        "promotion_eligible": True,
        "invocation": {
            "resolved": {
                **AUTHORITATIVE_EXECUTION,
                "seed": seed,
                "non_authoritative_smoke": False,
            }
        },
        "inputs": {
            "panel_manifest": {
                "path": "/panel.json",
                "sha256": "a" * 64,
                "expected_sha256": "a" * 64,
                "content_sha256": "b" * 64,
                "pre_deserialization_hash_match": True,
            }
        },
        "execution": {
            **AUTHORITATIVE_EXECUTION,
            "authoritative": True,
            "promotion_eligible": True,
            "non_authoritative_smoke": False,
            "determinism": {"seed": seed},
        },
        "contract": {
            "authoritative": True,
            "promotion_eligible": True,
            "arms": {arm: {} for arm in ("patch14_8x8", "patch7_16x16")},
            "families": list(FAMILIES),
            "panels": list(PANELS),
            "calibration_fitted_or_applied": False,
            "threshold_search_performed": False,
            "equal_samples_and_fixed_updates_between_arms": True,
        },
        "initialization": {
            "schema": "lewm_go2_micro_overfit_shared_initialization_v1",
            "seed": seed,
            "query_visibility_equal_before_shared_initialization_copy": True,
            "input_image_size_equal": True,
            "normalized_attention_sigma_equal": True,
        },
        "post_selection_support_audit": _support_audit(),
        "stages": {
            "production_faithful": faithful,
            "ceiling_optimizer": None,
        },
        "cross_arm_decision": decision,
        "artifact_verification": {
            "distinct_train_images_hashed": 960,
            "distinct_train_label_shards_hashed": 45,
            "non_train_images_hashed": 0,
            "non_train_label_shards_hashed": 0,
        },
        "access_ledger": {
            "runner_input_contains_only_train_rows": True,
            "train_image_paths_available": 960,
            "train_label_shard_paths_available": 45,
            "train_role_event_reconciliation": {
                "schema": (
                    "lewm_go2_physical_micro_overfit_train_access_"
                    "reconciliation_v1"
                ),
                "events_reconciled": True,
                "non_train_image_byte_open_events": 0,
                "non_train_label_shard_byte_open_events": 0,
                "non_train_model_output_frames": 0,
            },
            "checkpoint_selection": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
        },
        "git": {"start": {}, "end": {}},
        "source_hashes": source_hashes,
    }
    return _with_content_hash(core)


def _rehash_artifact(artifact: dict) -> None:
    artifact.pop("content_sha256", None)
    artifact["content_sha256"] = canonical_json_sha256(artifact)


def test_runner_authoritative_mode_requires_exact_registered_execution(
    tmp_path: Path,
) -> None:
    args = runner._parse_args(_runner_args(tmp_path))
    assert args.batch_size == AUTHORITATIVE_EXECUTION["batch_size"]
    assert args.faithful_steps == AUTHORITATIVE_EXECUTION["faithful_steps"]
    assert args.ceiling_steps == AUTHORITATIVE_EXECUTION["ceiling_steps"]
    assert args.evaluation_interval == AUTHORITATIVE_EXECUTION["evaluation_interval"]

    with pytest.raises(SystemExit):
        runner._parse_args(
            [*_runner_args(tmp_path), "--faithful-steps", "100"]
        )


def test_runner_smoke_mode_allows_only_bounded_divisible_alternates(
    tmp_path: Path,
) -> None:
    defaults = runner._parse_args(
        [*_runner_args(tmp_path), "--non-authoritative-smoke"]
    )
    assert defaults.batch_size == SMOKE_EXECUTION["batch_size"]
    assert defaults.faithful_steps == SMOKE_EXECUTION["faithful_steps"]
    assert defaults.ceiling_steps == SMOKE_EXECUTION["ceiling_steps"]
    assert defaults.evaluation_interval == SMOKE_EXECUTION["evaluation_interval"]

    args = runner._parse_args(
        [
            *_runner_args(tmp_path),
            "--non-authoritative-smoke",
            "--batch-size",
            "2",
            "--faithful-steps",
            "6",
            "--ceiling-steps",
            "12",
            "--evaluation-interval",
            "2",
        ]
    )
    assert args.non_authoritative_smoke is True
    assert args.batch_size == 2

    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                *_runner_args(tmp_path),
                "--non-authoritative-smoke",
                "--faithful-steps",
                "102",
                "--ceiling-steps",
                "12",
                "--evaluation-interval",
                "2",
            ]
        )
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                *_runner_args(tmp_path),
                "--non-authoritative-smoke",
                "--faithful-steps",
                "7",
                "--ceiling-steps",
                "12",
                "--evaluation-interval",
                "2",
            ]
        )


def test_verified_input_hashes_before_and_after_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"value":1}\n')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    payload, ledger = finalizer._load_expected_json(
        path, expected_sha256=expected
    )
    assert payload == {"value": 1}
    assert ledger["pre_deserialization_hash_match"] is True
    assert ledger["post_read_unchanged"] is True
    assert ledger["pre_deserialization_sha256"] == expected

    with pytest.raises(ValueError, match="precommitted"):
        finalizer._load_expected_json(path, expected_sha256="b" * 64)


def test_verified_input_rejects_change_between_read_and_post_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"value":1}\n')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = iter((expected, "b" * 64))
    monkeypatch.setattr(finalizer, "_sha256_file", lambda _path: next(observed))
    with pytest.raises(RuntimeError, match="during deserialization"):
        finalizer._load_expected_json(path, expected_sha256=expected)


def test_finalizer_recomputes_and_exactly_compares_stored_decision() -> None:
    faithful = _non_expressive_stage()
    ceiling = _non_expressive_stage()
    decision = classify_cross_arm_decision(
        faithful, ceiling, seed=20260710
    )
    assert finalizer._require_recomputed_decision(
        faithful, ceiling, decision, seed=20260710
    ) == decision

    tampered = copy.deepcopy(decision)
    tampered["classification"] = "fabricated"
    with pytest.raises(ValueError, match="differs from recomputed"):
        finalizer._require_recomputed_decision(
            faithful, ceiling, tampered, seed=20260710
        )


def test_stage_validation_accepts_internally_consistent_recomputed_curve() -> None:
    stage = _authoritative_stage(curve_passes=[True] * 20)
    assert finalizer._validate_stage(
        stage,
        stage_name="production_faithful",
        maximum_steps=AUTHORITATIVE_EXECUTION["faithful_steps"],
        learning_rate=2e-4,
        weight_decay=1e-4,
    ) == stage


def test_stage_validation_rejects_tampered_curve_gate_flag() -> None:
    stage = _authoritative_stage(curve_passes=[True] * 20)
    stage["patch14_8x8"]["learning_curve"][-1][
        "all_family_and_aggregate_fit_gate_pass"
    ] = False
    with pytest.raises(ValueError, match="curve|gate|pass"):
        finalizer._validate_stage(
            stage,
            stage_name="production_faithful",
            maximum_steps=AUTHORITATIVE_EXECUTION["faithful_steps"],
            learning_rate=2e-4,
            weight_decay=1e-4,
        )


def test_stage_validation_rejects_tampered_terminal_pass_vector() -> None:
    stage = _authoritative_stage(curve_passes=[True] * 20)
    arm = stage["patch14_8x8"]
    arm["terminal_fit_gate"]["terminal_evaluation_passes"] = [
        False,
        True,
        True,
    ]
    arm["terminal_fit_gate"]["passes"] = False
    arm["fit_gate_passed_terminal_three_evaluations"] = False
    with pytest.raises(ValueError, match="terminal|curve|pass"):
        finalizer._validate_stage(
            stage,
            stage_name="production_faithful",
            maximum_steps=AUTHORITATIVE_EXECUTION["faithful_steps"],
            learning_rate=2e-4,
            weight_decay=1e-4,
        )


def test_nominal_authoritative_artifact_passes_defense_in_depth() -> None:
    artifact = _authoritative_artifact()
    validated = finalizer._validate_authoritative_result(
        artifact, expected_seed=20260710
    )
    assert validated["content_sha256"] == artifact["content_sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_family",
        "aggregate_class_zero",
        "family_class_zero",
        "aggregate_gated_bin_below_1000",
        "family_gated_bin_below_100",
    ),
)
def test_finalizer_rejects_tampered_support_audit(mutation: str) -> None:
    artifact = _authoritative_artifact()
    support = artifact["post_selection_support_audit"]["fit"]
    if mutation == "missing_family":
        support["family_support"].pop(FAMILIES[-1])
    elif mutation == "aggregate_class_zero":
        support["class_counts"]["occupied"] = 0
    elif mutation == "family_class_zero":
        support["family_support"][FAMILIES[0]]["class_counts"]["free"] = 0
    elif mutation == "aggregate_gated_bin_below_1000":
        support["distance_free_support"]["2.0_to_3.0"] = 999
    elif mutation == "family_gated_bin_below_100":
        support["family_support"][FAMILIES[0]]["distance_free_support"][
            "3.0_plus"
        ] = 99
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)
    _rehash_artifact(artifact)
    with pytest.raises(ValueError, match="support|threshold|incomplete"):
        finalizer._validate_authoritative_result(
            artifact, expected_seed=20260710
        )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("distinct_train_images_hashed", 959),
        ("distinct_train_label_shards_hashed", 44),
        ("non_train_images_hashed", 1),
        ("non_train_label_shards_hashed", 1),
    ),
)
def test_finalizer_rejects_wrong_artifact_verification_counts(
    field: str, tampered_value: int
) -> None:
    artifact = _authoritative_artifact()
    artifact["artifact_verification"][field] = tampered_value
    _rehash_artifact(artifact)
    with pytest.raises(ValueError, match="artifact verification|frozen train panel"):
        finalizer._validate_authoritative_result(
            artifact, expected_seed=20260710
        )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("events_reconciled", False),
        ("non_train_image_byte_open_events", 1),
        ("non_train_label_shard_byte_open_events", 1),
        ("non_train_model_output_frames", 1),
    ),
)
def test_finalizer_rejects_unreconciled_or_nontrain_access_events(
    field: str, tampered_value: int | bool
) -> None:
    artifact = _authoritative_artifact()
    reconciliation = artifact["access_ledger"][
        "train_role_event_reconciliation"
    ]
    reconciliation[field] = tampered_value
    _rehash_artifact(artifact)
    with pytest.raises(ValueError, match="access events|reconciled"):
        finalizer._validate_authoritative_result(
            artifact, expected_seed=20260710
        )


def test_finalizer_rejects_smoke_wrong_config_and_minimal_artifacts() -> None:
    smoke = _with_content_hash(
        {
            "schema": SMOKE_RESULT_SCHEMA,
            "authoritative": False,
            "promotion_eligible": False,
        }
    )
    with pytest.raises(ValueError, match="authoritative"):
        finalizer._validate_authoritative_result(smoke, expected_seed=20260710)

    wrong_execution = dict(AUTHORITATIVE_EXECUTION)
    wrong_execution["faithful_steps"] = 100
    wrong_config = _with_content_hash(
        {
            "schema": "lewm_go2_physical_micro_overfit_result_v1",
            "authoritative": True,
            "promotion_eligible": True,
            "execution": {
                **wrong_execution,
                "authoritative": True,
                "promotion_eligible": True,
                "non_authoritative_smoke": False,
                "determinism": {"seed": 20260710},
            },
        }
    )
    with pytest.raises(ValueError, match="exact authoritative protocol"):
        finalizer._validate_authoritative_result(
            wrong_config, expected_seed=20260710
        )

    minimal = _with_content_hash(
        {
            "schema": "lewm_go2_physical_micro_overfit_result_v1",
            "authoritative": True,
            "promotion_eligible": True,
            "execution": {
                **AUTHORITATIVE_EXECUTION,
                "authoritative": True,
                "promotion_eligible": True,
                "non_authoritative_smoke": False,
                "determinism": {"seed": 20260710},
            },
        }
    )
    with pytest.raises(ValueError, match="invocation provenance"):
        finalizer._validate_authoritative_result(minimal, expected_seed=20260710)


def test_finalizer_cli_requires_two_expected_hashes_and_distinct_inputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "final.json"
    args = finalizer._parse_args(
        [
            "--seed-20260710-result",
            str(first),
            "--expected-seed-20260710-result-sha256",
            "a" * 64,
            "--seed-20260711-result",
            str(second),
            "--expected-seed-20260711-result-sha256",
            "b" * 64,
            "--output",
            str(output),
        ]
    )
    assert args.expected_seed_20260710_result_sha256 == "a" * 64
    assert args.expected_seed_20260711_result_sha256 == "b" * 64

    with pytest.raises(SystemExit):
        finalizer._parse_args(
            [
                "--seed-20260710-result",
                str(first),
                "--expected-seed-20260710-result-sha256",
                "a" * 64,
                "--seed-20260711-result",
                str(first),
                "--expected-seed-20260711-result-sha256",
                "a" * 64,
                "--output",
                str(output),
            ]
        )
