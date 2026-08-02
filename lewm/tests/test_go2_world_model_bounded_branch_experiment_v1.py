from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_world_model_bounded_branch_experiment_authority_v1 as authority
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as plan_builder
from scripts import build_go2_world_model_bounded_branch_scene_panel_v1 as panel_selector
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as calibration_collector
from scripts import dev_train_temporal_jepa_scaled as scaled
from scripts import evaluate_go2_world_model_bounded_branch_experiment_v1 as evaluator
from scripts import run_go2_world_model_bounded_branch_experiment_authorized_v1 as supervisor


REPO_ROOT = Path(__file__).resolve().parents[2]


def _conditioned(latent: np.ndarray) -> np.ndarray:
    return np.concatenate([latent, [0.0, 0.0, 1.0], latent * 0.0, latent * 0.0])


def _branch(
    rank: int,
    action: int,
    *,
    tape_class: int | None = None,
    pixel_class: int | None = None,
) -> SimpleNamespace:
    tape = action if tape_class is None else tape_class
    pixel = action if pixel_class is None else pixel_class
    return SimpleNamespace(
        oracle_dense_rank=rank,
        executed_command_tape_sha256=hashlib.sha256(
            f"tape-{tape}".encode()
        ).hexdigest(),
        target_rgb_pixel_sha256=hashlib.sha256(
            f"pixel-{pixel}".encode()
        ).hexdigest(),
    )


def _latent_fixture():
    truth = {}
    for state_index in range(4):
        truth[f"train-{state_index}"] = [
            _conditioned(np.eye(9, dtype=np.float64)[action])
            for action in range(9)
        ]
    mean, scale = evaluator.fit_train_latent_standardizer_v1(truth)
    groups = [
        SimpleNamespace(
            state_id=f"eval-{index}",
            scene_id=f"scene-{index}",
            family=pilot.FAMILIES[index % len(pilot.FAMILIES)],
            branches=tuple(
                _branch(action, action) for action in range(9)
            ),
        )
        for index in range(16)
    ]
    eval_truth = {
        group.state_id: [
            _conditioned(np.eye(9, dtype=np.float64)[action])
            for action in range(9)
        ]
        for group in groups
    }
    shuffled = {
        state_id: [rows[(action + 1) % 9] for action in range(9)]
        for state_id, rows in eval_truth.items()
    }
    return groups, eval_truth, shuffled, mean, scale


def test_direct_matched_branch_metric_detects_requested_action_identity():
    groups, truth, shuffled, mean, scale = _latent_fixture()
    exact = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    wrong = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=shuffled,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    assert exact["summary"]["matched_branch_error"] == 0.0
    assert exact["summary"]["branch_margin"] > 0.0
    assert exact["summary"]["equivalence_aware_retrieval_accuracy"] == 1.0
    assert wrong["summary"]["matched_branch_error"] > 0.0
    assert wrong["summary"]["branch_margin"] < 0.0
    assert wrong["summary"]["equivalence_aware_retrieval_accuracy"] == 0.0


def test_outcome_equivalent_actions_are_not_counted_as_identity_failures():
    basis = np.eye(9, dtype=np.float64)
    train_truth = {
        f"train-{index}": [
            _conditioned(basis[0] if action in (0, 1) else basis[action])
            for action in range(9)
        ]
        for index in range(8)
    }
    mean, scale = evaluator.fit_train_latent_standardizer_v1(train_truth)
    groups = [
        SimpleNamespace(
            state_id=f"eval-{index}",
            scene_id=f"scene-{index}",
            family=pilot.FAMILIES[index % len(pilot.FAMILIES)],
            branches=tuple(
                _branch(
                    0 if action in (0, 1) else action - 1,
                    action,
                )
                for action in range(9)
            ),
        )
        for index in range(16)
    ]
    truth = {
        group.state_id: [
            _conditioned(basis[0] if action in (0, 1) else basis[action])
            for action in range(9)
        ]
        for group in groups
    }
    swapped_equivalent = {
        state_id: [rows[1], rows[0], *rows[2:]] for state_id, rows in truth.items()
    }
    result = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=swapped_equivalent,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    assert result["summary"]["matched_branch_error"] == 0.0
    assert result["summary"]["equivalence_aware_retrieval_accuracy"] == 1.0
    assert result["summary"]["separable_action_coverage"] == 1.0


def test_joint_query_gate_rejects_cross_partition_alias_despite_aggregate_support():
    basis = np.eye(9, dtype=np.float64)
    train_truth = {
        "train": [_conditioned(basis[action]) for action in range(9)]
    }
    mean, scale = evaluator.fit_train_latent_standardizer_v1(train_truth)
    group = SimpleNamespace(
        state_id="eval",
        scene_id="scene",
        family=pilot.FAMILIES[0],
        branches=tuple(
            _branch(
                0 if action < 8 else 1,
                action,
                pixel_class=0 if action == 0 else 1,
            )
            for action in range(9)
        ),
    )
    truth = {"eval": [_conditioned(basis[action]) for action in range(9)]}
    result = evaluator.direct_matched_branch_metrics_v1(
        groups=[group],
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )

    # Tape, physical, and RGB partitions all have >=2 classes, but their
    # cross-partition aliases leave only query zero universally identifiable.
    assert result["group_results"][0]["physical_oracle_class_count"] == 2
    assert result["group_results"][0]["eligible_action_ids"] == [0]
    assert result["summary"]["separable_action_coverage"] == pytest.approx(1 / 9)


def test_scene_bootstrap_preserves_equal_weights_for_all_eight_families():
    rows = []
    for family_index, family in enumerate(pilot.FAMILIES):
        scene_count = 100 if family_index == 0 else 1
        for scene_index in range(scene_count):
            rows.append(
                {
                    "scene_id": f"{family}-{scene_index}",
                    "family": family,
                    "value": 1.0 if family_index == 0 else 0.0,
                }
            )
    interval = evaluator.scene_cluster_interval_v1(
        rows, field="value", resamples=200, seed=17
    )
    assert interval["family_strata"] == 8
    assert interval["mean"] == pytest.approx(1.0 / 8.0)
    assert interval["lower_95"] == pytest.approx(1.0 / 8.0)
    assert interval["upper_95"] == pytest.approx(1.0 / 8.0)


def _physical_arm(*, progress: float, unsafe: bool, regret: float):
    return {
        "group_results": [
            {
                "state_id": f"state-{index}",
                "scene_id": f"scene-{index}",
                "family": pilot.FAMILIES[index % len(pilot.FAMILIES)],
                "physical_fell": unsafe,
                "physical_tipped": False,
                "physical_target_progress_m": progress,
                "normalized_rank_regret": regret,
            }
            for index in range(16)
        ]
    }


def _fixture_verdict(groups, truth, shuffled, mean, scale):
    direct = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    direct_shuffled = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=shuffled,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    comparisons = {
        "ceiling_vs_current": {"upper_95": -0.10},
        "forecast_vs_current": {"upper_95": -0.08},
        "forecast_vs_task_action": {"upper_95": -0.01},
        "forecast_vs_hold_blind": {"upper_95": -0.01},
        "forecast_vs_shuffled": {"upper_95": -0.01},
        "forecast_vs_random": {"upper_95": -0.01},
    }
    return evaluator.preregistered_verdict_v1(
        direct_forecast=direct,
        direct_shuffled=direct_shuffled,
        physical_arms={
            "forecast": _physical_arm(progress=0.10, unsafe=False, regret=0.1),
            "current_state_action": _physical_arm(
                progress=0.0, unsafe=False, regret=0.2
            ),
        },
        comparisons=comparisons,
        resamples=200,
        seed=7,
    )


def test_preregistered_gates_require_useful_rank_safety_and_progress_effects():
    groups, truth, shuffled, mean, scale = _latent_fixture()
    direct = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    direct_shuffled = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=shuffled,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    comparisons = {
        "ceiling_vs_current": {"upper_95": -0.10},
        "forecast_vs_current": {"upper_95": -0.08},
        "forecast_vs_task_action": {"upper_95": -0.01},
        "forecast_vs_hold_blind": {"upper_95": -0.01},
        "forecast_vs_shuffled": {"upper_95": -0.01},
        "forecast_vs_random": {"upper_95": -0.01},
    }
    arms = {
        "forecast": _physical_arm(progress=0.10, unsafe=False, regret=0.1),
        "current_state_action": _physical_arm(progress=0.0, unsafe=False, regret=0.2),
    }
    verdict, gates = evaluator.preregistered_verdict_v1(
        direct_forecast=direct,
        direct_shuffled=direct_shuffled,
        physical_arms=arms,
        comparisons=comparisons,
        resamples=200,
        seed=7,
    )
    assert verdict == "CHECKPOINT_MEASUREMENT_PASSES_PREREGISTERED_GATES"
    assert all(
        row["passed"]
        for row in gates.values()
        if row.get("applicable", True)
    )
    comparisons["forecast_vs_current"] = {"upper_95": -0.01}
    verdict, gates = evaluator.preregistered_verdict_v1(
        direct_forecast=direct,
        direct_shuffled=direct_shuffled,
        physical_arms=arms,
        comparisons=comparisons,
        resamples=200,
        seed=7,
    )
    assert verdict == "CHECKPOINT_MEASUREMENT_FAILS_PREREGISTERED_GATES"
    assert gates["physical_rank_regret"]["passed"] is False


@pytest.mark.parametrize("missing_all", [False, True])
def test_insufficient_or_missing_family_physical_separability_is_inconclusive(
    missing_all,
):
    groups, truth, shuffled, mean, scale = _latent_fixture()
    for group in groups:
        if missing_all or group.family == pilot.FAMILIES[0]:
            group.branches = tuple(
                _branch(0, action) for action in range(9)
            )
    direct = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    direct_shuffled = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=shuffled,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    comparisons = {
        "ceiling_vs_current": {"upper_95": -0.10},
        "forecast_vs_current": {"upper_95": -0.08},
        "forecast_vs_task_action": {"upper_95": -0.01},
        "forecast_vs_hold_blind": {"upper_95": -0.01},
        "forecast_vs_shuffled": {"upper_95": -0.01},
        "forecast_vs_random": {"upper_95": -0.01},
    }
    verdict, gates = evaluator.preregistered_verdict_v1(
        direct_forecast=direct,
        direct_shuffled=direct_shuffled,
        physical_arms={
            "forecast": _physical_arm(progress=0.10, unsafe=False, regret=0.1),
            "current_state_action": _physical_arm(
                progress=0.0, unsafe=False, regret=0.2
            ),
        },
        comparisons=comparisons,
        resamples=200,
        seed=7,
    )
    assert verdict == "CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA"
    assert gates["direct_discrimination_coverage"]["passed"] is False
    assert gates["direct_branch_margin"]["applicable"] is False


def test_positive_but_below_quarter_family_joint_coverage_is_inconclusive():
    groups, truth, shuffled, mean, scale = _latent_fixture()
    for group in groups:
        if group.family == pilot.FAMILIES[0]:
            group.branches = tuple(
                _branch(
                    0 if action < 8 else 1,
                    action,
                    pixel_class=0 if action == 0 else 1,
                )
                for action in range(9)
            )
    verdict, gates = _fixture_verdict(groups, truth, shuffled, mean, scale)
    measurement = gates["direct_discrimination_coverage"]["measurement"]
    assert measurement["separable_action_coverage"] > 0.25
    assert measurement["separable_action_coverage_by_family"][
        pilot.FAMILIES[0]
    ] == pytest.approx(1 / 9)
    assert len(measurement["eligible_scene_ids_by_family"][pilot.FAMILIES[0]]) == 2
    assert verdict == "CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA"


def test_both_fixed_family_scenes_need_joint_support_even_above_quarter():
    groups, truth, shuffled, mean, scale = _latent_fixture()
    selected = [group for group in groups if group.family == pilot.FAMILIES[0]]
    selected[1].branches = tuple(_branch(0, action) for action in range(9))
    verdict, gates = _fixture_verdict(groups, truth, shuffled, mean, scale)
    measurement = gates["direct_discrimination_coverage"]["measurement"]
    assert measurement["separable_action_coverage_by_family"][
        pilot.FAMILIES[0]
    ] == 0.5
    assert len(measurement["eligible_scene_ids_by_family"][pilot.FAMILIES[0]]) == 1
    assert verdict == "CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA"


def test_evaluation_contract_forbids_future_tape_input_and_adaptation():
    contract = evaluator.evaluation_contract_v1()
    assert contract["candidate_model_input"] == "requested_action_id"
    assert contract["future_executed_command_tape_usage"] == "target_and_audit_only"
    assert "hyperparameter_adaptation" in contract["does_not_authorize"]
    assert contract["per_checkpoint_output"] == "measurement_only_no_scientific_verdict"
    assert contract["verdict_rule"].startswith(
        "arm_agnostic_plain_family_usefulness_only_after_all_12_fixed_members"
    )
    panel = contract["model_panel"]
    assert panel["arms"] == list(evaluator.MODEL_ARMS)
    assert panel["primary_arm_family"] == list(evaluator.PRIMARY_PLAIN_ARMS)
    assert panel["delta_arms"] == list(evaluator.MECHANISM_CONTROL_ARMS)
    assert panel["plain_arm_family_two_sided_alpha_each"] == 0.025
    assert contract["visual_domain_prerequisite"][
        "missing_or_failed_parity_evidence"
    ] == "STOP_NO_GENERATION_AUTHORITY"
    equivalence = contract["direct_latent"]["outcome_equivalence"]
    assert equivalence["minimum_discrimination_coverage"] == 0.25
    assert equivalence["family_requirement"] == (
        "eligible_query_coverage_at_least_0.25_in_each_fixed_family"
    )
    assert equivalence["scene_requirement"] == (
        "both_fixed_evaluation_scenes_in_each_family_have_at_least_one_"
        "eligible_query"
    )


def test_direct_summary_rejects_unrecomputed_joint_signature_tamper():
    groups, truth, _, mean, scale = _latent_fixture()
    direct = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=truth,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    direct["group_results"][0]["joint_contrast_signatures_by_action"][0][
        "executed_tape_class_sha256"
    ] = direct["group_results"][0]["joint_contrast_signatures_by_action"][1][
        "executed_tape_class_sha256"
    ]
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError,
        match="direct raw row value changed",
    ):
        evaluator._recomputed_direct_summary_v1(direct)  # noqa: SLF001


def test_decoded_pixel_verification_precedes_joint_eligibility_metrics():
    from scripts import dev_probe_counterfactual_action_fidelity as probe

    evaluation_source = inspect.getsource(evaluator.evaluate_bound_model_v1)
    extraction_source = inspect.getsource(evaluator.base._extract_features)  # noqa: SLF001
    decode_source = inspect.getsource(probe.decode)
    assert evaluation_source.rfind("base._extract_features(") < (
        evaluation_source.index("direct = {")
    )
    assert "probe.decode(" in extraction_source
    assert "read_bound_rgb_bytes_v1(pilot_bundle, path)" in decode_source


def test_progression_checkpoint_identity_is_exact_and_predeclared():
    payload = {
        "schema": evaluator.PROGRESSION_SNAPSHOT_SCHEMA,
        "status": "COMPLETE",
        "development_only": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "arm": "full_delta",
        "seed": evaluator.TRAINING_SEEDS[0],
        "update": evaluator.EXPECTED_TERMINAL_UPDATE,
        "full_grid_training": True,
        "action_auxiliary_weight": 0.1,
        "metrics": {},
        "arm_state_dict": {},
        "decoder_state_dict": {},
    }
    assert evaluator.validate_progression_snapshot_metadata_v1(
        payload,
        expected_arm="full_delta",
        expected_seed=evaluator.TRAINING_SEEDS[0],
        expected_update=evaluator.EXPECTED_TERMINAL_UPDATE,
    )["arm"] == "full_delta"
    with pytest.raises(evaluator.BoundedBranchEvaluationError, match="identity"):
        evaluator.validate_progression_snapshot_metadata_v1(
            payload,
            expected_arm="masked_plain",
            expected_seed=evaluator.TRAINING_SEEDS[0],
            expected_update=evaluator.EXPECTED_TERMINAL_UPDATE,
        )
    with pytest.raises(evaluator.BoundedBranchEvaluationError, match="outside"):
        evaluator.validate_progression_snapshot_metadata_v1(
            payload,
            expected_arm="full_delta",
            expected_seed=999,
            expected_update=evaluator.EXPECTED_TERMINAL_UPDATE,
        )


def test_progression_analysis_binds_snapshots_and_proves_scene_separation(
    tmp_path, monkeypatch
):
    root = tmp_path
    output_parent = root / ".generated/dev/world_model_progression_v1"
    output = output_parent / "comparison_20260802_v1"
    output.mkdir(parents=True)
    pack = root / ".generated/dev/pack"
    pack.mkdir(parents=True)
    monkeypatch.setattr(evaluator, "REPO_ROOT", root)
    monkeypatch.setattr(evaluator, "PROGRESSION_OUTPUT_PARENT", output_parent)
    monkeypatch.setattr(evaluator, "PROGRESSION_PACK_ROOT", pack)

    predecessor = root / "predecessor.pt"
    predecessor.write_bytes(b"predecessor")
    predecessor_binding = evaluator._sha_binding(  # noqa: SLF001
        predecessor, label="synthetic predecessor"
    )
    monkeypatch.setattr(evaluator, "PROGRESSION_PREDECESSOR", predecessor)
    monkeypatch.setattr(
        evaluator.progression_analyzer,
        "EXPECTED_PREDECESSOR",
        {
            "byte_count": predecessor_binding["byte_count"],
            "sha256": predecessor_binding["sha256"],
        },
    )
    terminal_access = root / "terminal_access.json"
    terminal_access.write_text("{}\n")
    terminal_access_binding = evaluator._sha_binding(  # noqa: SLF001
        terminal_access, label="synthetic predecessor terminal access"
    )
    monkeypatch.setattr(evaluator, "PREDECESSOR_TERMINAL_ACCESS", terminal_access)
    monkeypatch.setattr(
        evaluator,
        "PREDECESSOR_TERMINAL_ACCESS_BYTE_COUNT",
        terminal_access_binding["byte_count"],
    )
    monkeypatch.setattr(
        evaluator,
        "PREDECESSOR_TERMINAL_ACCESS_SHA256",
        terminal_access_binding["sha256"],
    )
    place_manifest = root / "place_manifest.json"
    place_manifest.write_text("{}\n")
    place_manifest_binding = evaluator._sha_binding(  # noqa: SLF001
        place_manifest, label="synthetic place manifest"
    )
    monkeypatch.setattr(
        evaluator,
        "_read_predecessor_observational_scenes_v1",
        lambda binding: (
            {"observational-ancestor"},
            {
                "predecessor_terminal_access_binding": dict(binding),
                "predecessor_index_bindings": {},
                "predecessor_place_manifest_binding": place_manifest_binding,
                "predecessor_observational_scene_count": 1,
            },
        ),
    )

    validated_roles = {}
    input_roles = {}
    for role, scene in (("train", "observational-train"), ("val", "observational-val")):
        metadata_path = pack / f"{role}_metadata.json"
        metadata_raw = json.dumps(
            {"scene_ids": [scene], "families": [pilot.FAMILIES[0]]},
            sort_keys=True,
        ).encode()
        metadata_path.write_bytes(metadata_raw)
        metadata = {
            "path": metadata_path.name,
            "byte_count": len(metadata_raw),
            "sha256": hashlib.sha256(metadata_raw).hexdigest(),
        }
        role_value = {
            "row_identity_sha256": hashlib.sha256(role.encode()).hexdigest(),
            "source_rgb": {"synthetic": role},
            "index_binding": {"synthetic": role},
            "frames": {"synthetic": role},
            "actions": {"synthetic": role},
            "metadata": metadata,
        }
        validated = {
            "manifest_path": pack / "manifest.json",
            "manifest_sha256": "f" * 64,
            "role": role_value,
            "paths": {"metadata": metadata_path},
        }
        validated_roles[role] = validated
        input_roles[role] = {
            "manifest_path": str(validated["manifest_path"]),
            "manifest_sha256": validated["manifest_sha256"],
            "role": role,
            **role_value,
        }
    monkeypatch.setattr(
        scaled,
        "validate_pack_role",
        lambda selected_root, role: validated_roles[role],
    )
    monkeypatch.setattr(
        evaluator.progression_analyzer, "EXPECTED_SOURCE_BINDINGS", tuple()
    )

    seed_results = {
        str(seed): {
            "build": {},
            "decoder_pretraining_trace": [],
            "decoder_anchor_balanced_accuracy": {},
            "update_zero": {},
            "terminal": {},
            "terminal_losses": {},
            "training_trace": [],
            "terminal_core_sha256": {},
            "terminal_decoder_sha256": "d" * 64,
            "wall_seconds": 1.0,
        }
        for seed in evaluator.TRAINING_SEEDS
    }
    result = {
        "schema": evaluator.progression_analyzer.RUNNER_SCHEMA,
        "status": evaluator.progression_analyzer.RUNNER_STATUS,
        "citable_as_scientific_evidence": False,
        "protected_material_opened": False,
        "configuration": copy.deepcopy(
            evaluator.progression_analyzer.EXPECTED_CONFIGURATION
        ),
        "runtime": {},
        "inputs": {
            "predecessor": predecessor_binding,
            "pack_root": str(pack.resolve()),
            **input_roles,
        },
        "source_bindings": [],
        "seed_results": seed_results,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    result_binding = pilot.file_binding(result_path)
    snapshot_bindings = {}
    for seed in evaluator.TRAINING_SEEDS:
        snapshot_bindings[str(seed)] = {}
        for arm in evaluator.MODEL_ARMS:
            checkpoint = (
                output
                / f"seed_{seed}"
                / f"{arm}_update_{evaluator.EXPECTED_TERMINAL_UPDATE:06d}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}/{seed}".encode())
            snapshot_bindings[str(seed)][arm] = evaluator._sha_binding(  # noqa: SLF001
                checkpoint, label="synthetic checkpoint"
            )
    analysis = {
        "schema": evaluator.PROGRESSION_ANALYSIS_SCHEMA,
        "status": "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS",
        "development_only": True,
        "citable_as_world_model_usefulness_evidence": False,
        "input_result": {
            "path": result_binding["path"],
            "byte_count": result_binding["byte_count"],
            "sha256": result_binding["file_sha256"],
        },
        "configuration": copy.deepcopy(
            evaluator.progression_analyzer.EXPECTED_CONFIGURATION
        ),
        "decoder_anchor_by_seed": {},
        "contrasts": {},
        "proxy_routing": {"decision": "DELTA_PROXY_MEANINGFUL"},
        "terminal_snapshot_bindings": snapshot_bindings,
        "uncertainty_limit": "synthetic",
    }
    monkeypatch.setattr(
        evaluator.progression_analyzer,
        "analyze",
        lambda payload, result_path: copy.deepcopy(analysis),
    )
    analysis_path = output / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, sort_keys=True) + "\n")
    analysis_binding = pilot.file_binding(analysis_path)
    seed = evaluator.TRAINING_SEEDS[0]
    checkpoint = Path(snapshot_bindings[str(seed)]["full_delta"]["path"])
    _document, receipt = evaluator.load_and_validate_progression_analysis_v1(
        analysis_path,
        expected_sha256=analysis_binding["file_sha256"],
        expected_byte_count=analysis_binding["byte_count"],
        selected_checkpoint=checkpoint,
        expected_arm="full_delta",
        expected_seed=seed,
        pilot_scene_ids={"branch-eval"},
    )
    assert receipt["scene_overlap"] == []
    assert set(receipt["checkpoint_panel_bindings"]) == {
        f"{arm}/seed_{panel_seed}"
        for arm in evaluator.MODEL_ARMS
        for panel_seed in evaluator.TRAINING_SEEDS
    }
    assert {
        "observational-ancestor",
        "observational-train",
        "observational-val",
    }.issubset(receipt["observational_scene_ids"])
    with pytest.raises(evaluator.BoundedBranchEvaluationError, match="overlap"):
        evaluator.load_and_validate_progression_analysis_v1(
            analysis_path,
            expected_sha256=analysis_binding["file_sha256"],
            expected_byte_count=analysis_binding["byte_count"],
            selected_checkpoint=checkpoint,
            expected_arm="full_delta",
            expected_seed=seed,
            pilot_scene_ids={"observational-val"},
        )


def test_predecessor_scene_union_includes_place_checkpoint_selection():
    binding = evaluator._sha_binding(  # noqa: SLF001
        evaluator.PREDECESSOR_TERMINAL_ACCESS,
        label="predecessor terminal access",
    )
    scenes, receipt = evaluator._read_predecessor_observational_scenes_v1(  # noqa: SLF001
        binding
    )
    assert "place" in receipt["predecessor_index_bindings"]
    assert receipt["predecessor_place_manifest_binding"]["sha256"] == (
        evaluator.PREDECESSOR_PLACE_MANIFEST_SHA256
    )
    assert {
        "large_enclosed_maze_d78318b1e87b",
        "visual_sensor_stress_dc440a3fb679",
    }.issubset(scenes)


def _gate_fixture(status: str = "COMPLETE_PENDING_TERMINAL_REVIEW"):
    receipt_binding = {"path": "/tmp/calibration.json", "file_sha256": "1" * 64, "byte_count": 1}
    terminal_binding = {"path": "/tmp/terminal.json", "file_sha256": "2" * 64, "byte_count": 1}
    review_binding = {"path": "/tmp/review.json", "file_sha256": "3" * 64, "byte_count": 1}
    parity_prerequisites = {
        "result_binding": _fake_binding("/tmp/parity-result.json"),
        "terminal_binding": _fake_binding("/tmp/parity-terminal.json"),
        "review_binding": _fake_binding("/tmp/parity-review.json"),
    }
    receipt = {
        "schema": plan_builder.calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
        "decision": "FREEZE_PILOT_CONTRACT",
        "calibration_contract": {"excluded_scene_ids": [f"cal-{index}" for index in range(8)]},
        "resource_measurements": {"stored_rgb_png": {"total_bytes": 1000}},
        "visual_domain_parity_prerequisites": parity_prerequisites,
    }
    terminal = {
        "schema": plan_builder.calibration_supervisor.TEXTURED_V03_TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "scientific_verdict_emitted": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "authority_binding": {},
        "plan_binding": {},
        "predecessor_failure_binding": {},
        "source_commit": "a" * 40,
        "attempt_root": "/tmp/attempt",
        "wall_elapsed_seconds": 80.0,
        "wall_ceiling_seconds": 1200.0,
        "phase_receipts": [],
        "physics_result_binding": {},
        "receipt_check_binding": {},
        "calibration_receipt_binding": receipt_binding,
        "calibration_decision": "FREEZE_PILOT_CONTRACT",
        "gpu_memory_measurement": {
            "scope": "selected_device_global_vram_not_process_attributed",
            "attribution_limitation": "global",
            "vendor_id": "0x1002",
            "device_id": "0x7551",
            "used_counter_path": "/tmp/used",
            "total_counter_path": "/tmp/total",
            "sample_interval_seconds": 0.05,
            "sample_count": 2,
            "read_errors": 0,
            "baseline_used_bytes": 100,
            "peak_used_bytes": 200,
            "peak_delta_above_baseline_bytes": 100,
            "device_total_bytes": 10_000,
        },
        "failure": None,
        "terminal_reviewer": "reviewer",
        "supervisor_nonce": "4" * 64,
        "visual_domain_parity_prerequisites": parity_prerequisites,
    }
    review = {
        "schema": plan_builder.TERMINAL_REVIEW_SCHEMA,
        "status": "PASS_FREEZE_PILOT_CONTRACT",
        "authority_granted_by_this_document": False,
        "scientific_claim_granted": False,
        "terminal_binding": terminal_binding,
        "calibration_receipt_binding": receipt_binding,
        "decision": "FREEZE_PILOT_CONTRACT",
        "reviewer": {"identity": "independent", "independence_basis": "separate process"},
        "reviewed_at": "2026-08-02T00:00:00Z",
        "checks": {
            "terminal_complete": True,
            "receipt_checker_passed": True,
            "calibration_decision_passed": True,
            "gpu_sampler_passed": True,
            "wall_ceiling_passed": True,
            "no_retry_or_resume": True,
        },
        "remaining_findings": [],
    }
    return receipt, receipt_binding, terminal, terminal_binding, review, review_binding


def test_calibration_gate_rejects_consumed_terminal_failure(monkeypatch):
    monkeypatch.setattr(
        plan_builder.calibration,
        "validate_calibration_receipt_v1",
        lambda value, verify_external_bindings: dict(value),
    )
    monkeypatch.setattr(plan_builder, "_binding", lambda value, label: dict(value))
    values = _gate_fixture(status="CONSUMED_TERMINAL_FAILURE")
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="did not pass"):
        plan_builder._validate_calibration_gate(
            values[0],
            receipt_binding=values[1],
            terminal=values[2],
            terminal_binding=values[3],
            terminal_review=values[4],
            terminal_review_binding=values[5],
        )


@pytest.mark.parametrize(
    "path",
    (
        "/tmp/sealed_test.json",
        "/tmp/sealed/manifest.json",
        "/tmp/sealed_future/manifest.json",
        "/tmp/heldout/manifest.json",
        "/tmp/held_out_future/manifest.json",
    ),
)
def test_plan_builder_rejects_protected_path_names_before_open(path):
    with pytest.raises(plan_builder.BoundedBranchPlanError, match="protected"):
        plan_builder._reject_protected_path(Path(path), label="input")


def _fake_binding(path: str):
    return {"path": path, "file_sha256": "a" * 64, "byte_count": 1}


def test_exact_bounded_plan_is_32_scenes_256_states_and_2304_branches(monkeypatch):
    gate = {
        "calibration_receipt_binding": _fake_binding("/tmp/cal.json"),
        "calibration_terminal_binding": _fake_binding("/tmp/terminal.json"),
        "calibration_terminal_review_binding": _fake_binding("/tmp/review.json"),
        "excluded_scene_ids": [f"cal-{index}" for index in range(8)],
        "calibration_wall_seconds": 80.0,
        "calibration_stored_rgb_bytes": 1000,
        "calibration_gpu_baseline_used_bytes": 100,
        "calibration_gpu_peak_used_bytes": 200,
        "calibration_gpu_peak_delta_bytes": 100,
        "selected_device_total_vram_bytes": 10_000,
    }
    model_panel = {
        "progression_analysis_binding": _fake_binding("/tmp/analysis.json"),
        "training_result_binding": _fake_binding("/tmp/result.json"),
        "progression_proxy_routing": {"decision": "DELTA_PROXY_MEANINGFUL"},
        "checkpoint_panel_bindings": {},
        "model_observational_scene_ids": ["model-observational"],
        "model_observational_scene_count": 1,
        "predecessor_terminal_access_binding": {},
        "predecessor_index_bindings": {},
        "predecessor_place_manifest_binding": _fake_binding(
            "/tmp/place-manifest.json"
        ),
        "training_pack_role_bindings": {},
        "training_pack_metadata_bindings": {},
    }
    panel = []
    for role in plan_builder.ROLE_NAMES:
        for family in pilot.FAMILIES:
            for slot in range(2):
                scene_id = f"{role}-{family}-{slot}"
                panel.append({
                    "role": role,
                    "family": family,
                    "scene_id": scene_id,
                    "scene_manifest_binding": _fake_binding(f"/tmp/{scene_id}/manifest.json"),
                    "scene_genesis_binding": _fake_binding(f"/tmp/{scene_id}/genesis_scene.json"),
                    "states": [
                        {
                            "state_id": f"{scene_id}-state-{index}",
                            "history_action_ids": list(plan_builder.HISTORY_PANEL[index]),
                            "target_xy_m": [0.0, 0.0],
                        }
                        for index in range(8)
                    ],
                })
    runtime = {name: _fake_binding(f"/tmp/{name}") for name in (
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
        "python_executable_target",
        "python_environment_config",
        "eglinfo_executable",
        "vulkaninfo_executable",
    )}
    execution = {
        "backend": "vulkan",
        "policy_device": "cpu",
        "seed": 1,
        "fall_z_threshold_m": 0.2,
        "tip_threshold_rad": 1.0,
        "policy_steps_per_command_tick": 5,
        "python_invocation_path": "/tmp/python",
        "environment": dict(pilot.EXECUTION_ENVIRONMENT),
        "graphics_preflight": dict(pilot.GRAPHICS_PREFLIGHT_EXPECTATION),
    }
    monkeypatch.setattr(plan_builder, "_validate_calibration_gate", lambda *args, **kwargs: gate)
    monkeypatch.setattr(
        plan_builder,
        "_validate_model_panel_freeze",
        lambda *args, **kwargs: model_panel,
    )
    scene_freeze = {
        "scene_panel_binding": _fake_binding("/tmp/scene-panel.json"),
        "scene_panel_schema": plan_builder.PANEL_SCHEMA,
        "scene_selection_contract": {},
        "scene_corpus_manifest_bindings": [],
        "scene_inventory_unique_train_scenes": 32,
        "scene_eligible_counts_by_family": {
            family: 4 for family in pilot.FAMILIES
        },
        "scene_excluded_scene_ids_sha256": "b" * 64,
        "scene_selection_rows": [],
    }
    monkeypatch.setattr(
        plan_builder,
        "_validate_scene_panel_receipt_v1",
        lambda *args, **kwargs: (panel, scene_freeze),
    )
    visual_parity_freeze = {
        "result_binding": _fake_binding("/tmp/visual-parity-result.json"),
        "terminal_binding": _fake_binding("/tmp/visual-parity-terminal.json"),
        "review_binding": _fake_binding("/tmp/visual-parity-review.json"),
        "source_rgb_reference_binding": _fake_binding("/tmp/source-rgb.json"),
        "candidate_rgb_panel_binding": _fake_binding("/tmp/candidate-rgb.json"),
        "source_producer_lineage": {"schema": "synthetic-source-lineage"},
        "candidate_producer_lineage": {
            "schema": "synthetic-candidate-lineage"
        },
        "candidate_collector_source_binding": _fake_binding(
            "/tmp/collector.py"
        ),
        "candidate_renderer_source_binding": _fake_binding(
            "/tmp/render_replay_v03.py"
        ),
        "reference_renderer_source_binding": _fake_binding(
            "/tmp/render_replay_v03.py"
        ),
        "reference_texture_source_binding": _fake_binding("/tmp/textures.py"),
        "evaluator_source_binding": _fake_binding("/tmp/evaluator.py"),
        "selected_texture_asset_bindings_by_scene": {},
        "evidence_scene_ids": ["visual-parity-scene"],
        "comparison_contract": {},
        "thresholds": {},
        "measurements": {},
    }
    gate.update({
        "visual_domain_parity_result_binding": visual_parity_freeze[
            "result_binding"
        ],
        "visual_domain_parity_terminal_binding": visual_parity_freeze[
            "terminal_binding"
        ],
        "visual_domain_parity_review_binding": visual_parity_freeze[
            "review_binding"
        ],
    })
    monkeypatch.setattr(
        plan_builder,
        "_validate_visual_domain_parity_gate_v1",
        lambda *args, **kwargs: visual_parity_freeze,
    )
    monkeypatch.setattr(
        plan_builder,
        "_validate_candidate_render_domain_contract_v1",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(plan_builder, "_validate_runtime_contract", lambda value: (runtime, execution))
    output = plan_builder.REPO_ROOT / ".generated/dev" / f"synthetic-{uuid.uuid4().hex}"
    plan, returned_gate = plan_builder.build_plan_v1(
        attempt_id="synthetic-bounded-plan",
        output_root=output,
        scene_panel={},
        scene_panel_binding=scene_freeze["scene_panel_binding"],
        visual_domain_parity_result={},
        visual_domain_parity_result_binding={},
        visual_domain_parity_review={},
        visual_domain_parity_review_binding={},
        runtime_contract={},
        calibration_receipt={},
        calibration_receipt_binding={},
        calibration_terminal={},
        calibration_terminal_binding={},
        calibration_terminal_review={},
        calibration_terminal_review_binding={},
        progression_analysis={},
        progression_analysis_binding={},
    )
    assert returned_gate == {
        **gate,
        **model_panel,
        **scene_freeze,
        "visual_domain_parity_freeze": visual_parity_freeze,
    }
    assert plan["expected_counts"] == {
        "scenes": 32,
        "states": 256,
        "roles": {"eval": 128, "train": 128},
        "actions": 9,
        "candidate_branches": 2304,
        "sentinel_branches": 0,
        "total_branches": 2304,
        "context_frames": 768,
        "target_frames": 2304,
    }
    assert plan["visual_domain_parity_result_binding"] == visual_parity_freeze[
        "result_binding"
    ]
    assert plan["texture_asset_bindings"] == [
        pilot.file_binding(plan_builder.REPO_ROOT / relative)
        for relative in pilot.TEXTURED_V03_TEXTURE_RELATIVE_PATHS
    ]
    assert all(state["sentinel_duplicate_action_id"] is None for state in plan["states"])


def test_scene_panel_selection_is_deterministic_balanced_and_model_disjoint(
    tmp_path, monkeypatch
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setattr(panel_selector, "SCENE_CORPUS_ROOT", tmp_path)
    inventory = []
    manifest_hashes = {}
    for family in pilot.FAMILIES:
        for index in range(6):
            scene_id = f"{family}-ordinary-{index}"
            digest = hashlib.sha256(scene_id.encode()).hexdigest()
            manifest_hashes[scene_id] = digest
            scene_root = campaign / "train" / family / scene_id
            scene_root.mkdir(parents=True)
            (scene_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "family": family,
                        "split": "train",
                        "manifest_sha256": digest,
                    }
                )
                + "\n"
            )
            (scene_root / "genesis_scene.json").write_text("{}\n")
            inventory.append(
                {
                    "family": family,
                    "scene_id": scene_id,
                    "manifest_sha256": digest,
                    "campaign_root": str(campaign),
                    "relative_dir": f"train/{family}/{scene_id}",
                    "inventory_rank": hashlib.sha256(
                        f"inventory/{scene_id}".encode()
                    ).hexdigest(),
                }
            )
    corpus_bindings = [_fake_binding(str(campaign / "corpus.json"))]
    current_inventory = list(inventory)
    monkeypatch.setattr(
        panel_selector,
        "_load_inventory",
        lambda: (corpus_bindings, list(current_inventory)),
    )

    def fake_file_binding(path: Path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }

    monkeypatch.setattr(panel_selector.pilot, "file_binding", fake_file_binding)
    monkeypatch.setattr(
        panel_selector,
        "_selected_texture_asset_bindings",
        lambda manifest: {
            category: _fake_binding(
                f"/tmp/assets/textures/{category}/{manifest['scene_id']}.png"
            )
            for category in ("floor", "wall", "obstacle")
        },
    )
    excluded = {f"{pilot.FAMILIES[0]}-ordinary-0", "model-observational"}
    first = panel_selector.derive_scene_panel_v1(excluded_scene_ids=excluded)
    current_inventory.reverse()
    second = panel_selector.derive_scene_panel_v1(excluded_scene_ids=excluded)
    assert first["scenes"] == second["scenes"]
    assert len(first["scenes"]) == 32
    assert not excluded & {row["scene_id"] for row in first["scenes"]}
    for family in pilot.FAMILIES:
        rows = [row for row in first["scenes"] if row["family"] == family]
        assert [row["role"] for row in rows].count("train") == 2
        assert [row["role"] for row in rows].count("eval") == 2


def _panel_member_reports(*, plain_pass: bool = True):
    checkpoint_panel = {
        f"{arm}/seed_{seed}": {
            "path": f"/tmp/{arm}-{seed}.pt",
            "byte_count": 1,
            "sha256": hashlib.sha256(f"{arm}/{seed}".encode()).hexdigest(),
        }
        for arm in evaluator.MODEL_ARMS
        for seed in evaluator.TRAINING_SEEDS
    }
    separation = {
        "progression_analysis_binding": _fake_binding("/tmp/analysis.json"),
        "training_result_binding": _fake_binding("/tmp/result.json"),
        "progression_proxy_routing": {
            "decision": "DELTA_PROXY_NOT_MEANINGFUL"
        },
        "predecessor_place_manifest_binding": _fake_binding(
            "/tmp/place-manifest.json"
        ),
        "checkpoint_panel_bindings": checkpoint_panel,
    }
    terminal_gate = {
        "status": "PASS_FROZEN_BOUNDED_PILOT",
        "model_panel_freeze": copy.deepcopy(separation),
    }
    groups, truth, shuffled, mean, scale = _latent_fixture()
    direct_shuffled = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=shuffled,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    hold = {
        state_id: [rows[0] for _ in range(9)]
        for state_id, rows in truth.items()
    }
    direct_hold = evaluator.direct_matched_branch_metrics_v1(
        groups=groups,
        predicted_features_by_state=hold,
        true_features_by_state=truth,
        train_mean=mean,
        train_scale=scale,
    )
    arm_error = {
        "masked_plain": 0.030,
        "masked_delta": 0.015,
        "full_plain": 0.015,
        "full_delta": 0.0,
    }

    def physical_rows(regret, progress):
        return {
            "group_results": [
                {
                    "state_id": group.state_id,
                    "scene_id": group.scene_id,
                    "family": group.family,
                    "normalized_rank_regret": regret,
                    "random_expected_normalized_rank_regret": 0.5,
                    "physical_fell": False,
                    "physical_tipped": False,
                    "physical_target_progress_m": progress,
                }
                for group in groups
            ]
        }

    reports = []
    for arm in evaluator.MODEL_ARMS:
        for seed_index, seed in enumerate(evaluator.TRAINING_SEEDS):
            key = f"{arm}/seed_{seed}"
            offset = arm_error[arm] + seed_index * 0.001
            forecast_features = {
                state_id: [
                    np.concatenate([row[:9] + offset, row[9:]])
                    for row in rows
                ]
                for state_id, rows in truth.items()
            }
            direct_forecast = evaluator.direct_matched_branch_metrics_v1(
                groups=groups,
                predicted_features_by_state=forecast_features,
                true_features_by_state=truth,
                train_mean=mean,
                train_scale=scale,
            )
            direct = {
                "forecast": direct_forecast,
                "shuffled": copy.deepcopy(direct_shuffled),
                "hold_blind": copy.deepcopy(direct_hold),
            }
            candidate_regret = (
                0.1 if plain_pass or arm not in evaluator.PRIMARY_PLAIN_ARMS
                else 0.4
            )
            candidate_progress = (
                0.10 if plain_pass or arm not in evaluator.PRIMARY_PLAIN_ARMS
                else 0.0
            )
            physical = {
                "true_future_ceiling": physical_rows(0.0, 0.12),
                "current_state_action": physical_rows(0.4, 0.0),
                "task_action_only": physical_rows(0.3, 0.01),
                "forecast": physical_rows(candidate_regret, candidate_progress),
                "shuffled": physical_rows(0.35, 0.01),
                "hold_blind": physical_rows(0.32, 0.01),
            }
            pairs = {
                "ceiling_vs_current": (
                    "true_future_ceiling", "current_state_action"
                ),
                "forecast_vs_current": ("forecast", "current_state_action"),
                "forecast_vs_task_action": ("forecast", "task_action_only"),
                "forecast_vs_hold_blind": ("forecast", "hold_blind"),
                "forecast_vs_shuffled": ("forecast", "shuffled"),
            }
            interval_alpha = evaluator.checkpoint_interval_alpha_v1(arm)
            comparisons = {
                name: evaluator._paired(  # noqa: SLF001
                    physical[candidate]["group_results"],
                    physical[baseline]["group_results"],
                    field="normalized_rank_regret",
                    resamples=evaluator.DEFAULT_RESAMPLES,
                    seed=evaluator.DEFAULT_BOOTSTRAP_SEED,
                    two_sided_alpha=interval_alpha,
                )
                for name, (candidate, baseline) in pairs.items()
            }
            random_rows = [
                {
                    **row,
                    "normalized_rank_regret": row[
                        "random_expected_normalized_rank_regret"
                    ],
                }
                for row in physical["forecast"]["group_results"]
            ]
            comparisons["forecast_vs_random"] = evaluator._paired(  # noqa: SLF001
                physical["forecast"]["group_results"],
                random_rows,
                field="normalized_rank_regret",
                resamples=evaluator.DEFAULT_RESAMPLES,
                seed=evaluator.DEFAULT_BOOTSTRAP_SEED,
                two_sided_alpha=interval_alpha,
            )
            status, gates = evaluator.preregistered_verdict_v1(
                direct_forecast=direct_forecast,
                direct_shuffled=direct_shuffled,
                physical_arms=physical,
                comparisons=comparisons,
                resamples=evaluator.DEFAULT_RESAMPLES,
                seed=evaluator.DEFAULT_BOOTSTRAP_SEED,
                two_sided_alpha=interval_alpha,
            )
            reports.append(
                {
                    "schema": evaluator.REPORT_SCHEMA,
                    "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "scientific_verdict_emitted": False,
                    "pilot_manifest_binding": _fake_binding("/tmp/manifest.json"),
                    "pilot_terminal_gate": terminal_gate,
                    "checkpoint": checkpoint_panel[key]["path"],
                    "checkpoint_binding": checkpoint_panel[key],
                    "checkpoint_panel_identity": {"arm": arm, "seed": seed},
                    "training_scene_separation": separation,
                    "model_label": key,
                    "model_identity": {"member": key},
                    "source_bindings": [],
                    "evaluation_contract": evaluator.evaluation_contract_v1(),
                    "latent_standardizer": {
                        "reference_arm": "masked_plain",
                        "reference_seed": evaluator.TRAINING_SEEDS[0],
                    },
                    "physical_outcome_equivalence": {
                        "basis": (
                            "equal_frozen_calibration_tolerance_aware_"
                            "physical_oracle_dense_rank"
                        ),
                        "model_dependent": False,
                        "latent_proximity_used": False,
                        "source": "bound_branch_oracle_dense_rank",
                    },
                    "direct_matched_branch_fidelity": direct,
                    "physical_arms": physical,
                    "paired_scene_cluster_comparisons": comparisons,
                    "preregistered_gates": gates,
                    "checkpoint_gate_status": status,
                }
            )
    return reports


def test_global_usefulness_requires_exact_12_member_primary_seed_panel(monkeypatch):
    monkeypatch.setattr(evaluator, "DEFAULT_RESAMPLES", 200)
    reports = _panel_member_reports()
    for report in reports:
        expected_alpha = evaluator.checkpoint_interval_alpha_v1(
            report["checkpoint_panel_identity"]["arm"]
        )
        assert report["paired_scene_cluster_comparisons"][
            "forecast_vs_current"
        ]["two_sided_alpha"] == expected_alpha
    aggregate = evaluator.aggregate_model_panel_v1(reports)
    assert aggregate["all_fixed_panel_members_reported"] is True
    assert aggregate["global_verdict"] == (
        "USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_DEVELOPMENT_ONLY"
    )
    assert all(
        all(rows.values())
        for rows in aggregate["primary_plain_seed_gate_passes"].values()
    )
    assert aggregate["delta_mechanism_controls"][
        "planning_usefulness_claim_eligible"
    ] is False
    assert aggregate["mechanism_adjudication"][
        "delta_observational_scale_route"
    ] == "STOP_DELTA_OBSERVATIONAL_SCALING_PROXY_NOT_MEANINGFUL"
    with pytest.raises(evaluator.BoundedBranchEvaluationError, match="exactly 12"):
        evaluator.aggregate_model_panel_v1(reports[:-1])
    fabricated = copy.deepcopy(reports)
    fabricated[0]["checkpoint_gate_status"] = (
        "CHECKPOINT_MEASUREMENT_FAILS_PREREGISTERED_GATES"
    )
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError, match="not derived"
    ):
        evaluator.aggregate_model_panel_v1(fabricated)
    mismatched_standardizer = copy.deepcopy(reports)
    mismatched_standardizer[0]["latent_standardizer"]["reference_seed"] = -1
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError,
        match="one frozen experiment identity",
    ):
        evaluator.aggregate_model_panel_v1(mismatched_standardizer)
    delta_only = evaluator.aggregate_model_panel_v1(
        _panel_member_reports(plain_pass=False)
    )
    assert delta_only["global_verdict"] == (
        "USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_NOT_ESTABLISHED"
    )
    assert delta_only["delta_mechanism_controls"][
        "planning_usefulness_claim_eligible"
    ] is False
    fabricated = copy.deepcopy(reports)
    fabricated[0]["preregistered_gates"] = {}
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError, match="not derived"
    ):
        evaluator.aggregate_model_panel_v1(fabricated)

    control_partition_drift = copy.deepcopy(reports)
    shuffled = control_partition_drift[0]["direct_matched_branch_fidelity"][
        "shuffled"
    ]
    shuffled["group_results"][0]["physical_oracle_dense_ranks"] = list(
        reversed(range(9))
    )
    shuffled["separable_group_results"][0][
        "physical_oracle_dense_ranks"
    ] = list(reversed(range(9)))
    for row in (
        shuffled["group_results"][0],
        shuffled["separable_group_results"][0],
    ):
        for action_id, signature in enumerate(
            row["joint_contrast_signatures_by_action"]
        ):
            signature["physical_outcome_dense_rank"] = 8 - action_id
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError,
        match="separability changed across controls",
    ):
        evaluator.aggregate_model_panel_v1(control_partition_drift)

    model_partition_drift = copy.deepcopy(reports)
    for direct_report in model_partition_drift[0][
        "direct_matched_branch_fidelity"
    ].values():
        direct_report["group_results"][0][
            "physical_oracle_dense_ranks"
        ] = list(reversed(range(9)))
        direct_report["separable_group_results"][0][
            "physical_oracle_dense_ranks"
        ] = list(reversed(range(9)))
        for row in (
            direct_report["group_results"][0],
            direct_report["separable_group_results"][0],
        ):
            for action_id, signature in enumerate(
                row["joint_contrast_signatures_by_action"]
            ):
                signature["physical_outcome_dense_rank"] = 8 - action_id
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError,
        match="physical separability changed across model-panel members",
    ):
        evaluator.aggregate_model_panel_v1(model_partition_drift)


def test_generation_evaluation_panel_lineage_rejects_panel_swap():
    separation = {
        "progression_analysis_binding": _fake_binding("/tmp/analysis.json"),
        "training_result_binding": _fake_binding("/tmp/result.json"),
        "checkpoint_panel_bindings": {"member": _fake_binding("/tmp/member.pt")},
        "progression_proxy_routing": {
            "decision": "DELTA_PROXY_NOT_MEANINGFUL"
        },
        "predecessor_place_manifest_binding": _fake_binding(
            "/tmp/place.json"
        ),
    }
    evaluator._require_model_panel_lineage_match_v1(  # noqa: SLF001
        copy.deepcopy(separation), separation
    )
    swapped = copy.deepcopy(separation)
    swapped["checkpoint_panel_bindings"]["member"] = _fake_binding(
        "/tmp/other.pt"
    )
    with pytest.raises(
        evaluator.BoundedBranchEvaluationError, match="lineage differ"
    ):
        evaluator._require_model_panel_lineage_match_v1(  # noqa: SLF001
            swapped, separation
        )


def test_terminal_reservation_content_is_reconstructed_not_just_rehashed():
    attempt = {
        "id": "attempt_v1",
        "root": "/tmp/attempt_v1",
        "maximum_attempts": 1,
        "must_be_absent": True,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    plan_binding = _fake_binding("/tmp/plan.json")
    authority_binding = _fake_binding("/tmp/authority.json")
    nonce = "b" * 64
    reservation = {
        "schema": "lewm_go2_world_model_counterfactual_attempt_reservation_v1",
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": attempt,
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "supervisor_nonce": nonce,
        "supervisor_pid": 101,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    evaluator._validate_reservation_document_v1(  # noqa: SLF001
        reservation,
        attempt=attempt,
        plan_binding=plan_binding,
        authority_binding=authority_binding,
        supervisor_nonce=nonce,
        supervisor_pid=101,
    )
    for field, bad in (
        ("retry_authorized", True),
        ("supervisor_nonce", "c" * 64),
        ("supervisor_pid", 102),
        ("status", "PASS"),
    ):
        fabricated = copy.deepcopy(reservation)
        fabricated[field] = bad
        with pytest.raises(
            evaluator.BoundedBranchEvaluationError, match="ownership/content"
        ):
            evaluator._validate_reservation_document_v1(  # noqa: SLF001
                fabricated,
                attempt=attempt,
                plan_binding=plan_binding,
                authority_binding=authority_binding,
                supervisor_nonce=nonce,
                supervisor_pid=101,
            )


def test_terminal_checker_counts_and_roles_are_recomputed_from_exact_plan():
    counts = {
        "scenes": 32,
        "states": 256,
        "roles": {"eval": 128, "train": 128},
        "actions": 9,
        "candidate_branches": 2304,
        "sentinel_branches": 0,
        "total_branches": 2304,
        "context_frames": 768,
        "target_frames": 2304,
    }
    manifest_binding = _fake_binding("/tmp/manifest.json")
    report = {
        "schema": "checker-v1",
        "status": "PASS",
        "phase": "joined_pilot",
        "authority_granted": False,
        "scientific_claim_granted": False,
        "runtime_payloads_opened": False,
        "rgb_bytes_opened": False,
        "checkpoints_opened": False,
        "manifest_binding": manifest_binding,
        "attempt_id": "attempt_v1",
        "purpose": "bounded_wm_a_pilot",
        "counts": counts,
        "roles": {"eval": 128, "train": 128},
        "can_freeze_pilot_contract": True,
        "rgb_artifacts": 3072,
    }
    evaluator._validate_checker_report_values_v1(  # noqa: SLF001
        report,
        report_schema="checker-v1",
        phase="joined_pilot",
        manifest_binding=manifest_binding,
        attempt_id="attempt_v1",
        expected_counts=counts,
    )
    for field, bad in (
        ("rgb_artifacts", 3071),
        ("roles", {"eval": 127, "train": 129}),
        ("counts", {**counts, "candidate_branches": 2303}),
    ):
        fabricated = copy.deepcopy(report)
        fabricated[field] = bad
        with pytest.raises(
            evaluator.BoundedBranchEvaluationError, match="measurements changed"
        ):
            evaluator._validate_checker_report_values_v1(  # noqa: SLF001
                fabricated,
                report_schema="checker-v1",
                phase="joined_pilot",
                manifest_binding=manifest_binding,
                attempt_id="attempt_v1",
                expected_counts=counts,
            )
def test_pilot_uses_separate_authority_without_weakening_calibration_collector():
    assert "bounded_wm_a_pilot" not in calibration_collector.NON_SMOKE_AUTHORITY_CONTRACTS
    paths = authority.canonical_source_paths_v1()
    assert paths["external_supervisor"].endswith(
        "run_go2_world_model_bounded_branch_experiment_authorized_v1.py"
    )
    assert paths["collector"] == "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"


def test_fresh_bounded_runtime_imports_are_inside_canonical_source_closure():
    probe = r'''
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(repo_root))
import scripts.run_go2_world_model_bounded_branch_experiment_authorized_v1  # noqa: F401,E402
import scripts.run_go2_world_model_bounded_branch_evaluation_panel_v1  # noqa: F401,E402

loaded = set()
for module in tuple(sys.modules.values()):
    module_file = getattr(module, "__file__", None)
    if not module_file:
        continue
    candidate = Path(module_file)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        candidate = candidate.resolve(strict=True)
        relative = candidate.relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        continue
    parts = tuple(part.lower() for part in Path(relative).parts)
    if relative.startswith(".generated/") or any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out", "protected"}
        or part.startswith(("heldout_", "held_out_", "held-out-", "protected_"))
        for part in parts
    ):
        continue
    if relative.endswith(".py"):
        loaded.add(relative)
print(json.dumps(sorted(loaded)))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    closure = set(authority.canonical_source_paths_v1().values())
    assert loaded <= closure, sorted(loaded - closure)
    assert set(authority.BOUNDED_DYNAMIC_IMPORT_SOURCE_PATHS.values()) <= loaded

    # These imports occur inside evaluation functions rather than only at
    # module import time, so keep their repository paths explicit as well.
    function_local_evaluation_imports = {
        "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
        "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py",
        "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
        "scripts/build_go2_world_model_bounded_branch_experiment_authority_v1.py",
        "scripts/check_go2_world_model_counterfactual_pilot_v1.py",
        "scripts/dev_probe_counterfactual_action_fidelity.py",
        "scripts/dev_train_temporal_jepa_scaled.py",
        "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
        "scripts/run_go2_world_model_bounded_branch_experiment_authorized_v1.py",
        "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py",
    }
    assert function_local_evaluation_imports <= closure


def test_calibrated_resource_projection_is_bounded():
    projected = authority.projected_caps_v1({
        "calibration_wall_seconds": 80.0,
        "calibration_stored_rgb_bytes": 1_000_000,
        "calibration_gpu_baseline_used_bytes": 100,
        "calibration_gpu_peak_used_bytes": 200,
        "calibration_gpu_peak_delta_bytes": 100,
        "selected_device_total_vram_bytes": 10_000,
    })
    assert projected["minimum_wall_seconds"] == 3600.0
    assert projected["stored_rgb_byte_ceiling"] == 512 * 1024**2
    assert projected["selected_device_vram_byte_ceiling"] == 550
    with pytest.raises(authority.BoundedBranchAuthorityError, match="wall hard cap"):
        authority.projected_caps_v1({
            "calibration_wall_seconds": 2000.0,
            "calibration_stored_rgb_bytes": 1,
            "calibration_gpu_baseline_used_bytes": 100,
            "calibration_gpu_peak_used_bytes": 200,
            "calibration_gpu_peak_delta_bytes": 100,
            "selected_device_total_vram_bytes": 10_000,
        })
    with pytest.raises(authority.BoundedBranchAuthorityError, match="95 percent"):
        authority.projected_caps_v1({
            "calibration_wall_seconds": 80.0,
            "calibration_stored_rgb_bytes": 1,
            "calibration_gpu_baseline_used_bytes": 100,
            "calibration_gpu_peak_used_bytes": 9_000,
            "calibration_gpu_peak_delta_bytes": 8_900,
            "selected_device_total_vram_bytes": 10_000,
        })


def test_reservation_is_one_shot_and_tampering_fails_ownership(tmp_path):
    authority_document = {
        "attempt": {
            "id": "attempt",
            "root": str(tmp_path),
            "maximum_attempts": 1,
            "must_be_absent": True,
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        }
    }
    authority_binding = _fake_binding("/tmp/authority.json")
    plan_binding = _fake_binding("/tmp/plan.json")
    nonce = "4" * 64
    reservation = {
        "schema": "lewm_go2_world_model_counterfactual_attempt_reservation_v1",
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": authority_document["attempt"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "supervisor_nonce": nonce,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    (tmp_path / "reservation.json").write_text(__import__("json").dumps(reservation))
    assert supervisor._owned_reservation(
        tmp_path,
        nonce=nonce,
        authority=authority_document,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
    reservation["retry_authorized"] = True
    (tmp_path / "reservation.json").write_text(__import__("json").dumps(reservation))
    assert not supervisor._owned_reservation(
        tmp_path,
        nonce=nonce,
        authority=authority_document,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
