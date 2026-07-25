from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_jepa_encoder_pretraining_v1_contract",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_MODULES_IMPORTED_BY_CONTRACT = set(sys.modules) - _MODULES_BEFORE


def _latent_flow(*, update_zero: bool = False) -> dict[str, Any]:
    per_action = {
        action: (False if update_zero else action != "hold")
        for action in contract.ACTION_VOCABULARY
    }
    return {
        "all_values_finite": True,
        "all_components_within_closed_one_patch_bound": True,
        "hold_flow_exactly_zero": True,
        "maximum_absolute_flow_cell": 0.0 if update_zero else 0.5,
        "non_hold_action_nonzero_count": 0 if update_zero else 8,
        "per_action_any_nonzero": per_action,
    }


def _dense_pairwise_inverse(
    *,
    nll: float,
    correct_per_action: int = 15,
) -> dict[str, Any]:
    row_counts = {
        action: (
            60
            if action == "hold"
            else (55 if index < 3 else 54)
        )
        for index, action in enumerate(contract.ACTION_VOCABULARY)
    }
    recalls = {
        action: float(correct_per_action) / float(row_count)
        for action, row_count in row_counts.items()
    }
    ratio = 0.95
    family_margins = {
        family: 0.01 if index < 6 else 0.0
        for index, family in enumerate(contract.SCENE_FAMILIES)
    }
    return {
        "all_values_finite": True,
        "probabilities_all_values_finite": True,
        "probability_rows_normalized": True,
        "volume_all_values_finite": True,
        "volume_values_within_closed_unit_interval": True,
        "volume_channel_conservation": True,
        "displacement_all_values_finite": True,
        "displacement_values_within_closed_two_bound": True,
        "maximum_absolute_displacement_component": 0.5,
        "cross_pair_displacement_rms": 0.1,
        "cross_pair_displacement_value_count": 495 * 2 * 16 * 16,
        "same_tensor_diff_exact_zero": True,
        "same_tensor_volume_exact_zero": True,
        "same_tensor_displacement_exact_zero": True,
        "head_parameters_all_values_finite": True,
        "head_parameter_count": 8_713,
        "head_weight_tensors_all_nonzero": True,
        "zero_logit_reference_nll": math.log(9.0),
        "unscaled_dense_inverse_nll": nll,
        "dense_inverse_top1_accuracy":
            (correct_per_action * len(contract.ACTION_VOCABULARY)) / 495.0,
        "per_executed_action_dense_inverse": {
            action: {
                "row_count": row_counts[action],
                "mean_nll": nll,
                "recall": recalls[action],
            }
            for action in contract.ACTION_VOCABULARY
        },
        "dense_inverse_macro_balanced_accuracy":
            sum(recalls.values()) / float(len(recalls)),
        "correct_pair_nll": nll,
        "correct_pair_count": 495,
        "deranged_next_nll": nll / ratio,
        "deranged_next_pair_count": 495,
        "correct_to_deranged_nll_ratio": ratio,
        "non_hold_correct_pair_nll": nll,
        "non_hold_correct_pair_count": 435,
        "non_hold_current_current_nll": nll / ratio,
        "non_hold_current_current_pair_count": 435,
        "non_hold_correct_to_current_current_nll_ratio": ratio,
        "deranged_positive_family_margin_count": 6,
        "per_family_deranged_minus_correct_nll": family_margins,
    }


def _metrics(
    *,
    nll: float,
    raw_rank: float,
    projected_rank: float,
    correct_per_action: int = 15,
) -> dict[str, Any]:
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "centered_raw_patch_effective_rank": raw_rank,
        "centered_projected_target_effective_rank": projected_rank,
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 2.0,
        "true_pair_mse": 0.85,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "cyclic_wrong_action_mse": 0.90,
        "cyclic_wrong_action_pair_count": 495,
        "all_wrong_action_candidate_count": 3_960,
        "hardest_wrong_action_mse": 0.90,
        "non_hold_pair_count": 435,
        "non_hold_true_pair_mse": 0.85,
        "hold_action_mse": 0.90,
        "hold_action_pair_count": 435,
        "hold_action_rows_match_non_hold_rows": True,
        "shuffled_current_mse": 1.0,
        "latent_flow": _latent_flow(),
        "dense_pairwise_inverse": _dense_pairwise_inverse(
            nll=nll,
            correct_per_action=correct_per_action,
        ),
        "per_family": {
            family: {
                "cyclic_wrong_action_minus_true_mse":
                    0.01 if index < 6 else 0.0,
                "hardest_wrong_action_minus_true_mse":
                    0.01 if index < 6 else 0.0,
                "hold_action_minus_non_hold_true_mse":
                    0.01 if index < 6 else 0.0,
                "hold_action_rows_match_non_hold_rows": True,
            }
            for index, family in enumerate(contract.SCENE_FAMILIES)
        },
    }


def _update0_metrics() -> dict[str, Any]:
    return {
        "raw_cross_sample_variance": 4.0,
        "content_residual_spatial_diversity": 8.0,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
        "latent_flow": _latent_flow(update_zero=True),
        "dense_pairwise_inverse": _dense_pairwise_inverse(
            nll=math.log(9.0),
            correct_per_action=10,
        ),
    }


def _integrity() -> dict[str, Any]:
    return {"rng_state_preserved": True, "state_mutation_count": 0}


def _set_dense_nll(metrics: dict[str, Any], nll: float) -> None:
    dense = metrics["dense_pairwise_inverse"]
    ratio = float(dense["correct_to_deranged_nll_ratio"])
    dense["unscaled_dense_inverse_nll"] = nll
    dense["correct_pair_nll"] = nll
    dense["deranged_next_nll"] = nll / ratio
    dense["non_hold_correct_pair_nll"] = nll
    dense["non_hold_current_current_nll"] = nll / ratio
    for row in dense["per_executed_action_dense_inverse"].values():
        row["mean_nll"] = nll


def _set_action_correct_counts(
    metrics: dict[str, Any],
    correct_counts: list[int],
) -> None:
    assert len(correct_counts) == len(contract.ACTION_VOCABULARY)
    dense = metrics["dense_pairwise_inverse"]
    total_correct = 0
    recalls: list[float] = []
    for action, correct in zip(contract.ACTION_VOCABULARY, correct_counts):
        row = dense["per_executed_action_dense_inverse"][action]
        recall = float(correct) / float(row["row_count"])
        row["recall"] = recall
        recalls.append(recall)
        total_correct += correct
    dense["dense_inverse_top1_accuracy"] = total_correct / 495.0
    dense["dense_inverse_macro_balanced_accuracy"] = sum(recalls) / 9.0


def _source_manifest_core() -> dict[str, Any]:
    bindings = [
        {
            "path": path,
            "file_sha256": hashlib.sha256(path.encode("ascii")).hexdigest(),
            "byte_count": len(path),
        }
        for path in contract.SOURCE_PATHS
    ]
    return {
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources":
            list(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES),
        "excluded_runtime_categories":
            list(contract.PROHIBITED_RUNTIME_CATEGORIES),
        "source_paths": list(contract.SOURCE_PATHS),
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": len(bindings),
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": dict(contract.SOURCE_ONLY_AUTHORITY),
    }


class JepaEncoderPretrainingV9ContractTests(unittest.TestCase):
    def test_import_and_frozen_evidence_are_exact(self) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }))
        self.assertEqual(
            contract.SCHEMA_PREFIX,
            "lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9",
        )
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "b775093897669c91d8c1b9e7d148e257881bcedf",
        )
        raw = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
        self.assertEqual(len(raw), 22_115)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "bfb0f1c2bd77ee78f6d4bf34cff8ec8d3d3c4bced8fb7b4269fa0a3f0bb30f2b",
        )
        self.assertEqual(contract.prior_terminal_audit_binding(), {
            "path":
                "docs/lewm_go2_rgb_action_conditioned_local_correspondence_"
                "all_candidate_identification_jepa_v8_terminal_audit_"
                "2026-07-25.json",
            "commit": "9f3e2bc96a6e4ea419574f109c890299d0608659",
            "file_sha256":
                "3ea4a8cc4405b0880d2e05217e4b4acefc5b9df5fad9bcdd9a682db42e273173",
            "content_sha256":
                "ff8339aa6109933e85d60ad118dc912fd091dddf7dfd80b18d00453ce7c01367",
            "byte_count": 20_028,
        })
        self.assertEqual(
            contract.OUTPUT_ROOT_RELATIVE_PATH,
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_probe_v9",
        )

    def test_exact_v5_base_and_v9_head_contract(self) -> None:
        self.assertEqual(contract.FLOW_PROJECTION_SHAPE, (2, 192))
        self.assertEqual(contract.FLOW_PROJECTION_PARAMETER_COUNT, 384)
        self.assertEqual(contract.DENSE_PAIRWISE_HEAD_PARAMETER_COUNT, 8_713)
        self.assertEqual(contract.DENSE_PAIRWISE_HEAD_INITIALIZATION_SEED, 20260725)
        self.assertEqual(contract.DENSE_PAIRWISE_VOLUME_INPUT_CHANNELS, 256)
        self.assertEqual(contract.DENSE_PAIRWISE_HEAD_HIDDEN_CHANNELS, 16)
        science = contract.science_contract()
        self.assertEqual(
            science["phase_a"]["reviewed_forward_base_commit"],
            "c93124b15387acf1fd440d281e9c4503a9e8355a",
        )
        objective = science["phase_a"]["objective"]
        self.assertTrue(
            objective["preserved_v5_forward"]
            ["current_plus_action_state_dependent_latent_flow"]
        )
        volume = objective["dense_pairwise_spatial_cost_volume"]
        self.assertEqual(
            volume["volume"],
            "diff.transpose(1,2).reshape(B,256,16,16).contiguous()",
        )
        self.assertEqual(
            volume["volume_axes"],
            ["batch", "target_token_channel", "source_y", "source_x"],
        )
        self.assertEqual(volume["head_input"], "V_only")
        displacement = objective["displacement_observation_only"]
        self.assertEqual(displacement["coordinate_columns"], ["dy", "dx"])
        self.assertEqual(displacement["head_or_loss_input_count"], 0)
        head = science["initialization"]["dense_pairwise_inverse_head"]
        self.assertEqual(head["parameter_count"], 8_713)
        self.assertEqual(head["linear_bias"], 0.0)
        self.assertTrue(
            head["all_three_weight_tensors_every_scalar_nonzero"]
        )
        serialized = json.dumps(science, sort_keys=True)
        self.assertNotIn("centered_log_soft_cross_entropy", serialized)
        self.assertNotIn("correspondence_action_identification", serialized)

    def test_optimizer_schedule_caps_roles_and_denials_are_exact(self) -> None:
        self.assertEqual(contract.PHASE_A_ENCODER_PARAMETER_PREFIXES, ("encoder.",))
        self.assertEqual(contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES, (
            "online_target_projector.",
            "prediction_projector.",
            "predictor.",
            "dense_pairwise_inverse_head.",
        ))
        optimizer = contract.science_contract()["phase_a"]["optimizer"]
        self.assertEqual(optimizer["encoder_learning_rate"], 1e-4)
        self.assertEqual(optimizer["other_learning_rate"], 3e-4)
        self.assertEqual(optimizer["global_clip_norm"], 1.0)
        self.assertEqual(contract.CHECKPOINT_UPDATES, (100, 400, 1_000))
        self.assertEqual(contract.PHASE_A_MAXIMUM_PRESENTATIONS, 16_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_PRESENTATIONS, 32_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_UPDATE, 2_000)
        self.assertEqual(contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        self.assertEqual(contract.PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        for phase in ("phase_a", "phase_b"):
            identity = contract.build_schedule_identity(phase)
            self.assertEqual(identity["updates"], 1_000)
            self.assertEqual(identity["presentations"], 16_000)
            self.assertEqual(
                contract.validate_schedule_identity(identity, phase=phase),
                identity,
            )
        runtime = contract.runtime_authorization_template()
        self.assertEqual(contract.validate_runtime_inputs(runtime), runtime)
        self.assertEqual(
            runtime["raw"]["role_policy"]["metadata_only_roles"],
            ["authority", "index"],
        )
        self.assertEqual(
            runtime["raw"]["role_policy"]["model_facing_roles"],
            ["train", "checkpoint_selection"],
        )
        self.assertTrue(all(value is False
                            for value in contract.DOWNSTREAM_DENIALS.values()))

    def test_receipt_status_and_forbidden_access_contracts(self) -> None:
        self.assertEqual(contract.ATTEMPT_INDEX, 1)
        self.assertEqual(contract.MAXIMUM_ATTEMPTS, 1)
        self.assertEqual(contract.NORMAL_PHASE_A_RECEIPT_PATHS, (
            "reservation.json",
            "phase_a/metrics.json",
            "phase_a/artifact.json",
            "access.json",
            "result.json",
            "completed.json",
        ))
        lifecycle = contract.science_contract()["lifecycle"]
        self.assertTrue(
            lifecycle["operational_failure"]
            ["missing_receipts_never_synthesized_or_fabricated"]
        )
        self.assertEqual(
            lifecycle["operational_failure"]
            ["reservation_publication_failure_status"],
            "TERMINAL_RESERVATION_PUBLICATION_FAILURE",
        )
        counters = {field: 0 for field in contract.ACCESS_ZERO_COUNTER_FIELDS}
        self.assertEqual(contract.validate_access_zero_counters(counters), counters)
        changed = dict(counters)
        changed["production_input_open_count"] = 1
        with self.assertRaises(PermissionError):
            contract.validate_access_zero_counters(changed)
        for control in contract.PHASE_A_FAILURE_CONTROLS:
            chain = {
                "metrics": control,
                "artifact": control,
                "result": control,
                "completion": control,
            }
            self.assertEqual(
                contract.validate_phase_a_failure_status_chain(chain),
                chain,
            )
            broken = dict(chain)
            broken["completion"] = "TERMINAL_FAIL"
            with self.assertRaises(ValueError):
                contract.validate_phase_a_failure_status_chain(broken)

    def test_staged_update100_update400_and_terminal_pass(self) -> None:
        update0 = _update0_metrics()
        update100 = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        first = contract.evaluate_phase_a_continuation(
            100, update100, update0, _integrity()
        )
        self.assertTrue(first["passed"])
        self.assertEqual(first["control"], contract.CONTROL_CONTINUE)
        update400 = _metrics(nll=1.9, raw_rank=38.0, projected_rank=33.0)
        second = contract.evaluate_phase_a_continuation(
            400,
            update400,
            update0,
            _integrity(),
            previous_metrics=update100,
        )
        self.assertTrue(second["passed"])
        self.assertEqual(second["control"], contract.CONTROL_CONTINUE)
        update1000 = _metrics(nll=1.8, raw_rank=48.0, projected_rank=48.0)
        terminal = contract.evaluate_phase_a(
            update1000,
            update0,
            _integrity(),
            update400,
        )
        self.assertTrue(terminal["passed"])
        self.assertEqual(terminal["control"], contract.CONTROL_PHASE_A_PASS)
        self.assertEqual(terminal["latent_flow"], _latent_flow())
        self.assertEqual(
            terminal["dense_pairwise_inverse"],
            _dense_pairwise_inverse(nll=1.8),
        )

    def test_update100_is_staged_and_does_not_require_forward_ordering(self) -> None:
        metrics = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        metrics["mean_target_mse"] = 0.80
        metrics["cyclic_wrong_action_mse"] = 0.80
        metrics["hardest_wrong_action_mse"] = 0.80
        metrics["hold_action_mse"] = 0.80
        for row in metrics["per_family"].values():
            row["cyclic_wrong_action_minus_true_mse"] = -0.1
            row["hold_action_minus_non_hold_true_mse"] = -0.1
        result = contract.evaluate_phase_a_continuation(
            100, metrics, _update0_metrics(), _integrity()
        )
        self.assertTrue(result["passed"])
        self.assertNotIn(
            "true_strictly_below_point99_cyclic_wrong_action",
            result["conjuncts"],
        )

    def test_every_update100_dense_gate_is_strict(self) -> None:
        update0 = _update0_metrics()
        cases: list[tuple[str, Any]] = []
        nll_boundary = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        _set_dense_nll(
            nll_boundary,
            0.98
            * nll_boundary["dense_pairwise_inverse"]
            ["zero_logit_reference_nll"],
        )
        cases.append(("dense_inverse_nll_strictly_below_point98_log9", nll_boundary))

        macro_boundary = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        _set_action_correct_counts(macro_boundary, [55, 55, 0, 0, 0, 0, 0, 0, 0])
        cases.append((
            "dense_inverse_macro_balanced_accuracy_strictly_above_two_ninths",
            macro_boundary,
        ))
        deranged_ratio = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        dense = deranged_ratio["dense_pairwise_inverse"]
        dense["correct_to_deranged_nll_ratio"] = 0.99
        dense["deranged_next_nll"] = dense["correct_pair_nll"] / 0.99
        cases.append(("correct_to_deranged_nll_ratio_strictly_below_point99", deranged_ratio))
        current_ratio = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        dense = current_ratio["dense_pairwise_inverse"]
        dense["non_hold_correct_to_current_current_nll_ratio"] = 0.99
        dense["non_hold_current_current_nll"] = dense["non_hold_correct_pair_nll"] / 0.99
        cases.append((
            "non_hold_correct_to_current_current_nll_ratio_strictly_below_point99",
            current_ratio,
        ))
        family_count = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        dense = family_count["dense_pairwise_inverse"]
        dense["per_family_deranged_minus_correct_nll"] = {
            family: 0.01 if index < 5 else 0.0
            for index, family in enumerate(contract.SCENE_FAMILIES)
        }
        dense["deranged_positive_family_margin_count"] = 5
        cases.append(("deranged_nll_margin_positive_in_at_least_six_families", family_count))
        raw_rank = _metrics(
            nll=2.0,
            raw_rank=contract.PHASE_A_UPDATE_100_THRESHOLDS[
                "centered_raw_patch_effective_rank_strictly_greater_than"
            ],
            projected_rank=20.0,
        )
        cases.append(("centered_raw_rank_above_v3_update_zero", raw_rank))
        projected_rank = _metrics(
            nll=2.0,
            raw_rank=30.0,
            projected_rank=contract.PHASE_A_UPDATE_100_THRESHOLDS[
                "centered_projected_target_effective_rank_strictly_greater_than"
            ],
        )
        cases.append(("centered_projected_rank_above_v3_update_zero", projected_rank))
        health = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        health["raw_cross_sample_variance"] = 0.999
        cases.append(("raw_cross_sample_variance_at_least_quarter_update0", health))
        flow = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        flow["latent_flow"]["per_action_any_nonzero"]["arc_left"] = False
        flow["latent_flow"]["non_hold_action_nonzero_count"] = 7
        cases.append(("all_eight_non_hold_actions_have_nonzero_flow", flow))

        for conjunct, metrics in cases:
            with self.subTest(conjunct=conjunct):
                result = contract.evaluate_phase_a_continuation(
                    100, metrics, update0, _integrity()
                )
                self.assertFalse(result["passed"])
                self.assertFalse(result["conjuncts"][conjunct])
                self.assertEqual(
                    result["control"],
                    contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
                )

    def test_update400_requires_progress_and_new_forward_gates(self) -> None:
        update0 = _update0_metrics()
        update100 = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        equal_nll = _metrics(nll=2.0, raw_rank=38.0, projected_rank=33.0)
        result = contract.evaluate_phase_a_continuation(
            400, equal_nll, update0, _integrity(), update100
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]["dense_inverse_nll_strictly_lower_than_update100"]
        )
        no_macro_regression = _metrics(
            nll=1.9,
            raw_rank=38.0,
            projected_rank=33.0,
            correct_per_action=15,
        )
        prior_better_macro = deepcopy(update100)
        _set_action_correct_counts(prior_better_macro, [16] * 9)
        result = contract.evaluate_phase_a_continuation(
            400,
            no_macro_regression,
            update0,
            _integrity(),
            prior_better_macro,
        )
        self.assertFalse(
            result["conjuncts"]
            ["dense_inverse_macro_balanced_accuracy_not_below_update100"]
        )
        cyclic_boundary = _metrics(nll=1.9, raw_rank=38.0, projected_rank=33.0)
        cyclic_boundary["true_pair_mse"] = 0.99
        cyclic_boundary["cyclic_wrong_action_mse"] = 1.0
        result = contract.evaluate_phase_a_continuation(
            400,
            cyclic_boundary,
            update0,
            _integrity(),
            update100,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]
            ["true_strictly_below_point99_cyclic_wrong_action"]
        )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                400,
                _metrics(nll=1.9, raw_rank=38.0, projected_rank=33.0),
                update0,
                _integrity(),
            )

    def test_terminal_requires_update400_progress_and_complete_v5_gate(self) -> None:
        update0 = _update0_metrics()
        update400 = _metrics(nll=1.9, raw_rank=38.0, projected_rank=33.0)
        equal_nll = _metrics(nll=1.9, raw_rank=48.0, projected_rank=48.0)
        result = contract.evaluate_phase_a(
            equal_nll, update0, _integrity(), update400
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]["dense_inverse_nll_strictly_lower_than_update400"]
        )
        boundary = _metrics(nll=1.8, raw_rank=48.0, projected_rank=48.0)
        boundary["true_pair_mse"] = 0.90
        boundary["mean_target_mse"] = 1.0
        boundary["cyclic_wrong_action_mse"] = 0.90 / 0.95
        boundary["hardest_wrong_action_mse"] = 0.90 / 0.95
        boundary["non_hold_true_pair_mse"] = 0.90
        boundary["hold_action_mse"] = 0.90 / 0.95
        result = contract.evaluate_phase_a(
            boundary, update0, _integrity(), update400
        )
        self.assertTrue(result["passed"])

    def test_dense_receipt_populations_ratios_and_health_fail_closed(self) -> None:
        base = _metrics(nll=2.0, raw_rank=30.0, projected_rank=20.0)
        malformed = deepcopy(base)
        malformed["dense_pairwise_inverse"]["correct_pair_count"] = 494
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                100, malformed, _update0_metrics(), _integrity()
            )
        impossible_action_population = deepcopy(base)
        action_rows = impossible_action_population["dense_pairwise_inverse"][
            "per_executed_action_dense_inverse"
        ]
        action_rows["arc_left"]["row_count"] += 1
        action_rows["hold"]["row_count"] -= 1
        _set_action_correct_counts(
            impossible_action_population,
            [15] * len(contract.ACTION_VOCABULARY),
        )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                100,
                impossible_action_population,
                _update0_metrics(),
                _integrity(),
            )
        malformed = deepcopy(base)
        malformed["dense_pairwise_inverse"][
            "cross_pair_displacement_value_count"
        ] -= 1
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                100, malformed, _update0_metrics(), _integrity()
            )
        malformed = deepcopy(base)
        malformed["dense_pairwise_inverse"][
            "per_executed_action_dense_inverse"
        ]["arc_left"]["recall"] = 0.1
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                100, malformed, _update0_metrics(), _integrity()
            )
        unhealthy = deepcopy(base)
        unhealthy["dense_pairwise_inverse"]["volume_channel_conservation"] = False
        result = contract.evaluate_phase_a_continuation(
            100, unhealthy, _update0_metrics(), _integrity()
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]
            ["dense_pairwise_values_probabilities_volume_and_displacement_healthy"]
        )

    def test_phase_b_thresholds_are_unchanged(self) -> None:
        passing = {
            "complete_physical_scope_count": 1,
            "margin_count": 189,
            "passed_margin_count": 98,
            "total_shortfall": 41.0,
            "rough_motion": {
                "pixel_balanced_accuracy": 0.82,
                "ground_balanced_accuracy": 0.65,
                "depth_p95_m": 0.97,
            },
        }
        self.assertTrue(contract.evaluate_phase_b(passing)["passed"])
        boundary = deepcopy(passing)
        boundary["passed_margin_count"] = 97
        self.assertFalse(contract.evaluate_phase_b(boundary)["passed"])

    def test_canonical_json_manifest_review_and_authorization_validators(self) -> None:
        manifest_raw = contract.canonical_json_bytes(
            contract.with_content_sha256(_source_manifest_core())
        ) + b"\n"
        manifest = contract.validate_source_manifest(manifest_raw)
        expected_sources = {
            binding["path"]: binding["file_sha256"]
            for binding in manifest["source_bindings"]
        }
        expected_sources[contract.SOURCE_MANIFEST_RELATIVE_PATH] = "a" * 64
        expected_sources[contract.PREREGISTRATION_RELATIVE_PATH] = (
            contract.PREREGISTRATION_FILE_SHA256
        )
        expected_sources[contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] = (
            contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
        )
        manifest_binding = {
            "path": contract.SOURCE_MANIFEST_RELATIVE_PATH,
            "file_sha256": "a" * 64,
            "content_sha256": manifest["content_sha256"],
            "byte_count": len(manifest_raw),
        }
        review_core = {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS_SOURCE_AND_SCIENCE",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_v9_reviewer",
            "reviewed_sources": expected_sources,
            "source_manifest": manifest_binding,
            "preregistration": contract.preregistration_binding(),
            "prior_terminal_audit": contract.prior_terminal_audit_binding(),
            "science_contract": contract.science_contract(),
            "source_only_checks": {
                "stdlib_only_contract_import": True,
                "generated_inputs_opened": [],
                "checkpoints_or_tensors_opened": [],
                "sealed_or_heldout_opened": [],
            },
            "scientific_checks": dict(contract.SCIENTIFIC_REVIEW_CHECKS),
            "findings": [],
            "authority": dict(contract.REVIEW_AUTHORITY),
        }
        review = contract.with_content_sha256(review_core)
        self.assertEqual(
            contract.validate_review(review, expected_sources=expected_sources),
            review,
        )
        review_raw = contract.canonical_json_bytes(review) + b"\n"
        review_binding = contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=review["content_sha256"],
        )
        authorization_core = {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED_ONE_EXACT_TWO_PHASE_PROBE",
            "authorizer": "/root/independent_v9_authorizer",
            "independent_source_review": review_binding,
            "preregistration": contract.preregistration_binding(),
            "runtime_inputs": contract.runtime_authorization_template(),
            "experiment": contract.science_contract(),
            "authority": dict(contract.EXECUTION_AUTHORITY),
        }
        authorization = contract.with_content_sha256(authorization_core)
        self.assertEqual(
            contract.validate_authorization(
                authorization,
                review_binding=review_binding,
                reviewer=review["reviewer"],
            ),
            authorization,
        )


if __name__ == "__main__":
    unittest.main()
