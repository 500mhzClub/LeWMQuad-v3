from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
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


def _per_action(value: Any) -> dict[str, Any]:
    return {action: deepcopy(value) for action in contract.ACTION_VOCABULARY}


def _local_correspondence(*, update_zero: bool = False) -> dict[str, Any]:
    if update_zero:
        correct = math.log(9.0)
        deranged = correct
        hardest = correct
        deranged_margins = {
            family: 0.0 for family in contract.SCENE_FAMILIES
        }
        hardest_margins = dict(deranged_margins)
        different = _per_action(False)
        return {
            "all_values_finite": True,
            "target_all_values_finite": True,
            "target_all_strictly_positive": True,
            "target_rows_normalized": True,
            "student_all_strictly_positive": True,
            "student_rows_normalized": True,
            "transport_weight_all_values_finite": True,
            "transport_weight_any_nonzero": False,
            "maximum_absolute_student_logit": 0.0,
            "correct_centered_log_cross_entropy": correct,
            "deranged_centered_log_cross_entropy": deranged,
            "correct_to_deranged_cross_entropy_ratio": 1.0,
            "deranged_positive_family_margin_count": 0,
            "per_family_deranged_minus_correct_cross_entropy":
                deranged_margins,
            "per_action_correct_target_centered_log_cross_entropy":
                _per_action(correct),
            "hardest_wrong_centered_log_cross_entropy": hardest,
            "executed_to_hardest_wrong_cross_entropy_ratio": 1.0,
            "hardest_wrong_positive_family_margin_count": 0,
            "per_family_hardest_wrong_minus_executed_cross_entropy":
                hardest_margins,
            "mean_target_kl_to_uniform": 0.05,
            "per_action_probability_rows_positive_and_normalized":
                _per_action(True),
            "non_hold_action_distribution_different_from_hold_count": 0,
            "per_action_distribution_different_from_hold": different,
            "maximum_absolute_expected_offset_component": 0.0,
            "hold_probabilities_bitwise_uniform": True,
            "hold_expected_offset_exactly_zero": True,
            "hold_transport_identity_exact": True,
            "all_action_distributions_bitwise_equal_to_hold": True,
            "all_action_distributions_bitwise_equal_to_uniform": True,
            "correct_and_deranged_cross_entropy_bitwise_equal": True,
            "all_action_transports_identity_exact": True,
        }

    correct = 1.90
    deranged = 2.00
    hardest = 2.00
    deranged_margins = {
        family: 0.02 if index < 6 else 0.0
        for index, family in enumerate(contract.SCENE_FAMILIES)
    }
    hardest_margins = {
        family: 0.01 if index < 6 else 0.0
        for index, family in enumerate(contract.SCENE_FAMILIES)
    }
    different = {
        action: action != "hold" for action in contract.ACTION_VOCABULARY
    }
    return {
        "all_values_finite": True,
        "target_all_values_finite": True,
        "target_all_strictly_positive": True,
        "target_rows_normalized": True,
        "student_all_strictly_positive": True,
        "student_rows_normalized": True,
        "transport_weight_all_values_finite": True,
        "transport_weight_any_nonzero": True,
        "maximum_absolute_student_logit": 1.0,
        "correct_centered_log_cross_entropy": correct,
        "deranged_centered_log_cross_entropy": deranged,
        "correct_to_deranged_cross_entropy_ratio": correct / deranged,
        "deranged_positive_family_margin_count": 6,
        "per_family_deranged_minus_correct_cross_entropy":
            deranged_margins,
        "per_action_correct_target_centered_log_cross_entropy":
            _per_action(1.95),
        "hardest_wrong_centered_log_cross_entropy": hardest,
        "executed_to_hardest_wrong_cross_entropy_ratio":
            correct / hardest,
        "hardest_wrong_positive_family_margin_count": 6,
        "per_family_hardest_wrong_minus_executed_cross_entropy":
            hardest_margins,
        "mean_target_kl_to_uniform": 0.05,
        "per_action_probability_rows_positive_and_normalized":
            _per_action(True),
        "non_hold_action_distribution_different_from_hold_count": 8,
        "per_action_distribution_different_from_hold": different,
        "maximum_absolute_expected_offset_component": 0.5,
        "hold_probabilities_bitwise_uniform": True,
        "hold_expected_offset_exactly_zero": True,
        "hold_transport_identity_exact": True,
        "all_action_distributions_bitwise_equal_to_hold": False,
        "all_action_distributions_bitwise_equal_to_uniform": False,
        "correct_and_deranged_cross_entropy_bitwise_equal": False,
        "all_action_transports_identity_exact": False,
    }


def _passing_phase_a_metrics() -> dict[str, Any]:
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "centered_raw_patch_effective_rank": 48.0,
        "centered_projected_target_effective_rank": 48.0,
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 2.0,
        "true_pair_mse": 0.85,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "cyclic_wrong_action_mse": 0.90,
        "cyclic_wrong_action_pair_count": 495,
        "all_wrong_action_candidate_count": 3_960,
        "hardest_wrong_action_mse": 0.90,
        "non_hold_pair_count": contract.SELECTION_NON_HOLD_PAIR_COUNT,
        "non_hold_true_pair_mse": 0.85,
        "hold_action_mse": 0.90,
        "hold_action_pair_count": contract.SELECTION_NON_HOLD_PAIR_COUNT,
        "hold_action_rows_match_non_hold_rows": True,
        "shuffled_current_mse": 0.90,
        "local_correspondence": _local_correspondence(),
        "per_family": {
            family: {
                "cyclic_wrong_action_minus_true_mse":
                    0.01 if index < 6 else 0.0,
                "hardest_wrong_action_minus_true_mse":
                    0.005 if index < 6 else 0.0,
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
        "local_correspondence": _local_correspondence(update_zero=True),
    }


def _observation_integrity() -> dict[str, Any]:
    return {"rng_state_preserved": True, "state_mutation_count": 0}


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


class JepaEncoderPretrainingContractTests(unittest.TestCase):
    def test_import_and_frozen_v7_evidence_are_exact(self) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }))
        self.assertEqual(
            contract.SCHEMA_PREFIX,
            "lewm_go2_rgb_action_conditioned_local_correspondence_"
            "transport_jepa_v7",
        )
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "094a478d219488a953b008bd7487b0a4a729a5bb",
        )
        raw = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
        self.assertEqual(len(raw), 21_190)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "b17b543bc63b4201e8a049c90dfa43eaaf78b6035ccf4d94235c5bbc110b8aa0",
        )
        self.assertEqual(contract.prior_terminal_audit_binding(), {
            "path":
                "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_"
                "v6_existing_pair_inverse_dynamics_terminal_audit_"
                "2026-07-25.json",
            "commit": "c3259adc3b48a3f1d5784a1ada0eaac8b12f7855",
            "file_sha256":
                "9ab15ff2100e46fd341d4266c534c289bd453f74517743886ceccea165e15d01",
            "content_sha256":
                "aed1a84c890ae5a7b5ad068cffe7cc1b3260bc4b39919366fcb9bee888666c6d",
            "byte_count": 19_590,
        })
        self.assertEqual(
            contract.OUTPUT_ROOT_RELATIVE_PATH,
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_action_conditioned_local_correspondence_transport_jepa_"
            "probe_v7",
        )

    def test_exact_transport_architecture_and_determinism_contract(self) -> None:
        self.assertEqual(contract.TRANSPORT_PROJECTION_SHAPE, (8, 192))
        self.assertEqual(contract.TRANSPORT_PROJECTION_PARAMETER_COUNT, 1_536)
        self.assertEqual(contract.LOCAL_CORRESPONDENCE_CENTER_INDEX, 4)
        self.assertEqual(contract.LOCAL_CORRESPONDENCE_FULL_OFFSETS, (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ))
        self.assertEqual(
            contract.LOCAL_CORRESPONDENCE_NONCENTER_OFFSETS,
            contract.LOCAL_CORRESPONDENCE_FULL_OFFSETS[:4]
            + contract.LOCAL_CORRESPONDENCE_FULL_OFFSETS[5:],
        )
        science = contract.science_contract()
        projection = science["initialization"]["transport_projection"]
        self.assertEqual(
            projection["path"],
            "prediction_projector.transport_weight",
        )
        self.assertEqual(projection["shape"], [8, 192])
        self.assertFalse(projection["bias"])
        self.assertEqual(projection["parameter_count"], 1_536)
        self.assertEqual(projection["value"], 0.0)
        target = science["phase_a"]["objective"][
            "detached_correspondence_target"
        ]
        self.assertEqual(
            target["target_logit"],
            "dot(LN(zn_i),LN(zc_J(i,o)))/sqrt(192)",
        )
        transport = science["phase_a"]["objective"][
            "local_correspondence_transport"
        ]
        self.assertEqual(
            transport["center_logit"],
            "g_center=-sum_noncenter(g_noncenter)",
        )
        self.assertEqual(
            transport["centered_coefficients"],
            "C_i_a_o=P_i_a_o-U_i_a_o",
        )
        self.assertEqual(transport["forbidden_operations"], [
            "grid_sample",
            "unfold",
            "differentiable_padding",
            "materialize_B_by_9_by_256_by_9_by_192",
        ])
        centered = science["phase_a"]["objective"][
            "centered_log_soft_cross_entropy"
        ]
        self.assertEqual(
            centered["formula"],
            "Hc(Q,logP)=-logP_4-sum_o(Q_o*(logP_o-logP_4))",
        )
        self.assertEqual(centered["loss_weight"], 1.0)
        self.assertEqual(
            science["phase_a"]["optimizer"]["determinism"],
            {
                "strict_deterministic_algorithms": True,
                "warn_only": False,
                "expected_warning_count": 0,
                "permitted_warning_count": 0,
                "strict_state_restored": True,
            },
        )
        self.assertFalse(
            science["phase_b"]["transport_projection_optimizer_included"]
        )
        self.assertFalse(
            science["phase_b"]["transport_projection_copied_into_phase_b_model"]
        )

    def test_exact_budgets_schedule_and_runtime_authority(self) -> None:
        self.assertEqual(contract.CHECKPOINT_UPDATES, (100, 400, 1_000))
        self.assertEqual(contract.PHASE_A_MAXIMUM_PRESENTATIONS, 16_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_PRESENTATIONS, 32_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_UPDATE, 2_000)
        self.assertEqual(contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        self.assertEqual(contract.PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        self.assertEqual(contract.CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES, 120)
        self.assertEqual(
            contract.science_contract()["cumulative_caps"][
                "phase_b_gpu_active_minutes"
            ],
            60,
        )
        for phase in ("phase_a", "phase_b"):
            identity = contract.build_schedule_identity(phase)
            self.assertEqual(identity["updates"], 1_000)
            self.assertEqual(identity["presentations"], 16_000)
            self.assertEqual(identity["effective_batch_size"], 16)
            self.assertEqual(
                contract.validate_schedule_identity(identity, phase=phase),
                identity,
            )
        runtime = contract.runtime_authorization_template()
        self.assertEqual(contract.validate_runtime_inputs(runtime), runtime)
        self.assertFalse(
            runtime["raw"]["phase_a_grant"][
                "general_raw_v13_frame_loader_authorized"
            ]
        )

    def test_phase_a_terminal_and_continuation_pass(self) -> None:
        metrics = _passing_phase_a_metrics()
        update0 = _update0_metrics()
        terminal = contract.evaluate_phase_a(
            metrics, update0, _observation_integrity()
        )
        self.assertTrue(terminal["passed"])
        self.assertEqual(
            terminal["control"],
            contract.CONTROL_PHASE_A_PASS,
        )
        self.assertEqual(
            terminal["local_correspondence"],
            _local_correspondence(),
        )
        for update in (100, 400):
            result = contract.evaluate_phase_a_continuation(
                update, metrics, update0, _observation_integrity()
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["control"], contract.CONTROL_CONTINUE)

    def test_each_local_correspondence_mechanism_gate_fails_closed(self) -> None:
        cases: list[tuple[str, str, Any]] = [
            (
                "transport_weight_finite_and_bitwise_nonzero",
                "transport_weight_any_nonzero",
                False,
            ),
            (
                "correct_correspondence_cross_entropy_strictly_below_"
                "update_zero",
                "correct_centered_log_cross_entropy",
                math.log(9.0),
            ),
            (
                "deranged_correspondence_margin_positive_in_at_least_six_"
                "families",
                "deranged_family_count",
                5,
            ),
            (
                "hardest_wrong_correspondence_margin_positive_in_six_"
                "families",
                "hardest_family_count",
                5,
            ),
            (
                "all_eight_non_hold_distributions_differ_from_hold",
                "active_count",
                7,
            ),
            (
                "hold_uniform_zero_offset_and_identity_transport_exact",
                "hold_transport_identity_exact",
                False,
            ),
            (
                "expected_offset_components_within_closed_unit_bound",
                "maximum_absolute_expected_offset_component",
                1.000001,
            ),
            (
                "local_correspondence_values_finite_positive_and_normalized",
                "target_rows_normalized",
                False,
            ),
        ]
        for conjunct, field, value in cases:
            with self.subTest(conjunct=conjunct):
                metrics = _passing_phase_a_metrics()
                observation = metrics["local_correspondence"]
                if field == "deranged_family_count":
                    margins = observation[
                        "per_family_deranged_minus_correct_cross_entropy"
                    ]
                    margins[contract.SCENE_FAMILIES[5]] = 0.0
                    observation[
                        "deranged_positive_family_margin_count"
                    ] = value
                elif field == "hardest_family_count":
                    margins = observation[
                        "per_family_hardest_wrong_minus_executed_cross_entropy"
                    ]
                    margins[contract.SCENE_FAMILIES[5]] = 0.0
                    observation[
                        "hardest_wrong_positive_family_margin_count"
                    ] = value
                elif field == "active_count":
                    action = next(
                        item for item in contract.ACTION_VOCABULARY
                        if item != "hold"
                    )
                    observation[
                        "per_action_distribution_different_from_hold"
                    ][action] = False
                    observation[
                        "non_hold_action_distribution_different_from_hold_count"
                    ] = value
                else:
                    observation[field] = value
                    if field == "correct_centered_log_cross_entropy":
                        observation[
                            "correct_to_deranged_cross_entropy_ratio"
                        ] = (
                            value
                            / observation[
                                "deranged_centered_log_cross_entropy"
                            ]
                        )
                        observation[
                            "executed_to_hardest_wrong_cross_entropy_ratio"
                        ] = (
                            value
                            / observation[
                                "hardest_wrong_centered_log_cross_entropy"
                            ]
                        )
                result = contract.evaluate_phase_a(
                    metrics, _update0_metrics(), _observation_integrity()
                )
                self.assertFalse(result["passed"])
                self.assertFalse(result["conjuncts"][conjunct])

    def test_correspondence_ratios_are_strict_and_internally_bound(self) -> None:
        metrics = _passing_phase_a_metrics()
        corr = metrics["local_correspondence"]
        corr["deranged_centered_log_cross_entropy"] = (
            corr["correct_centered_log_cross_entropy"] / 0.99
        )
        corr["correct_to_deranged_cross_entropy_ratio"] = 0.99
        result = contract.evaluate_phase_a(
            metrics, _update0_metrics(), _observation_integrity()
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "correct_to_deranged_correspondence_ratio_strictly_below_"
                "point99"
            ]
        )

        metrics = _passing_phase_a_metrics()
        corr = metrics["local_correspondence"]
        corr["hardest_wrong_centered_log_cross_entropy"] = (
            corr["correct_centered_log_cross_entropy"] / 0.99
        )
        corr["executed_to_hardest_wrong_cross_entropy_ratio"] = 0.99
        result = contract.evaluate_phase_a(
            metrics, _update0_metrics(), _observation_integrity()
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "executed_to_hardest_wrong_correspondence_ratio_below_point99"
            ]
        )

        malformed = _passing_phase_a_metrics()
        malformed["local_correspondence"][
            "correct_to_deranged_cross_entropy_ratio"
        ] = 0.5
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                malformed, _update0_metrics(), _observation_integrity()
            )

    def test_update_zero_correspondence_is_exact_and_target_is_viable(
        self,
    ) -> None:
        update0 = _update0_metrics()
        update0["local_correspondence"]["mean_target_kl_to_uniform"] = 0.0
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )
        update0 = _update0_metrics()
        update0["local_correspondence"][
            "transport_weight_any_nonzero"
        ] = True
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )

    def test_forward_boundary_semantics_are_preserved(self) -> None:
        metrics = _passing_phase_a_metrics()
        threshold100 = contract.PHASE_A_UPDATE_100_THRESHOLDS
        metrics["centered_raw_patch_effective_rank"] = threshold100[
            "centered_raw_patch_effective_rank_strictly_greater_than"
        ]
        result = contract.evaluate_phase_a_continuation(
            100, metrics, _update0_metrics(), _observation_integrity()
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
        )

        metrics = _passing_phase_a_metrics()
        threshold400 = contract.PHASE_A_UPDATE_400_THRESHOLDS
        metrics["centered_raw_patch_effective_rank"] = threshold400[
            "centered_raw_patch_effective_rank_minimum"
        ]
        metrics["centered_projected_target_effective_rank"] = threshold400[
            "centered_projected_target_effective_rank_minimum"
        ]
        self.assertTrue(contract.evaluate_phase_a_continuation(
            400, metrics, _update0_metrics(), _observation_integrity()
        )["passed"])
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                1_000,
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

    def test_observation_integrity_and_populations_fail_closed(self) -> None:
        result = contract.evaluate_phase_a(
            _passing_phase_a_metrics(),
            _update0_metrics(),
            {"rng_state_preserved": False, "state_mutation_count": 0},
        )
        self.assertFalse(result["passed"])
        changed = _passing_phase_a_metrics()
        changed["pair_count"] = 494
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                changed, _update0_metrics(), _observation_integrity()
            )

    def test_phase_b_thresholds_remain_exact(self) -> None:
        passing = {
            "complete_physical_scope_count": 1,
            "margin_count": 189,
            "passed_margin_count": 98,
            "total_shortfall": 41.0,
            "rough_motion": {
                "pixel_balanced_accuracy": 0.82,
                "ground_balanced_accuracy": 0.648,
                "depth_p95_m": 0.977,
            },
        }
        self.assertTrue(contract.evaluate_phase_b(passing)["passed"])
        passing["total_shortfall"] = 41.01776266878769
        self.assertFalse(contract.evaluate_phase_b(passing)["passed"])

    def test_manifest_review_and_authorization_templates_bind_v7(self) -> None:
        manifest = contract.with_content_sha256(_source_manifest_core())
        raw = contract.canonical_json_bytes(manifest) + b"\n"
        self.assertEqual(contract.validate_source_manifest(raw), manifest)
        manifest_binding = contract.artifact_binding(
            contract.SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=manifest["content_sha256"],
        )
        sources = {
            path: hashlib.sha256(path.encode("ascii")).hexdigest()
            for path in contract.SOURCE_PATHS
        }
        sources[contract.SOURCE_MANIFEST_RELATIVE_PATH] = (
            manifest_binding["file_sha256"]
        )
        sources[contract.PREREGISTRATION_RELATIVE_PATH] = (
            contract.PREREGISTRATION_FILE_SHA256
        )
        sources[contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] = (
            contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
        )
        review_core = {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS_SOURCE_AND_SCIENCE",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_reviewer",
            "reviewed_sources": sources,
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
            "scientific_checks":
                dict(contract.SCIENTIFIC_REVIEW_CHECKS),
            "findings": [],
            "authority": dict(contract.REVIEW_AUTHORITY),
        }
        review = contract.with_content_sha256(review_core)
        self.assertEqual(
            contract.validate_review(review, expected_sources=sources),
            review,
        )
        review_raw = contract.canonical_json_bytes(review) + b"\n"
        review_binding = contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=review["content_sha256"],
        )
        authorization = contract.with_content_sha256({
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED_ONE_EXACT_TWO_PHASE_PROBE",
            "authorizer": "/root/independent_authorizer",
            "independent_source_review": review_binding,
            "preregistration": contract.preregistration_binding(),
            "runtime_inputs": contract.runtime_authorization_template(),
            "experiment": contract.science_contract(),
            "authority": dict(contract.EXECUTION_AUTHORITY),
        })
        self.assertEqual(
            contract.validate_authorization(
                authorization,
                review_binding=review_binding,
                reviewer="/root/independent_reviewer",
            ),
            authorization,
        )

    def test_authorities_deny_retry_and_downstream_access(self) -> None:
        self.assertTrue(
            all(
                value is False
                for value in contract.SOURCE_ONLY_AUTHORITY.values()
            )
        )
        self.assertFalse(contract.EXECUTION_AUTHORITY["g2_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["heldout_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["sealed_authorized"])
        self.assertFalse(
            contract.EXECUTION_AUTHORITY[
                "retry_resume_second_seed_schedule_extension_or_"
                "replacement_authorized"
            ]
        )

    def test_static_phase_b_adapter_remains_hash_bound(self) -> None:
        raw = (
            ROOT / contract.STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            contract.STATIC_PHYSICAL_CONTRACT_FILE_SHA256,
        )
        self.assertEqual(len(contract.SCOPES), 9)
        self.assertEqual(contract.MARGIN_COUNT, 189)


if __name__ == "__main__":
    unittest.main()
