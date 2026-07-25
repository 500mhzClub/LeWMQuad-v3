from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
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


def _latent_flow_observation(
    *,
    active_non_hold_count: int = 8,
    maximum_absolute_flow_cell: float = 0.5,
) -> dict[str, Any]:
    per_action = {
        action: (
            action != "hold"
            and index < active_non_hold_count
        )
        for index, action in enumerate(
            action
            for action in contract.ACTION_VOCABULARY
            if action != "hold"
        )
    }
    per_action = {
        action: per_action.get(action, False)
        for action in contract.ACTION_VOCABULARY
    }
    return {
        "all_values_finite": True,
        "all_components_within_closed_one_patch_bound":
            maximum_absolute_flow_cell <= contract.FLOW_CELL_BOUND,
        "hold_flow_exactly_zero": True,
        "maximum_absolute_flow_cell": maximum_absolute_flow_cell,
        "non_hold_action_nonzero_count": active_non_hold_count,
        "per_action_any_nonzero": per_action,
    }


def _inverse_dynamics_observation(
    *,
    update_zero: bool = False,
) -> dict[str, Any]:
    if update_zero:
        correct = float(math.log(len(contract.ACTION_VOCABULARY)))
        deranged = correct
        margins = {
            family: 0.0 for family in contract.SCENE_FAMILIES
        }
        return {
            "all_values_finite": True,
            "weight_any_nonzero": False,
            "maximum_absolute_logit": 0.0,
            "correct_unscaled_cross_entropy": correct,
            "deranged_unscaled_cross_entropy": deranged,
            "correct_to_deranged_cross_entropy_ratio": 1.0,
            "top1_accuracy": 1.0 / 9.0,
            "macro_balanced_accuracy": 1.0 / 9.0,
            "positive_family_margin_count": 0,
            "per_family_deranged_minus_correct_cross_entropy": margins,
        }
    correct = 2.0
    deranged = 2.1
    margins = {
        family: 0.01 if index < 6 else 0.0
        for index, family in enumerate(contract.SCENE_FAMILIES)
    }
    return {
        "all_values_finite": True,
        "weight_any_nonzero": True,
        "maximum_absolute_logit": 1.0,
        "correct_unscaled_cross_entropy": correct,
        "deranged_unscaled_cross_entropy": deranged,
        "correct_to_deranged_cross_entropy_ratio": correct / deranged,
        "top1_accuracy": 0.4,
        "macro_balanced_accuracy": 0.3,
        "positive_family_margin_count": 6,
        "per_family_deranged_minus_correct_cross_entropy": margins,
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
        "latent_flow": _latent_flow_observation(),
        "inverse_dynamics": _inverse_dynamics_observation(),
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
        "latent_flow": _latent_flow_observation(
            active_non_hold_count=0,
            maximum_absolute_flow_cell=0.0,
        ),
        "inverse_dynamics": _inverse_dynamics_observation(
            update_zero=True,
        ),
        "all_inverse_logits_bitwise_zero": True,
        "correct_and_deranged_cross_entropy_bitwise_equal": True,
    }


def _observation_integrity() -> dict[str, Any]:
    return {
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }


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


def _canonical_line(value: dict[str, Any]) -> bytes:
    return contract.canonical_json_bytes(value) + b"\n"


class JepaEncoderPretrainingContractTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_preregistration_is_exact(self) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch",
            "numpy",
            "PIL",
            "cv2",
            "jax",
            "tensorflow",
        }))
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "d552c16ebc813e49dca015354c080720b6bcd3a6",
        )
        self.assertEqual(
            contract.PREREGISTRATION_RELATIVE_PATH,
            "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_"
            "v6_existing_pair_inverse_dynamics_"
            "preregistration_2026-07-25.md",
        )
        raw = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
        self.assertEqual(contract.PREREGISTRATION_BYTE_COUNT, 14_178)
        self.assertEqual(
            contract.PREREGISTRATION_FILE_SHA256,
            "43efa19c33d81795386940d138f5609a7a381240c3b3952e6ed6094ce646f703",
        )
        self.assertEqual(len(raw), 14_178)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "43efa19c33d81795386940d138f5609a7a381240c3b3952e6ed6094ce646f703",
        )
        self.assertEqual(
            contract.SCHEMA_PREFIX,
            "lewm_go2_rgb_patch_whitened_action_residual_jepa_"
            "v6_existing_pair_inverse_dynamics",
        )
        self.assertEqual(contract.preregistration_binding(), {
            "path": contract.PREREGISTRATION_RELATIVE_PATH,
            "commit": contract.PREREGISTRATION_COMMIT,
            "file_sha256": contract.PREREGISTRATION_FILE_SHA256,
            "byte_count": contract.PREREGISTRATION_BYTE_COUNT,
        })
        self.assertEqual(contract.prior_terminal_audit_binding(), {
            "path":
                "docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_"
                "v5_state_dependent_latent_flow_terminal_audit_2026-07-25.json",
            "commit": "c7bd138cf9a7a8195c968199fcbda5025564fe55",
            "file_sha256":
                "aed6af35922d55cfa292c243fee8cf0cd27b43b01d96caca8bf12562f042406d",
            "content_sha256":
                "f55b6854ca873e49598909ce2f238f098945a11d320bbe9a734f59479add9164",
            "byte_count": 16_222,
        })
        self.assertEqual(contract.temporal_feasibility_audit_binding(), {
            "path":
                "docs/lewm_go2_rgb_causal_two_frame_latent_motion_state_"
                "source_feasibility_audit_2026-07-25.json",
            "commit": "6598b49e40be12da008de47590c129a611e8ae43",
            "file_sha256":
                "2e34e4f4ef22abdc8d5ecd89c1c12acff8541e41b8440a307ba8cbeb4a432c3d",
            "content_sha256":
                "2d947ea6857ddcbed82d778692c18831b2c920da6b3aa7c5723833ecb495a929",
            "byte_count": 3_261,
        })
        self.assertIn(
            contract.SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH,
            contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
        )
        self.assertEqual(
            contract.OUTPUT_ROOT_RELATIVE_PATH,
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_patch_whitened_action_residual_jepa_"
            "probe_v6_existing_pair_inverse_dynamics",
        )

    def test_phase_a_model_and_optimizer_contract_are_exact(self) -> None:
        config = contract.phase_a_model_config()
        self.assertEqual(config["latent_dim"], 192)
        self.assertEqual(config["image_size"], 112)
        self.assertEqual(config["patch_size"], 7)
        self.assertEqual(config["encoder_depth"], 6)
        self.assertEqual(config["encoder_heads"], 6)
        self.assertEqual(config["encoder_mlp_ratio"], 4)
        self.assertEqual(config["cmd_dim"], 9)
        self.assertEqual(config["pred_layers"], 2)
        self.assertEqual(config["pred_heads"], 6)
        self.assertEqual(config["pred_dim_head"], 32)
        self.assertEqual(config["pred_mlp_dim"], 384)
        self.assertEqual(config["target_geometry"], "patch")
        self.assertEqual(config["prediction_input_mode"], "state_action")
        self.assertEqual(config["target_ema_momentum"], 0.996)
        self.assertEqual(config["sigreg_projections"], 64)
        self.assertEqual(config["sigreg_knots"], 9)
        self.assertTrue(config["detach_action_control_state"])
        self.assertEqual(config["appearance_sigreg_lambda"], 0.0)
        self.assertEqual(config["spatial_variance_lambda"], 0.0)
        self.assertEqual(config["action_identifiability_lambda"], 0.0)
        self.assertEqual(config["zero_action_lambda"], 0.0)
        science = contract.science_contract()
        objective = science["phase_a"]["objective"]
        flow = science["initialization"][
            "shared_flow_projection"
        ]
        self.assertEqual(
            flow["path"],
            "prediction_projector.flow_weight",
        )
        self.assertEqual(flow["shape"], [2, 192])
        self.assertFalse(flow["bias"])
        self.assertEqual(flow["parameter_count"], 384)
        self.assertEqual(flow["value"], 0.0)
        self.assertEqual(flow["rng_draw_count"], 0)
        self.assertEqual(
            flow["output_component_order"],
            ["grid_x_column", "grid_y_row"],
        )
        self.assertEqual(
            contract.FLOW_PROJECTION_PARAMETER_COUNT,
            384,
        )
        self.assertEqual(contract.FLOW_PROJECTION_SHAPE, (2, 192))
        self.assertEqual(contract.FLOW_CELL_BOUND, 1.0)
        self.assertEqual(contract.FLOW_GRID_SCALE, 2.0 / 15.0)
        self.assertEqual(contract.FLOW_X_COMPONENT_INDEX, 0)
        self.assertEqual(contract.FLOW_Y_COMPONENT_INDEX, 1)
        self.assertEqual(
            science["initialization"]["existing_action_embedder"],
            {
                "path": "predictor.action_embed",
                "reused_without_reset": True,
                "used_outside_adaln_only": True,
                "trainable": True,
                "hold_relative_self_subtraction": True,
                "centered_prototypes_reused_by_inverse": True,
            },
        )
        inverse_projection = science["initialization"][
            "inverse_projection"
        ]
        self.assertEqual(
            inverse_projection["path"],
            "prediction_projector.inverse_weight",
        )
        self.assertEqual(inverse_projection["shape"], [192, 576])
        self.assertFalse(inverse_projection["bias"])
        self.assertEqual(inverse_projection["parameter_count"], 110_592)
        self.assertEqual(inverse_projection["value"], 0.0)
        self.assertEqual(inverse_projection["rng_draw_count"], 0)
        self.assertEqual(
            contract.INVERSE_PROJECTION_SHAPE,
            (192, 576),
        )
        self.assertEqual(
            contract.INVERSE_PROJECTION_PARAMETER_COUNT,
            110_592,
        )
        self.assertEqual(contract.INVERSE_DYNAMICS_LOSS_WEIGHT, 1.0)
        self.assertEqual(
            contract.PHASE_A_INVERSE_DYNAMICS_THRESHOLDS,
            {
                "correct_unscaled_cross_entropy_strictly_less_than":
                    math.log(9.0),
                "correct_to_deranged_cross_entropy_ratio_strictly_less_than":
                    0.99,
                "macro_balanced_accuracy_strictly_greater_than": 2.0 / 9.0,
                "positive_family_margin_count_minimum": 6,
            },
        )
        inverse_objective = objective["inverse_dynamics"]
        self.assertEqual(
            inverse_objective["normalized_difference"],
            "d_i=F.layer_norm(n_i-c_i,(192,),weight=None,"
            "bias=None,eps=1e-5)",
        )
        self.assertEqual(
            inverse_objective["concatenation"],
            "x_i=concat(c_i,n_i,d_i)_in_that_exact_order",
        )
        self.assertEqual(inverse_objective["input_dimension"], 576)
        self.assertEqual(
            inverse_objective["projection_path"],
            "prediction_projector.inverse_weight",
        )
        self.assertEqual(
            inverse_objective["projection_shape"],
            [192, 576],
        )
        self.assertEqual(
            inverse_objective["centered_action_prototypes"],
            "p_a=E(a)-mean_b(E(b))",
        )
        self.assertEqual(
            inverse_objective["logits"],
            "ell_a=dot(q,p_a)/sqrt(192)",
        )
        self.assertEqual(inverse_objective["loss_weight"], 1.0)
        self.assertTrue(inverse_objective["training_only"])
        self.assertEqual(inverse_objective["ema_tensor_count"], 0)
        self.assertEqual(contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT, 1.0)
        self.assertEqual(
            objective["action_indexed_energy_nll_weight"],
            contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT,
        )
        self.assertEqual(
            objective["detached_row_energy_scale_epsilon"],
            1e-8,
        )
        self.assertEqual(
            objective["detached_row_energy_scale"],
            "m_i=stop_gradient(mean_a(E_i_a)).clamp_min(1e-8)",
        )
        self.assertEqual(
            objective["action_indexed_energy_nll"],
            "mean_i(m_i*cross_entropy("
            "-E_i_all/m_i,executed_action_i))",
        )
        self.assertFalse(
            objective["action_independent_shared_trunk"][
                "action_embedder_passed_to_adaln"
            ]
        )
        self.assertTrue(
            objective["action_independent_shared_trunk"][
                "action_embedder_reused_by_flow"
            ]
        )
        self.assertEqual(
            objective["action_independent_shared_trunk"][
                "block_conditioning"
            ],
            "same_exact_all_zero_tensor_for_all_rows_and_candidates",
        )
        self.assertFalse(
            objective["fixed_temperature_or_temperature_sweep"]
        )
        self.assertEqual(objective["wrong_action_hinge_count"], 0)
        self.assertEqual(objective["hold_action_hinge_count"], 0)
        self.assertFalse(
            objective["executed_action_shared_h_and_r_shared_detached"]
        )
        self.assertTrue(
            objective["wrong_action_shared_h_and_r_shared_detached"]
        )
        self.assertFalse(
            objective["wrong_action_flow_projection_detached"]
        )
        self.assertFalse(
            objective["wrong_action_action_embedder_detached"]
        )
        latent_flow = objective["latent_flow_transition"]
        self.assertEqual(
            latent_flow["cell_displacement"],
            "delta_cell_i_a=tanh(W_flow*u_i_a)",
        )
        self.assertEqual(
            latent_flow["cell_displacement_closed_bound"],
            [-1.0, 1.0],
        )
        self.assertEqual(
            latent_flow["normalized_grid_displacement"],
            "delta_grid_i_a=(2/15)*delta_cell_i_a",
        )
        self.assertEqual(latent_flow["grid_shape"], [16, 16])
        self.assertEqual(
            latent_flow["flow_output_component_order"],
            ["grid_x_column", "grid_y_row"],
        )
        self.assertEqual(latent_flow["grid_sample"], {
            "mode": "bilinear",
            "padding_mode": "border",
            "align_corners": True,
            "coordinates": "identity_grid_plus_delta_grid_i_a",
        })
        self.assertTrue(
            latent_flow["hold_flow_exactly_zero_by_self_subtraction"]
        )
        self.assertEqual(latent_flow["per_action_flow_bank_count"], 0)
        self.assertEqual(objective["residual_scale"], contract.RESIDUAL_SCALE)
        self.assertTrue(objective["ema_current_skip_stop_gradient"])
        self.assertTrue(objective["ema_next_target_stop_gradient"])
        self.assertTrue(objective["appearance_projector_frozen"])
        self.assertEqual(
            objective["whitening"]["variance_weight"],
            contract.WHITENING_VARIANCE_WEIGHT,
        )
        self.assertEqual(
            objective["whitening"]["covariance_weight"],
            contract.WHITENING_COVARIANCE_WEIGHT,
        )
        self.assertEqual(
            science["phase_a"]["training_action_candidates"],
            "all_nine_real_actions_in_frozen_vocabulary_order",
        )
        self.assertEqual(
            science["phase_a"]["training_action_input"],
            "executed_action_index_and_uniformly_ordered_nine_energies_only",
        )
        optimizer = science["phase_a"]["optimizer"]
        self.assertEqual(optimizer["encoder_learning_rate"], 1e-4)
        self.assertEqual(optimizer["other_learning_rate"], 3e-4)
        self.assertEqual(
            optimizer["determinism_compatibility"],
            {
                "strict_deterministic_algorithms_before_and_after": True,
                "warn_only_scope":
                    "phase_a_training_and_checkpoint_selection",
                "sole_permitted_warning_prefix":
                    contract.PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX,
                "unexpected_warning_count": 0,
            },
        )
        self.assertEqual(
            tuple(optimizer["other_prefixes"]),
            contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES,
        )
        self.assertEqual(optimizer["inverse_projection"], {
            "path": "prediction_projector.inverse_weight",
            "parameter_tensor_count": 1,
            "parameter_count": 110_592,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "included_in_global_clip_norm": True,
            "phase_b_optimizer_included": False,
        })
        self.assertIn(
            "appearance_projector.",
            contract.PHASE_A_FROZEN_PARAMETER_PREFIXES,
        )
        self.assertNotIn(
            "appearance_projector.",
            contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES,
        )
        self.assertEqual(
            contract.PHASE_B_TRAINABLE_PARAMETER_PREFIXES,
            ("evidence_head.",),
        )
        self.assertFalse(
            science["phase_b"]["promotable_shared_v5_checkpoint"]
        )
        self.assertTrue(science["phase_b"]["inverse_projection_frozen"])
        self.assertFalse(
            science["phase_b"]["inverse_projection_optimizer_included"]
        )
        self.assertFalse(
            science["phase_b"]["inverse_projection_copied_into_phase_b_model"]
        )
        self.assertEqual(science["phase_b"]["hard_sync"], {
            "count": 1,
            "copied_prefixes": ["target_encoder."],
            "forbidden_copy_prefixes": ["target_bev_decoder."],
            "target_bev_decoder_initialization_identity_verified_without_copy":
                True,
        })

    def test_exact_budgets_and_schedule_identities(self) -> None:
        self.assertEqual(contract.MAXIMUM_UPDATE, 1_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_UPDATE, 2_000)
        self.assertEqual(contract.MAXIMUM_PRESENTATIONS, 16_000)
        self.assertEqual(
            contract.CUMULATIVE_MAXIMUM_PRESENTATIONS,
            32_000,
        )
        expected_prefix = {
            "100":
                "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
            "400":
                "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
            "1000":
                "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
        }
        phase_a = contract.build_schedule_identity("phase_a")
        phase_b = contract.build_schedule_identity("phase_b")
        self.assertEqual(phase_a["prefix_sha256"], expected_prefix)
        self.assertEqual(phase_b["prefix_sha256"], expected_prefix)
        self.assertEqual(phase_a["seed"], 20260713)
        self.assertEqual(phase_a["presentations"], 16_000)
        self.assertTrue(phase_a["reuse_same_frozen_prefix_independently"])
        self.assertEqual(
            contract.validate_schedule_identity(phase_a, phase="phase_a"),
            phase_a,
        )
        changed = deepcopy(phase_b)
        changed["presentations"] += 1
        with self.assertRaises(PermissionError):
            contract.validate_schedule_identity(changed, phase="phase_b")
        with self.assertRaises(ValueError):
            contract.build_schedule_identity("phase_c")

    def test_runtime_authority_binds_corrected_raw_n320_and_schedule(self) -> None:
        runtime = contract.runtime_authorization_template()
        self.assertEqual(
            runtime["raw"]["manifest"]["byte_count"],
            311_598,
        )
        self.assertEqual(runtime["raw"]["audit"]["byte_count"], 26_975)
        self.assertEqual(
            runtime["camera"]["checkpoint"]["byte_count"],
            13_777_100,
        )
        self.assertFalse(
            runtime["raw"]["phase_a_grant"][
                "camera_supervision_array_open_authorized"
            ]
        )
        self.assertFalse(
            runtime["raw"]["phase_a_grant"][
                "general_raw_v13_frame_loader_authorized"
            ]
        )
        self.assertEqual(contract.validate_runtime_inputs(runtime), runtime)
        changed = deepcopy(runtime)
        changed["raw"]["manifest"]["byte_count"] = 1
        with self.assertRaises(PermissionError):
            contract.validate_runtime_inputs(changed)

    def test_phase_a_exact_boundary_passes(self) -> None:
        result = contract.evaluate_phase_a(
            _passing_phase_a_metrics(),
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["control"], contract.CONTROL_PHASE_A_PASS)
        self.assertTrue(all(result["conjuncts"].values()))
        self.assertEqual(
            result["ratios"]["raw_cross_sample_variance_to_update0"],
            0.25,
        )
        self.assertEqual(
            result["counts"][
                "cyclic_wrong_action_positive_family_count"
            ],
            6,
        )
        self.assertEqual(
            result["counts"]["hold_action_positive_family_count"],
            6,
        )
        self.assertEqual(
            result["counts"]["all_wrong_action_candidate_count"],
            3_960,
        )
        self.assertEqual(
            result["counts"]["non_hold_action_nonzero_flow_count"],
            8,
        )
        self.assertTrue(
            result["conjuncts"][
                "latent_flow_finite_within_bound_and_hold_exactly_zero"
            ]
        )
        self.assertTrue(
            result["conjuncts"][
                "update_zero_all_action_flows_exactly_zero"
            ]
        )
        self.assertTrue(
            result["conjuncts"][
                "inverse_update_zero_logits_and_cross_entropy_exact"
            ]
        )
        self.assertEqual(
            result["counts"]["inverse_positive_family_margin_count"],
            6,
        )
        self.assertEqual(
            result["inverse_dynamics"],
            _inverse_dynamics_observation(),
        )

    def test_phase_a_each_gate_fails_closed(self) -> None:
        mutations = {
            "finite": ("all_values_finite", False),
            "ema": ("ema_target_gradient_free", False),
            "raw_rank": ("centered_raw_patch_effective_rank", 47.999),
            "projected_rank":
                ("centered_projected_target_effective_rank", 47.999),
            "raw_health": ("raw_cross_sample_variance", 0.999),
            "spatial_health":
                ("content_residual_spatial_diversity", 1.999),
            "shuffled_next": ("shuffled_next_mse", 0.90),
            "mean_target": ("mean_target_mse", 0.90),
            "cyclic_wrong_action": ("cyclic_wrong_action_mse", 0.85),
            "hardest_wrong_action": ("hardest_wrong_action_mse", 0.85),
            "hold_action": ("hold_action_mse", 0.85),
            "shuffled_current": ("shuffled_current_mse", 0.85),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name):
                metrics = _passing_phase_a_metrics()
                metrics[field] = value
                result = contract.evaluate_phase_a(
                    metrics,
                    _update0_metrics(),
                    _observation_integrity(),
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["control"],
                    contract.CONTROL_PHASE_A_FAIL,
                )
        metrics = _passing_phase_a_metrics()
        metrics["per_family"][contract.SCENE_FAMILIES[5]][
            "cyclic_wrong_action_minus_true_mse"
        ] = 0.0
        self.assertFalse(
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )["passed"]
        )
        metrics = _passing_phase_a_metrics()
        metrics["per_family"][contract.SCENE_FAMILIES[5]][
            "hold_action_minus_non_hold_true_mse"
        ] = 0.0
        self.assertFalse(
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )["passed"]
        )

    def test_phase_a_invalid_populations_and_denominators_are_rejected(self) -> None:
        metrics = _passing_phase_a_metrics()
        metrics["pair_count"] = 494
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )
        metrics = _passing_phase_a_metrics()
        metrics["hold_action_rows_match_non_hold_rows"] = False
        self.assertFalse(
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )["passed"]
        )
        metrics = _passing_phase_a_metrics()
        metrics["hold_action_pair_count"] += 1
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )
        metrics = _passing_phase_a_metrics()
        metrics["all_wrong_action_candidate_count"] -= 1
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )
        metrics = _passing_phase_a_metrics()
        metrics["shuffled_next_mse"] = 0.0
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )
        update0 = _update0_metrics()
        update0["raw_cross_sample_variance"] = 0.0
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )
        update0 = _update0_metrics()
        update0["all_action_predictions_bitwise_equal"] = False
        result = contract.evaluate_phase_a(
            _passing_phase_a_metrics(),
            update0,
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "update_zero_all_action_predictions_bitwise_equal"
            ]
        )
        update0 = _update0_metrics()
        update0["all_action_unordered_pair_count"] = 35
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )
        metrics = _passing_phase_a_metrics()
        metrics["per_family"] = dict(reversed(metrics["per_family"].items()))
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

    def test_phase_a_observation_integrity_receipt_fails_closed(self) -> None:
        for name, integrity in {
            "rng_changed": {
                "rng_state_preserved": False,
                "state_mutation_count": 0,
            },
            "state_changed": {
                "rng_state_preserved": True,
                "state_mutation_count": 1,
            },
        }.items():
            with self.subTest(name=name):
                result = contract.evaluate_phase_a(
                    _passing_phase_a_metrics(),
                    _update0_metrics(),
                    integrity,
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["control"],
                    contract.CONTROL_PHASE_A_FAIL,
                )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                _update0_metrics(),
                {"rng_state_preserved": True},
            )

    def test_latent_flow_observation_is_exact_and_fails_closed(self) -> None:
        metrics = _passing_phase_a_metrics()
        metrics["latent_flow"]["all_values_finite"] = False
        result = contract.evaluate_phase_a(
            metrics,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "latent_flow_finite_within_bound_and_hold_exactly_zero"
            ]
        )

        metrics = _passing_phase_a_metrics()
        metrics["latent_flow"]["maximum_absolute_flow_cell"] = 1.01
        metrics["latent_flow"][
            "all_components_within_closed_one_patch_bound"
        ] = False
        result = contract.evaluate_phase_a(
            metrics,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])

        metrics = _passing_phase_a_metrics()
        metrics["latent_flow"]["hold_flow_exactly_zero"] = False
        metrics["latent_flow"]["per_action_any_nonzero"]["hold"] = True
        result = contract.evaluate_phase_a(
            metrics,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])

        metrics = _passing_phase_a_metrics()
        metrics["latent_flow"]["non_hold_action_nonzero_count"] = 7
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

        update0 = _update0_metrics()
        update0["latent_flow"] = _latent_flow_observation(
            active_non_hold_count=1,
            maximum_absolute_flow_cell=0.01,
        )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )

    def test_inverse_dynamics_observation_and_gates_fail_closed(self) -> None:
        metrics = _passing_phase_a_metrics()
        metrics["inverse_dynamics"].pop("top1_accuracy")
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

        metrics = _passing_phase_a_metrics()
        metrics["inverse_dynamics"][
            "correct_to_deranged_cross_entropy_ratio"
        ] += 1e-3
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

        metrics = _passing_phase_a_metrics()
        metrics["inverse_dynamics"][
            "deranged_unscaled_cross_entropy"
        ] = 0.0
        metrics["inverse_dynamics"][
            "correct_to_deranged_cross_entropy_ratio"
        ] = 0.0
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )

        for name, mutate in {
            "finite": lambda row: row.update(
                all_values_finite=False
            ),
            "weight": lambda row: row.update(weight_any_nonzero=False),
            "chance_ce": lambda row: (
                row.update({
                    "correct_unscaled_cross_entropy": math.log(9.0),
                    "deranged_unscaled_cross_entropy": 3.0,
                    "correct_to_deranged_cross_entropy_ratio":
                        math.log(9.0) / 3.0,
                })
            ),
            "ratio": lambda row: (
                row.update({
                    "deranged_unscaled_cross_entropy": 2.0 / 0.99,
                    "correct_to_deranged_cross_entropy_ratio": 0.99,
                })
            ),
            "macro": lambda row: row.update(
                macro_balanced_accuracy=2.0 / 9.0
            ),
        }.items():
            with self.subTest(name=name):
                metrics = _passing_phase_a_metrics()
                mutate(metrics["inverse_dynamics"])
                result = contract.evaluate_phase_a(
                    metrics,
                    _update0_metrics(),
                    _observation_integrity(),
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["control"],
                    contract.CONTROL_PHASE_A_FAIL,
                )

        metrics = _passing_phase_a_metrics()
        family = contract.SCENE_FAMILIES[5]
        metrics["inverse_dynamics"][
            "per_family_deranged_minus_correct_cross_entropy"
        ][family] = 0.0
        metrics["inverse_dynamics"]["positive_family_margin_count"] = 5
        result = contract.evaluate_phase_a(
            metrics,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "inverse_deranged_margin_positive_in_at_least_six_families"
            ]
        )

        update0 = _update0_metrics()
        update0["all_inverse_logits_bitwise_zero"] = False
        result = contract.evaluate_phase_a(
            _passing_phase_a_metrics(),
            update0,
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "inverse_update_zero_logits_and_cross_entropy_exact"
            ]
        )

        update0 = _update0_metrics()
        update0["inverse_dynamics"]["weight_any_nonzero"] = True
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a(
                _passing_phase_a_metrics(),
                update0,
                _observation_integrity(),
            )

    def test_hardest_wrong_action_terminal_gate_is_inclusive(self) -> None:
        metrics = _passing_phase_a_metrics()
        metrics["hardest_wrong_action_mse"] = (
            metrics["true_pair_mse"]
            / contract.PHASE_A_PASS_THRESHOLDS[
                "hardest_wrong_action_ratio_maximum"
            ]
        )
        for row in metrics["per_family"].values():
            row["hardest_wrong_action_minus_true_mse"] = -10.0
        result = contract.evaluate_phase_a(
            metrics,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(
            result["ratios"]["true_to_hardest_wrong_action"],
            0.95,
        )
        metrics["hardest_wrong_action_mse"] = (
            metrics["true_pair_mse"] / (0.95 + 1e-6)
        )
        self.assertFalse(
            contract.evaluate_phase_a(
                metrics,
                _update0_metrics(),
                _observation_integrity(),
            )["passed"]
        )

    def test_phase_a_update_100_continuation_gate_is_strict(self) -> None:
        thresholds = contract.PHASE_A_UPDATE_100_THRESHOLDS
        passing = _passing_phase_a_metrics()
        passing["centered_raw_patch_effective_rank"] = (
            thresholds[
                "centered_raw_patch_effective_rank_strictly_greater_than"
            ] + 1e-12
        )
        passing["centered_projected_target_effective_rank"] = (
            thresholds[
                "centered_projected_target_effective_rank_"
                "strictly_greater_than"
            ] + 1e-12
        )
        result = contract.evaluate_phase_a_continuation(
            100,
            passing,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["control"], contract.CONTROL_CONTINUE)

        exact_failures = {
            "raw_rank": (
                "centered_raw_patch_effective_rank",
                thresholds[
                    "centered_raw_patch_effective_rank_"
                    "strictly_greater_than"
                ],
            ),
            "projected_rank": (
                "centered_projected_target_effective_rank",
                thresholds[
                    "centered_projected_target_effective_rank_"
                    "strictly_greater_than"
                ],
            ),
            "cyclic_ratio": (
                "cyclic_wrong_action_mse",
                passing["true_pair_mse"]
                / thresholds[
                    "cyclic_wrong_action_ratio_strictly_less_than"
                ],
            ),
            "hardest_wrong_ratio": (
                "hardest_wrong_action_mse",
                passing["true_pair_mse"]
                / thresholds[
                    "hardest_wrong_action_ratio_strictly_less_than"
                ],
            ),
            "hold_ratio": (
                "hold_action_mse",
                passing["non_hold_true_pair_mse"]
                / thresholds["hold_action_ratio_strictly_less_than"],
            ),
            "mean_target_ratio": (
                "mean_target_mse",
                passing["true_pair_mse"]
                / thresholds["mean_target_ratio_strictly_less_than"],
            ),
        }
        for name, (field, value) in exact_failures.items():
            with self.subTest(name=name):
                metrics = deepcopy(passing)
                metrics[field] = value
                result = contract.evaluate_phase_a_continuation(
                    100,
                    metrics,
                    _update0_metrics(),
                    _observation_integrity(),
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["control"],
                    contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
                )

        inactive_flow = deepcopy(passing)
        inactive_flow["latent_flow"] = _latent_flow_observation(
            active_non_hold_count=7,
        )
        result = contract.evaluate_phase_a_continuation(
            100,
            inactive_flow,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "all_eight_non_hold_actions_have_nonzero_flow"
            ]
        )
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
        )

        inverse_failure = deepcopy(passing)
        inverse_failure["inverse_dynamics"][
            "macro_balanced_accuracy"
        ] = 2.0 / 9.0
        result = contract.evaluate_phase_a_continuation(
            100,
            inverse_failure,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "inverse_macro_balanced_accuracy_strictly_above_two_ninths"
            ]
        )
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
        )

        result = contract.evaluate_phase_a_continuation(
            100,
            passing,
            _update0_metrics(),
            {
                "rng_state_preserved": False,
                "state_mutation_count": 0,
            },
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
        )

    def test_phase_a_update_400_continuation_boundaries_are_inclusive(
        self,
    ) -> None:
        thresholds = contract.PHASE_A_UPDATE_400_THRESHOLDS
        passing = _passing_phase_a_metrics()
        passing["centered_raw_patch_effective_rank"] = thresholds[
            "centered_raw_patch_effective_rank_minimum"
        ]
        passing["centered_projected_target_effective_rank"] = thresholds[
            "centered_projected_target_effective_rank_minimum"
        ]
        passing["cyclic_wrong_action_mse"] = (
            passing["true_pair_mse"]
            / thresholds["cyclic_wrong_action_ratio_maximum"]
        )
        passing["hardest_wrong_action_mse"] = (
            passing["true_pair_mse"]
            / thresholds["hardest_wrong_action_ratio_maximum"]
        )
        passing["hold_action_mse"] = (
            passing["non_hold_true_pair_mse"]
            / thresholds["hold_action_ratio_maximum"]
        )
        result = contract.evaluate_phase_a_continuation(
            400,
            passing,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["control"], contract.CONTROL_CONTINUE)

        failing = deepcopy(passing)
        failing["centered_raw_patch_effective_rank"] -= 1e-12
        result = contract.evaluate_phase_a_continuation(
            400,
            failing,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        )

        failing = deepcopy(passing)
        failing["hardest_wrong_action_mse"] = (
            failing["true_pair_mse"]
            / (thresholds["hardest_wrong_action_ratio_maximum"] + 1e-6)
        )
        result = contract.evaluate_phase_a_continuation(
            400,
            failing,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        )

        failing = deepcopy(passing)
        failing["hold_action_mse"] = (
            failing["non_hold_true_pair_mse"]
            / (thresholds["hold_action_ratio_maximum"] + 1e-6)
        )
        result = contract.evaluate_phase_a_continuation(
            400,
            failing,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        )

        inverse_failure = deepcopy(passing)
        inverse_failure["inverse_dynamics"]["weight_any_nonzero"] = False
        result = contract.evaluate_phase_a_continuation(
            400,
            inverse_failure,
            _update0_metrics(),
            _observation_integrity(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"][
                "inverse_values_finite_and_weight_nonzero"
            ]
        )
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        )

        result = contract.evaluate_phase_a_continuation(
            400,
            passing,
            _update0_metrics(),
            {
                "rng_state_preserved": True,
                "state_mutation_count": 1,
            },
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["control"],
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                1_000,
                passing,
                _update0_metrics(),
                _observation_integrity(),
            )

    def test_phase_b_thresholds_are_strict_where_preregistered(self) -> None:
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
        result = contract.evaluate_phase_b(passing)
        self.assertTrue(result["passed"])
        self.assertEqual(result["control"], contract.CONTROL_PASS)
        strict_equalities = (
            ("total_shortfall", 41.01776266878769),
            (
                "rough_motion.pixel_balanced_accuracy",
                0.8198594673963917,
            ),
            (
                "rough_motion.ground_balanced_accuracy",
                0.647134926562893,
            ),
            ("rough_motion.depth_p95_m", 0.9777327477931971),
        )
        for name, value in strict_equalities:
            with self.subTest(name=name):
                row = deepcopy(passing)
                if name.startswith("rough_motion."):
                    row["rough_motion"][name.partition(".")[2]] = value
                else:
                    row[name] = value
                self.assertFalse(contract.evaluate_phase_b(row)["passed"])

    def test_canonical_json_and_bindings_fail_closed(self) -> None:
        value = contract.with_content_sha256({"schema": "example", "x": 1})
        raw = _canonical_line(value)
        self.assertEqual(
            contract.parse_canonical_json(raw, name="example"),
            value,
        )
        with self.assertRaises(ValueError):
            contract.parse_canonical_json(raw.rstrip(b"\n"), name="example")
        duplicate = (
            b'{"content_sha256":"' + b"0" * 64
            + b'","schema":"example","schema":"changed"}\n'
        )
        with self.assertRaises(ValueError):
            contract.parse_canonical_json(duplicate, name="duplicate")
        binding = contract.artifact_binding(
            "docs/example.json",
            raw,
            content_sha256=value["content_sha256"],
        )
        self.assertEqual(
            contract.validate_binding(binding, path="docs/example.json"),
            binding,
        )
        with self.assertRaises(ValueError):
            contract.validate_binding(binding, path="docs/other.json")

    def test_source_manifest_and_live_rehash_are_strict(self) -> None:
        manifest = contract.with_content_sha256(_source_manifest_core())
        raw = _canonical_line(manifest)
        self.assertEqual(contract.validate_source_manifest(raw), manifest)
        changed = deepcopy(manifest)
        changed["sealed_or_heldout_open_count"] = 1
        changed.pop("content_sha256")
        with self.assertRaises(PermissionError):
            contract.validate_source_manifest(
                _canonical_line(contract.with_content_sha256(changed))
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_bindings = []
            for path in contract.SOURCE_PATHS:
                payload = (path + "\n").encode("ascii")
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                live_bindings.append({
                    "path": path,
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                })
            live_core = _source_manifest_core()
            live_core["source_bindings"] = live_bindings
            live_core["source_bindings_sha256"] = (
                contract.canonical_json_sha256(live_bindings)
            )
            live = contract.with_content_sha256(live_core)
            manifest_path = root / contract.SOURCE_MANIFEST_RELATIVE_PATH
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_raw = _canonical_line(live)
            manifest_path.write_bytes(manifest_raw)
            preregistration_path = (
                root / contract.PREREGISTRATION_RELATIVE_PATH
            )
            preregistration_path.parent.mkdir(parents=True, exist_ok=True)
            preregistration_path.write_bytes(
                (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
            )
            prior_terminal_audit_path = (
                root / contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH
            )
            prior_terminal_audit_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            prior_terminal_audit_path.write_bytes(
                (
                    ROOT / contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH
                ).read_bytes()
            )
            temporal_feasibility_audit_path = (
                root / contract.TEMPORAL_FEASIBILITY_AUDIT_RELATIVE_PATH
            )
            temporal_feasibility_audit_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            temporal_feasibility_audit_path.write_bytes(
                (
                    ROOT
                    / contract.TEMPORAL_FEASIBILITY_AUDIT_RELATIVE_PATH
                ).read_bytes()
            )
            self.assertEqual(
                contract.current_source_bindings(root),
                {
                    **{
                        row["path"]: row["file_sha256"]
                        for row in live_bindings
                    },
                    contract.SOURCE_MANIFEST_RELATIVE_PATH:
                        hashlib.sha256(manifest_raw).hexdigest(),
                    contract.PREREGISTRATION_RELATIVE_PATH:
                        contract.PREREGISTRATION_FILE_SHA256,
                    contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH:
                        contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256,
                    contract.TEMPORAL_FEASIBILITY_AUDIT_RELATIVE_PATH:
                        contract.TEMPORAL_FEASIBILITY_AUDIT_FILE_SHA256,
                },
            )
            (root / contract.CONTRACT_RELATIVE_PATH).write_bytes(b"changed\n")
            with self.assertRaises(PermissionError):
                contract.current_source_bindings(root)

    def test_review_and_authorization_bind_exact_science_and_authority(
        self,
    ) -> None:
        manifest = contract.with_content_sha256(_source_manifest_core())
        manifest_raw = _canonical_line(manifest)
        manifest_binding = contract.artifact_binding(
            contract.SOURCE_MANIFEST_RELATIVE_PATH,
            manifest_raw,
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
        sources[contract.TEMPORAL_FEASIBILITY_AUDIT_RELATIVE_PATH] = (
            contract.TEMPORAL_FEASIBILITY_AUDIT_FILE_SHA256
        )
        review_core = {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS_SOURCE_AND_SCIENCE",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_reviewer",
            "reviewed_sources": sources,
            "source_manifest": manifest_binding,
            "preregistration": contract.preregistration_binding(),
            "prior_terminal_audit":
                contract.prior_terminal_audit_binding(),
            "temporal_feasibility_audit":
                contract.temporal_feasibility_audit_binding(),
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
        changed_review = deepcopy(review)
        changed_review["scientific_checks"][
            "continuation_gates_exact"
        ] = False
        changed_review.pop("content_sha256")
        changed_review = contract.with_content_sha256(changed_review)
        with self.assertRaises(PermissionError):
            contract.validate_review(
                changed_review,
                expected_sources=sources,
            )
        review_raw = _canonical_line(review)
        review_binding = contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=review["content_sha256"],
        )
        authorization_core = {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED_ONE_EXACT_TWO_PHASE_PROBE",
            "authorizer": "/root/independent_authorizer",
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
                reviewer="/root/independent_reviewer",
            ),
            authorization,
        )
        changed = deepcopy(authorization)
        changed["authority"]["heldout_authorized"] = True
        changed.pop("content_sha256")
        changed = contract.with_content_sha256(changed)
        with self.assertRaises(PermissionError):
            contract.validate_authorization(
                changed,
                review_binding=review_binding,
                reviewer="/root/independent_reviewer",
            )

    def test_source_and_review_authorities_deny_every_runtime_action(
        self,
    ) -> None:
        self.assertTrue(
            all(value is False for value in contract.SOURCE_ONLY_AUTHORITY.values())
        )
        self.assertEqual(
            contract.REVIEW_AUTHORITY,
            contract.SOURCE_ONLY_AUTHORITY,
        )
        self.assertFalse(contract.EXECUTION_AUTHORITY["g2_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["navigation_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["heldout_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["sealed_authorized"])
        self.assertFalse(
            contract.EXECUTION_AUTHORITY[
                "retry_resume_second_seed_schedule_extension_or_"
                "replacement_authorized"
            ]
        )
        self.assertTrue(
            contract.EXECUTION_AUTHORITY[
                "phase_b_only_after_exact_phase_a_pass_authorized"
            ]
        )

    def test_static_physical_adapter_is_hash_bound(self) -> None:
        raw = (
            ROOT / contract.STATIC_PHYSICAL_CONTRACT_RELATIVE_PATH
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            contract.STATIC_PHYSICAL_CONTRACT_FILE_SHA256,
        )
        self.assertEqual(len(contract.SCOPES), 9)
        self.assertEqual(contract.MARGIN_COUNT, 189)
        self.assertEqual(
            contract.learning_rates(100),
            contract._STATIC_PHYSICAL.learning_rates(100),
        )


if __name__ == "__main__":
    unittest.main()
