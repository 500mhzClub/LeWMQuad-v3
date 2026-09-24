from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_causal_motion_alignment_v1_contract",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_MODULES_IMPORTED_BY_CONTRACT = set(sys.modules) - _MODULES_BEFORE


def _evaluation(
    *,
    complete: int = 1,
    passed: int = 98,
    shortfall: float = 41.0,
    pixel: float = 0.82,
    ground: float = 0.65,
    depth: float = 0.97,
) -> dict[str, Any]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _runtime_leaf(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": contract.RUNTIME_FILE_SHA256[path],
        "content_sha256": contract.RUNTIME_CONTENT_SHA256[path],
        "byte_count": contract.RUNTIME_BYTE_COUNTS.get(path, 1),
    }


def _runtime_inputs() -> dict[str, Any]:
    return {
        "raw": {
            "root": contract.RAW_ROOT_RELATIVE_PATH,
            "manifest": _runtime_leaf(contract.RAW_MANIFEST_RELATIVE_PATH),
            "audit": _runtime_leaf(contract.RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": dict(contract.TRAIN_ROLE_COUNTS),
                "checkpoint_selection":
                    dict(contract.SELECTION_ROLE_COUNTS),
            },
            "grant": {
                "allowed_roles": ["train", "checkpoint_selection"],
                "allowed_operations": [
                    "development_rgb_decode",
                    "causal_motion_alignment_training",
                    "physical_checkpoint_selection",
                ],
                "calibration_g2_navigation_heldout_or_production_use": False,
            },
        },
        "camera": {
            "root": contract.N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(contract.N320_GATE_RELATIVE_PATH),
            "checkpoint":
                _runtime_leaf(contract.N320_CHECKPOINT_RELATIVE_PATH),
            "seed": 20_260_710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(contract.SCHEDULE_RELATIVE_PATH),
    }


class MotionAlignmentContractTests(unittest.TestCase):
    def test_import_and_preregistration_bindings_are_source_only_and_exact(
        self,
    ) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }))
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "a3cea116e5cdf6cfec3801624c51306742e0f0f5",
        )
        preregistration = ROOT / contract.PREREGISTRATION_RELATIVE_PATH
        review = ROOT / contract.PREREGISTRATION_REVIEW_RELATIVE_PATH
        preregistration_raw = preregistration.read_bytes()
        review_raw = review.read_bytes()
        self.assertEqual(
            len(preregistration_raw), contract.PREREGISTRATION_BYTE_COUNT
        )
        self.assertEqual(
            hashlib.sha256(preregistration_raw).hexdigest(),
            contract.PREREGISTRATION_FILE_SHA256,
        )
        self.assertEqual(len(review_raw), contract.PREREGISTRATION_REVIEW_BYTE_COUNT)
        self.assertEqual(
            hashlib.sha256(review_raw).hexdigest(),
            contract.PREREGISTRATION_REVIEW_FILE_SHA256,
        )
        review_value = json.loads(review_raw)
        review_core = dict(review_value)
        declared = review_core.pop("content_sha256")
        self.assertEqual(
            declared, contract.PREREGISTRATION_REVIEW_CONTENT_SHA256
        )
        self.assertEqual(contract.canonical_json_sha256(review_core), declared)
        self.assertEqual(review_value["verdict"], "PASS")

    def test_successor_source_model_output_and_prohibited_identities(self) -> None:
        self.assertEqual(
            contract.MODEL_RELATIVE_PATH,
            "lewm/models/"
            "shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1.py",
        )
        self.assertEqual(
            contract.RUNNER_RELATIVE_PATH,
            "scripts/run_go2_rgb_causal_motion_alignment_v1.py",
        )
        self.assertEqual(
            contract.LAUNCHER_RELATIVE_PATH,
            "scripts/launch_go2_rgb_causal_motion_alignment_v1.py",
        )
        self.assertEqual(
            contract.MODEL_FAMILY,
            "shared_observable_camera_ray_jepa_v5_"
            "multires_motion_alignment_v1",
        )
        self.assertTrue(contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
            "/rgb_causal_motion_alignment_probe_v1"
        ))
        self.assertNotIn(
            contract.OUTPUT_ROOT_RELATIVE_PATH,
            contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
        )
        self.assertIn(
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_causal_temporal_perception_probe_v1",
            contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
        )
        self.assertEqual(
            contract.FROZEN_SOURCE_SHA256[
                contract.TEMPORAL_CONTRACT_RELATIVE_PATH
            ],
            contract.TEMPORAL_CONTRACT_FILE_SHA256,
        )
        self.assertEqual(
            contract.FROZEN_SOURCE_SHA256[
                contract.TEMPORAL_MODEL_RELATIVE_PATH
            ],
            contract.TEMPORAL_MODEL_FILE_SHA256,
        )
        self.assertEqual(
            contract.FROZEN_SOURCE_SHA256[
                contract.TEMPORAL_RUNNER_RELATIVE_PATH
            ],
            contract.TEMPORAL_RUNNER_FILE_SHA256,
        )
        self.assertEqual(
            contract.FROZEN_SOURCE_SHA256[
                contract.TEMPORAL_LAUNCHER_RELATIVE_PATH
            ],
            contract.TEMPORAL_LAUNCHER_FILE_SHA256,
        )
        self.assertEqual(
            contract.FROZEN_SOURCE_SHA256[
                contract.TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH
            ],
            contract.TEMPORAL_SOURCE_MANIFEST_FILE_SHA256,
        )
        self.assertTrue(
            set(contract._TEMPORAL.FROZEN_SOURCE_SHA256).issubset(
                contract.FROZEN_SOURCE_SHA256
            )
        )
        self.assertTrue({
            contract.CONTRACT_RELATIVE_PATH,
            contract.MODEL_RELATIVE_PATH,
            contract.RUNNER_RELATIVE_PATH,
            contract.LAUNCHER_RELATIVE_PATH,
            contract.CONTRACT_TEST_RELATIVE_PATH,
            contract.RECEIPT_BOUNDARY_TEST_RELATIVE_PATH,
            contract.TEMPORAL_CONTRACT_RELATIVE_PATH,
            contract.TEMPORAL_RUNNER_RELATIVE_PATH,
            contract.TEMPORAL_LAUNCHER_RELATIVE_PATH,
        }.issubset(contract.SOURCE_PATHS))

    def test_exact_alignment_science_capacity_and_causal_inputs(self) -> None:
        science = contract.science_contract()
        mechanism = science["motion_alignment_mechanism"]
        self.assertEqual(
            science["one_science_delta"],
            "causal_motion_conditioned_dense_previous_token_alignment_"
            "before_retained_temporal_residual",
        )
        self.assertEqual(
            mechanism["inputs"],
            [
                "previous_raw_visual_tokens_at_fixed_lag",
                "current_raw_visual_tokens",
                "causal_motion_condition_5d",
                "history_valid",
            ],
        )
        self.assertEqual(
            mechanism["condition_components"],
            [
                "nominal_forward_m",
                "nominal_left_m",
                "nominal_yaw_rad",
                "relative_roll_rad",
                "relative_pitch_rad",
            ],
        )
        forbidden = set(mechanism["forbidden_inputs"])
        self.assertTrue({
            "outgoing_primitive",
            "per_sample_realized_relative_se2_current_frame",
            "exact_simulator_pose",
            "future_realized_motion",
            "prior_run_output",
            "failed_temporal_or_multiresolution_checkpoint",
        }.issubset(forbidden))
        self.assertTrue(mechanism["incoming_predecessor_primitive_only"])
        self.assertTrue(mechanism["cold_history_exact_bypass"])
        self.assertFalse(
            mechanism["per_sample_realized_se2_materialized_to_model"]
        )
        self.assertEqual(
            science["initialization"]["alignment_local_cpu_seed"],
            20_260_726,
        )
        self.assertEqual(science["parameter_counts"], {
            "evidence_head": 368_681,
            "encoder": 2_747_520,
            "alignment": 12_832,
            "temporal": 3_160,
            "changed_post_encoder": 15_992,
            "total_trainable": 3_116_201,
        })
        self.assertEqual(science["parameter_tensor_counts"], {
            "evidence_head": 35,
            "encoder": 78,
            "alignment": 4,
            "temporal": 5,
            "changed_post_encoder": 9,
            "total_trainable": 113,
        })
        self.assertEqual(
            contract.EVIDENCE_HEAD_PARAMETER_CEILING, 368_681
        )

    def test_roles_optimizer_schedule_losses_and_caps_are_inherited_exactly(
        self,
    ) -> None:
        science = contract.science_contract()
        self.assertEqual(science["data"]["train"], {
            "pairs": 4_262,
            "unique_endpoints": 7_777,
            "scenes": 72,
        })
        self.assertEqual(science["data"]["checkpoint_selection"], {
            "pairs": 495,
            "unique_endpoints": 924,
            "warm_endpoints": 495,
            "cold_endpoints": 429,
            "both_roles": 66,
            "ambiguous_predecessors": 0,
            "scenes": 8,
        })
        self.assertEqual(science["optimizer"], {
            "name": "AdamW",
            "group_order": ["evidence_head", "encoder"],
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 1e-4,
            "amsgrad": False,
            "precision": "float32",
            "autocast": False,
            "encoder_learning_rate_scale": 1.0,
            "learning_rate_horizon_updates": 8_000,
            "independent_group_clip_norm": 1.0,
            "microbatch_size": 4,
            "microbatches_per_update": 4,
        })
        self.assertEqual(science["schedule"]["maximum_updates"], 1_000)
        self.assertEqual(science["schedule"]["maximum_presentations"], 16_000)
        self.assertEqual(science["schedule"]["checkpoints"], [100, 400, 1_000])
        self.assertEqual(
            science["schedule"]["prefix_sha256"],
            {
                "100":
                    "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
                "400":
                    "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
                "1000":
                    "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
            },
        )
        self.assertEqual(science["jepa_objective_count"], 0)
        self.assertEqual(science["jepa_backward_count"], 0)
        self.assertEqual(science["prior_runtime_output_open_count"], 0)
        runtime_inputs = _runtime_inputs()
        self.assertEqual(
            contract.validate_runtime_inputs(runtime_inputs), runtime_inputs
        )

    def test_six_part_gate_and_full_parser_lifecycle_are_exact(self) -> None:
        passed = contract.checkpoint_control_decision(
            update=1_000,
            evaluation=_evaluation(),
            integrity_pass=True,
        )
        self.assertEqual(passed["action"], contract.CONTROL_PASS)
        self.assertEqual(len(passed["conjuncts"]), 6)
        equality = contract.checkpoint_control_decision(
            update=1_000,
            evaluation=_evaluation(
                shortfall=contract.PASS_THRESHOLDS[
                    "total_shortfall_strictly_less_than"
                ],
                pixel=contract.PASS_THRESHOLDS[
                    "rough_pixel_balanced_accuracy_strictly_greater_than"
                ],
                ground=contract.PASS_THRESHOLDS[
                    "rough_ground_balanced_accuracy_strictly_greater_than"
                ],
                depth=contract.PASS_THRESHOLDS[
                    "rough_depth_p95_m_strictly_less_than"
                ],
            ),
            integrity_pass=True,
        )
        self.assertEqual(equality["action"], contract.CONTROL_FAIL)
        lifecycle = contract.lifecycle_contract()
        self.assertTrue(
            lifecycle["scientific_admissibility_requires_full_corrected_parser"]
        )
        self.assertTrue(
            lifecycle["full_parser_must_include_terminal_rehashes_and_terminal_record"]
        )
        self.assertTrue(
            lifecycle["parser_failure_consumes_attempt_as_contract_invalid"]
        )
        self.assertFalse(
            lifecycle[
                "preledger_metric_artifacts_scientifically_admissible"
            ]
        )
        self.assertFalse(
            lifecycle["preledger_integrity_or_pass_fail_control_emitted"]
        )
        self.assertTrue(
            lifecycle[
                "terminal_control_materialized_after_finalized_parser_only"
            ]
        )
        self.assertFalse(
            lifecycle["receipt_validator_repair_science_change"]
        )
        self.assertTrue(all(
            value is False for value in contract.SOURCE_ONLY_AUTHORITY.values()
        ))


if __name__ == "__main__":
    unittest.main()
