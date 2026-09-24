from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_overlapping_tokenization_v1_contract",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_MODULES_IMPORTED_BY_CONTRACT = set(sys.modules) - _MODULES_BEFORE


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
                    "overlapping_tokenization_training",
                    "physical_checkpoint_selection",
                ],
                "calibration_g2_navigation_heldout_or_production_use": False,
            },
        },
        "camera": {
            "root": contract.N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(contract.N320_GATE_RELATIVE_PATH),
            "checkpoint": _runtime_leaf(
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ),
            "seed": 20_260_710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(contract.SCHEDULE_RELATIVE_PATH),
    }


def _evaluation() -> dict[str, Any]:
    rows = {
        scope: {"physical_margins": [-1.0] * 21, "passes": False}
        for scope in contract.SCOPES
    }
    return {
        "scope_evaluations": rows,
        "complete_physical_scope_count": 0,
        "margin_count": 189,
        "passed_margin_count": 0,
        "total_shortfall": 189.0,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": 0.5,
            "ground_balanced_accuracy": 0.5,
            "depth_p95_m": 1.0,
        },
    }


def _provisional_metric() -> dict[str, Any]:
    return {
        "update": 100,
        "role": "checkpoint_selection",
        "pair_count": 495,
        "unique_endpoint_count": 924,
        "scopes": {scope: {} for scope in contract.SCOPES},
        "aggregate_complete_v4_tail_depth_loss": 1.0,
        "evaluation": _evaluation(),
        "preledger_model_state_checks_pass": True,
        "state_sha256_before": "1" * 64,
        "state_sha256_after": "1" * 64,
        "frozen_state_sha256_before_and_after": "2" * 64,
        "state_mutation_count": 0,
    }


class OverlappingTokenizationContractTests(unittest.TestCase):
    def test_import_and_preregistration_are_exact_source_only(self) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }))
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "c88eadf269d9acc8c4ca87576fea48ce14721ee5",
        )
        prereg_raw = (
            ROOT / contract.PREREGISTRATION_RELATIVE_PATH
        ).read_bytes()
        review_raw = (
            ROOT / contract.PREREGISTRATION_REVIEW_RELATIVE_PATH
        ).read_bytes()
        self.assertEqual(len(prereg_raw), contract.PREREGISTRATION_BYTE_COUNT)
        self.assertEqual(
            hashlib.sha256(prereg_raw).hexdigest(),
            contract.PREREGISTRATION_FILE_SHA256,
        )
        self.assertEqual(
            len(review_raw), contract.PREREGISTRATION_REVIEW_BYTE_COUNT
        )
        self.assertEqual(
            hashlib.sha256(review_raw).hexdigest(),
            contract.PREREGISTRATION_REVIEW_FILE_SHA256,
        )
        review = json.loads(review_raw)
        core = dict(review)
        declared = core.pop("content_sha256")
        self.assertEqual(
            declared, contract.PREREGISTRATION_REVIEW_CONTENT_SHA256
        )
        self.assertEqual(contract.canonical_json_sha256(core), declared)
        self.assertEqual(review["verdict"], "PASS")
        self.assertIn(
            contract.TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH,
            contract.SOURCE_REVIEW_ADDITIONAL_PATHS,
        )

    def test_exact_overlap_topology_counts_and_hash(self) -> None:
        from lewm.models.shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1 import (  # noqa: E501
            overlapping_tokenization_architecture_contract_v1 as model_contract,
        )

        architecture = (
            contract.overlapping_tokenization_architecture_contract_v1()
        )
        self.assertEqual(
            architecture,
            contract.OVERLAPPING_TOKENIZATION_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(
            contract.canonical_json_sha256(architecture),
            contract.ARCHITECTURE_CONTRACT_SHA256,
        )
        self.assertEqual(architecture, model_contract())
        self.assertEqual(architecture["schema"], contract.ARCHITECTURE_SCHEMA)
        self.assertEqual(architecture["model_family"], contract.MODEL_FAMILY)
        patch = architecture["patch_projection"]
        self.assertEqual(patch["kernel_size"], [11, 11])
        self.assertEqual(patch["stride"], [7, 7])
        self.assertEqual(patch["padding"], [2, 2])
        self.assertEqual(patch["dilation"], [1, 1])
        self.assertEqual(patch["groups"], 1)
        self.assertIs(patch["bias"], True)
        self.assertEqual(patch["padding_mode"], "zeros")
        self.assertEqual(patch["center_copy_slice"], [2, 9, 2, 9])
        self.assertEqual(patch["central_weight_scalar_count"], 28_224)
        self.assertEqual(patch["outer_ring_scalar_count"], 41_472)
        self.assertEqual(
            architecture["token_geometry"]["patch_token_count"], 256
        )
        self.assertEqual(
            architecture["trainable"]["total_parameter_count"], 3_141_681
        )
        self.assertEqual(
            architecture["complete_model"]["parameter_count"], 7_049_460
        )

    def test_science_is_static_v3_except_preregistered_fields(self) -> None:
        baseline = contract._base_static_science_contract()
        science = contract.science_contract()
        preserved = deepcopy(science)
        preserved.pop("schema")
        preserved.pop("architecture_contract")
        preserved.pop("architecture_contract_sha256")
        preserved["model_family"] = baseline["model_family"]
        preserved["model_runtime_version"] = baseline["model_runtime_version"]
        preserved["one_science_delta"] = baseline["one_science_delta"]
        preserved["initialization"] = baseline["initialization"]
        preserved["parameter_counts"] = baseline["parameter_counts"]
        preserved["parameter_tensor_counts"] = (
            baseline["parameter_tensor_counts"]
        )
        self.assertEqual(preserved, baseline)
        self.assertEqual(
            science["one_science_delta"],
            "overlapping_rgb_patch_tokenization_relative_to_"
            "static_multires_v3_only",
        )
        self.assertEqual(science["initialization"]["exact_copy_entry_count"], 83)
        self.assertEqual(
            science["initialization"]["transformed_entry_count"], 1
        )
        self.assertEqual(
            science["initialization"]["n320_derived_entry_count"], 84
        )
        self.assertEqual(
            science["initialization"]["transformed_entry"],
            "encoder.patch_embed.weight",
        )
        self.assertEqual(science["parameter_counts"], {
            "evidence_head": 352_689,
            "encoder": 2_788_992,
            "total_trainable": 3_141_681,
            "complete_model": 7_049_460,
        })
        self.assertEqual(science["schedule"], baseline["schedule"])
        self.assertEqual(science["optimizer"], baseline["optimizer"])
        self.assertEqual(science["operation_cap"], baseline["operation_cap"])
        self.assertEqual(science["pass_thresholds"], baseline["pass_thresholds"])

    def test_identities_prior_roots_and_source_authority_fail_closed(self) -> None:
        self.assertEqual(
            contract.MODEL_FAMILY,
            "shared_observable_camera_ray_jepa_v5_multires_"
            "overlapping_tokenization_v1",
        )
        self.assertTrue(
            contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
                "/rgb_overlapping_tokenization_probe_v1"
            )
        )
        self.assertNotIn(
            contract.OUTPUT_ROOT_RELATIVE_PATH,
            contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
        )
        self.assertIn(
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_causal_motion_alignment_probe_v1",
            contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
        )
        self.assertTrue(
            set(contract._MOTION.PROHIBITED_RUNTIME_OUTPUT_ROOTS).issubset(
                contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
            )
        )
        self.assertTrue(all(
            value is False for value in contract.SOURCE_ONLY_AUTHORITY.values()
        ))
        self.assertTrue(all(
            value is False for value in contract.REVIEW_AUTHORITY.values()
        ))
        self.assertFalse(
            contract.SOURCE_ONLY_AUTHORITY["execution_authorized"]
        )
        self.assertFalse(
            contract.SOURCE_ONLY_AUTHORITY[
                "checkpoint_or_tensor_open_authorized"
            ]
        )
        self.assertFalse(
            contract.SOURCE_ONLY_AUTHORITY["gpu_or_hardware_authorized"]
        )

    def test_static_runtime_grant_and_corrected_ledger_lifecycle(self) -> None:
        runtime_inputs = _runtime_inputs()
        self.assertEqual(
            contract.validate_runtime_inputs(runtime_inputs), runtime_inputs
        )
        wrong = deepcopy(runtime_inputs)
        wrong["raw"]["grant"]["allowed_operations"][1] = (
            "causal_motion_alignment_training"
        )
        with self.assertRaises(PermissionError):
            contract.validate_runtime_inputs(wrong)
        lifecycle = contract.lifecycle_contract()
        self.assertTrue(
            lifecycle[
                "scientific_admissibility_requires_full_corrected_parser"
            ]
        )
        self.assertTrue(
            lifecycle[
                "full_parser_must_include_terminal_rehashes_and_terminal_record"
            ]
        )
        self.assertTrue(
            lifecycle["parser_failure_consumes_attempt_as_contract_invalid"]
        )
        self.assertFalse(
            lifecycle["preledger_metric_artifacts_scientifically_admissible"]
        )
        self.assertFalse(
            lifecycle["preledger_integrity_or_pass_fail_control_emitted"]
        )
        self.assertTrue(
            lifecycle[
                "terminal_control_materialized_after_finalized_parser_only"
            ]
        )
        self.assertTrue(
            lifecycle["execution_requires_future_exact_authorization"]
        )

    def test_static_provisional_metric_is_inadmissible_and_exact(self) -> None:
        metric = _provisional_metric()
        self.assertEqual(
            contract.validate_provisional_metric(metric, update=100), metric
        )
        for forbidden in ("integrity_pass", "temporal_population"):
            changed = deepcopy(metric)
            changed[forbidden] = True
            with self.assertRaises(PermissionError):
                contract.validate_provisional_metric(changed, update=100)
        contract.validate_no_preledger_scientific_control(metric)
        changed = deepcopy(metric)
        changed["integrity_pass"] = True
        with self.assertRaises(PermissionError):
            contract.validate_no_preledger_scientific_control(changed)
        for field, invalid in (
            ("complete_physical_scope_count", -1),
            ("margin_count", 0),
            ("passed_margin_count", -10),
            ("total_shortfall", -1.0),
        ):
            changed = deepcopy(metric)
            changed["evaluation"][field] = invalid
            with self.subTest(field=field):
                with self.assertRaises(PermissionError):
                    contract.validate_provisional_metric(
                        changed, update=100
                    )
        changed = deepcopy(metric)
        changed["evaluation"]["rough_motion"][
            "pixel_balanced_accuracy"
        ] = 99.0
        with self.assertRaises(PermissionError):
            contract.validate_provisional_metric(changed, update=100)

    def test_static_provisional_sidecar_validates_without_temporal_fields(
        self,
    ) -> None:
        metric = _provisional_metric()
        core = {
            "schema": contract.METRIC_SIDECAR_SCHEMA,
            "status":
                "PROVISIONAL_INADMISSIBLE_PENDING_FINALIZED_LEDGER_PARSE",
            "update": 100,
            "checkpoint": {
                "path": "checkpoints/update_100.pt",
                "file_sha256": "3" * 64,
                "content_sha256": "4" * 64,
                "byte_count": 1,
                "state_sha256": "5" * 64,
                "frozen_state_sha256": "6" * 64,
            },
            "metric": metric,
            "inline_evaluation_count": 1,
            "state_mutation_count": 0,
            "publication_order": [
                "cpu_snapshot",
                "inline_nonmutating_selection_evaluation",
                "atomic_mode_0444_provisional_sidecar",
                "internal_fixed_training_flow_only",
            ],
            "continuation": contract.provisional_checkpoint_control(100),
            "scientifically_admissible": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        }
        sidecar = contract.with_content_sha256(core)
        self.assertEqual(
            contract.validate_metric_sidecar(sidecar, update=100),
            sidecar,
        )
        changed = deepcopy(sidecar)
        changed["metric"]["temporal_population"] = {}
        changed = contract.with_content_sha256({
            key: value
            for key, value in changed.items()
            if key != "content_sha256"
        })
        with self.assertRaises(PermissionError):
            contract.validate_metric_sidecar(changed, update=100)


if __name__ == "__main__":
    unittest.main()
