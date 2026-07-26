from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_direct_egocentric_bev_state_jepa_v1_contract",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_IMPORTED = set(sys.modules) - _MODULES_BEFORE


def _update_zero_metrics() -> dict[str, object]:
    return {
        "three_logit_bottleneck_exact": True,
        "no_hidden_or_auxiliary_bypass": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "target_parameters_gradient_free": True,
        "intended_online_path_gradient_nonzero": True,
        "six_call_graph_isolation_exact": True,
        "all_registered_values_finite": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "G": 1.0,
        "J": 1.0,
    }


def _update_100_metrics() -> dict[str, object]:
    return {
        "G": 0.99,
        "J": 0.99,
        "action_nll": math.nextafter(math.log(9.0), -math.inf),
        "action_macro_balanced_accuracy": math.nextafter(
            1.0 / 9.0, math.inf
        ),
        "correct_rgb_scene_win_count": 6,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
    }


def _update_400_metrics() -> dict[str, object]:
    return {
        "G": 0.90,
        "J": 0.90,
        "action_nll": math.nextafter(0.99 * math.log(9.0), -math.inf),
        "action_macro_balanced_accuracy": math.nextafter(0.15, math.inf),
        "hardest_wrong_positive_scene_count": 4,
        "same_action_target_nll": math.nextafter(
            0.99 * math.log(2.0), -math.inf
        ),
        "same_action_target_strict_win_rate": 0.60,
        "correct_rgb_scene_win_count": 8,
    }


def _update_1000_metrics() -> dict[str, object]:
    threshold = contract.GATE_THRESHOLDS[1_000]
    return {
        "aggregate_raster_balanced_accuracy": math.nextafter(
            threshold[
                "aggregate_raster_balanced_accuracy_strictly_greater_than"
            ],
            math.inf,
        ),
        "aggregate_free_recall": math.nextafter(
            threshold["aggregate_free_recall_strictly_greater_than"],
            math.inf,
        ),
        "aggregate_occupied_recall": math.nextafter(
            threshold["aggregate_occupied_recall_strictly_greater_than"],
            math.inf,
        ),
        "aggregate_raster_nll": math.nextafter(
            threshold["aggregate_raster_nll_strictly_less_than"],
            -math.inf,
        ),
        "rough_raster_balanced_accuracy": math.nextafter(
            threshold[
                "rough_raster_balanced_accuracy_strictly_greater_than"
            ],
            math.inf,
        ),
        "rough_raster_occupied_recall": math.nextafter(
            threshold[
                "rough_raster_occupied_recall_strictly_greater_than"
            ],
            math.inf,
        ),
        "correct_rgb_scene_win_count": 8,
        "action_nll": math.nextafter(
            threshold["action_nll_strictly_less_than"], -math.inf
        ),
        "action_macro_balanced_accuracy": math.nextafter(
            threshold[
                "action_macro_balanced_accuracy_strictly_greater_than"
            ],
            math.inf,
        ),
        "hardest_wrong_positive_scene_count": 6,
        "same_action_target_nll": math.nextafter(
            threshold["same_action_target_nll_strictly_less_than"],
            -math.inf,
        ),
        "same_action_target_strict_win_rate": 0.65,
        "target_positive_scene_count": 6,
    }


class DirectBevContractTests(unittest.TestCase):
    def test_import_is_stdlib_source_only(self) -> None:
        forbidden = {"torch", "numpy", "PIL"}
        self.assertFalse(forbidden & {name.split(".", 1)[0] for name in _IMPORTED})
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "4831f77d9ddae15fa8504ffb1d06f73e0af427a4",
        )
        self.assertFalse(contract.SOURCE_ONLY_AUTHORITY["execution_authorized"])

    def test_governing_document_hashes_and_independent_review(self) -> None:
        bindings = contract.validate_governing_documents(ROOT)
        self.assertEqual(len(bindings), 4)
        self.assertEqual(
            bindings[contract.PREREGISTRATION_RELATIVE_PATH],
            "6863041a0a498a297c92c011ede97f1ffebaeb2121eebbf61054243a185bb3c0",
        )
        self.assertEqual(contract.PREREGISTRATION_BYTE_COUNT, 21_561)
        self.assertEqual(
            contract.PREREGISTRATION_CONTENT_SHA256,
            "8d97a9f4769f8ce1ebf69c7665c9fe4a57f693eb5b02acd6a9b3224b796c0943",
        )
        self.assertEqual(contract.ARCHITECTURE_DECISION_BYTE_COUNT, 16_096)
        self.assertFalse(
            contract.SOURCE_ONLY_AUTHORITY["generated_input_access_authorized"]
        )

    def test_exact_data_mapping_fallbacks_and_families(self) -> None:
        train = contract.TARGET_MAPPING_BINDINGS["train"]
        selection = contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
        self.assertEqual(train["same_action_eligible_count"], 4_237)
        self.assertEqual(train["fallback_count"], 25)
        self.assertEqual(4_237 + 25, 4_262)
        self.assertEqual(selection["same_action_eligible_count"], 494)
        self.assertEqual(selection["fallback_count"], 1)
        self.assertEqual(494 + 1, 495)
        self.assertEqual(
            train["mapping_sha256"],
            "c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b",
        )
        self.assertEqual(
            selection["mapping_sha256"],
            "95d42273a8319316ad68781cb2158146e7672eda529984c3aeddc0937d87a9c1",
        )
        self.assertEqual(
            sum(item["row_count"] for item in contract.SELECTION_FAMILY_BINDINGS.values()),
            495,
        )
        self.assertEqual(
            sum(
                item["same_action_row_count"]
                for item in contract.SELECTION_FAMILY_BINDINGS.values()
            ),
            494,
        )
        self.assertEqual(contract.FIXED_MAPPED_NEGATIVE_RULE["random_draw_count"], 0)
        self.assertIn("complete_same_scene", contract.FIXED_MAPPED_NEGATIVE_RULE["singleton"])

    def test_six_call_graph_has_no_leakage_or_hidden_bypass(self) -> None:
        self.assertEqual(
            set(contract.CALL_GRAPH),
            {
                "O_current_rgb",
                "O_next_rgb",
                "T_next_rgb",
                "T_current_rgb",
                "T_fixed_negative_rgb",
                "O_fixed_negative_rgb",
            },
        )
        self.assertEqual(contract.CALL_GRAPH["O_next_rgb"]["consumers"], ["G_next"])
        self.assertEqual(
            contract.CALL_GRAPH["O_fixed_negative_rgb"]["consumers"],
            ["wrong_rgb_grounding_control_only"],
        )
        self.assertEqual(contract.CALL_GRAPH["T_next_rgb"]["gradient"], "none")
        causal = [
            name
            for name, value in contract.CALL_GRAPH.items()
            if value["causal_or_deployment_path"]
        ]
        self.assertEqual(causal, ["O_current_rgb"])
        self.assertEqual(
            contract.CAUSAL_RUNTIME_INPUTS,
            ("current_rgb", "executed_action_identity"),
        )
        self.assertIn("hidden_64d_decoder_bypass", contract.FORBIDDEN_CALL_GRAPH_CONSUMERS)

    def test_model_and_parameter_inventory_are_exact(self) -> None:
        model = contract.model_config()
        self.assertEqual(model["state_head"]["classes_in_order"], ["UNKNOWN", "FREE", "OCCUPIED"])
        self.assertTrue(model["state_head"]["sole_state_bottleneck"])
        self.assertFalse(model["state_head"]["hidden_bypass_authorized"])
        self.assertEqual(model["transition"]["bev_dim"], 3)
        self.assertEqual(model["transition"]["action_dim"], 9)
        self.assertEqual(model["transition"]["warp_calls"], 0)
        inventory = contract.MODEL_PARAMETER_INVENTORY
        self.assertEqual(inventory["encoder"]["parameter_count"], 2_747_520)
        self.assertEqual(inventory["decoder_state"]["parameter_count"], 370_051)
        self.assertEqual(inventory["predictor"]["parameter_count"], 160_134)
        self.assertEqual(
            inventory["detached_target_encoder_decoder_state"]["parameter_count"],
            3_117_571,
        )
        self.assertEqual(inventory["total"], {"parameter_count": 6_395_276, "tensor_count": 208})
        for group in ("encoder", "decoder_state", "predictor", "detached_target_encoder_decoder_state"):
            self.assertTrue(contract.is_sha256(inventory[group]["ordered_parameter_name_sha256"]))

    def test_objective_optimizer_schedule_and_caps_are_frozen(self) -> None:
        objective = contract.objective_contract()
        self.assertEqual(objective["G"]["formula"], "mean(G_current,G_next)")
        self.assertEqual(objective["C"]["hold_candidate_count"], 10)
        self.assertEqual(objective["C"]["non_hold_candidate_count"], 11)
        self.assertEqual(objective["C"]["all_train_rows_included"], 4_262)
        self.assertEqual(objective["C"]["train_fallback_rows_included"], 25)
        self.assertEqual(objective["same_action_target_metrics"]["population"], 494)
        optimizer = contract.optimizer_contract()
        self.assertEqual(optimizer["name"], "AdamW")
        self.assertEqual(optimizer["precision"], "float32")
        self.assertEqual(optimizer["learning_rates"]["encoder"], 1e-4)
        schedule = contract.build_schedule_identity()
        self.assertEqual(schedule["seed"], 20260713)
        self.assertEqual(schedule["effective_batch_size"], 16)
        self.assertEqual(schedule["presentations"], 16_000)
        self.assertEqual(schedule["checkpoints"], [100, 400, 1_000])
        self.assertEqual(contract.MAXIMUM_ATTEMPTS, 1)
        self.assertEqual(contract.GPU_ACTIVE_TIME_CAP_MINUTES, 60)

    def test_exact_endpoint_populations_and_raster_reduction(self) -> None:
        definitions = contract.OBSERVATION_METRIC_CONTRACT
        aggregate = definitions["aggregate_raster"]
        rough = definitions["rough_raster"]
        reduction = definitions["raster_reduction"]
        self.assertEqual(aggregate["population"], 924)
        self.assertEqual(
            aggregate["ordered_endpoint_identity_sha256"],
            "dd84fc73e14056c9d6c8f7c066c2dcafe9726827193c42982d51f412ea744fa4",
        )
        self.assertEqual(rough, {
            "population": 123,
            "family": "rough_local_dynamics",
            "construction": "same_endpoint_protocol_restricted_to_frozen_family",
        })
        self.assertEqual(reduction["confusion_shape"], [3, 3])
        self.assertEqual(reduction["confusion_orientation"], "target_rows_predicted_columns")
        self.assertEqual(reduction["tie_break"], "lowest_class_index")
        populations = contract.science_contract()["gates"]["metric_populations"]
        self.assertEqual(populations["aggregate_raster_unique_endpoints"], 924)
        self.assertEqual(populations["rough_raster_unique_endpoints"], 123)

    def test_raster_label_grant_and_honest_access_counters(self) -> None:
        runtime = contract.runtime_authorization_template()
        grant = runtime["raw"]["grant"]
        self.assertEqual(grant["allowed_supervision_arrays"], ["raster_labels.u1"])
        self.assertFalse(grant["other_supervision_array_access_authorized"])
        self.assertFalse(grant["general_raw_frame_loader_authorized"])
        counters = {field: 0 for field in contract.ACCESS_COUNTER_FIELDS}
        counters["current_rgb_row_request_count"] = 16_000
        counters["next_rgb_row_request_count"] = 16_000
        counters["endpoint_rgb_row_request_count"] = 924
        counters["rgb_cache_hit_count"] = 32_914
        counters["rgb_cache_miss_count"] = 10
        counters["rgb_physical_file_open_count"] = 10
        counters["raster_label_row_request_count"] = 32_924
        counters["raster_label_row_cache_hit_count"] = 32_914
        counters["raster_label_row_cache_miss_count"] = 10
        counters["raster_label_underlying_array_cache_hit_count"] = 9
        counters["raster_label_underlying_array_cache_miss_count"] = 1
        counters["raster_label_physical_array_open_count"] = 1
        self.assertEqual(
            contract.validate_access_counters(counters)[
                "endpoint_rgb_row_request_count"
            ],
            924,
        )
        mapping = runtime["access_counter_fields"]["runner_counter_mapping"]
        self.assertEqual(
            mapping["endpoint_rgb_row_request_count"],
            "rgb_request_count.endpoint",
        )
        self.assertEqual(
            mapping["raster_label_row_cache_miss_count"],
            "sum(raster_row_cache_miss_count.*)",
        )
        self.assertEqual(
            mapping["raster_label_underlying_array_cache_miss_count"],
            "raster_underlying_array_cache_miss_count",
        )
        bad = dict(counters)
        bad["other_supervision_array_open_count"] = 1
        with self.assertRaises(PermissionError):
            contract.validate_access_counters(bad)
        bad = dict(counters)
        bad["current_rgb_row_request_count"] = -1
        with self.assertRaises(PermissionError):
            contract.validate_access_counters(bad)
        bad = dict(counters)
        bad["raster_label_row_cache_hit_count"] -= 1
        with self.assertRaises(PermissionError):
            contract.validate_access_counters(bad)

    def test_update_zero_and_update_100_strict_gates(self) -> None:
        update0 = _update_zero_metrics()
        self.assertTrue(contract.evaluate_gate(0, update0)["passed"])
        failed = dict(update0)
        failed["all_nine_action_predictions_bitwise_equal"] = False
        self.assertFalse(contract.evaluate_gate(0, failed)["passed"])

        update100 = _update_100_metrics()
        self.assertTrue(
            contract.evaluate_gate(100, update100, update_zero=update0)["passed"]
        )
        failed = dict(update100)
        failed["G"] = 1.0
        self.assertFalse(
            contract.evaluate_gate(100, failed, update_zero=update0)["passed"]
        )
        failed = dict(update100)
        failed["action_nll"] = math.log(9.0)
        self.assertFalse(
            contract.evaluate_gate(100, failed, update_zero=update0)["passed"]
        )

    def test_update_400_non_strict_and_strict_boundaries(self) -> None:
        update0 = _update_zero_metrics()
        metrics = _update_400_metrics()
        self.assertTrue(
            contract.evaluate_gate(400, metrics, update_zero=update0)["passed"]
        )
        failed = dict(metrics)
        failed["G"] = math.nextafter(0.90, math.inf)
        self.assertFalse(
            contract.evaluate_gate(400, failed, update_zero=update0)["passed"]
        )
        failed = dict(metrics)
        failed["action_nll"] = 0.99 * math.log(9.0)
        self.assertFalse(
            contract.evaluate_gate(400, failed, update_zero=update0)["passed"]
        )
        failed = dict(metrics)
        failed["same_action_target_strict_win_rate"] = math.nextafter(0.60, -math.inf)
        self.assertFalse(
            contract.evaluate_gate(400, failed, update_zero=update0)["passed"]
        )

    def test_update_1000_gate_and_prior_stop_rule(self) -> None:
        metrics = _update_1000_metrics()
        self.assertTrue(contract.evaluate_gate(1_000, metrics)["passed"])
        failed = dict(metrics)
        failed["aggregate_raster_balanced_accuracy"] = contract.GATE_THRESHOLDS[1_000][
            "aggregate_raster_balanced_accuracy_strictly_greater_than"
        ]
        self.assertFalse(contract.evaluate_gate(1_000, failed)["passed"])
        self.assertFalse(
            contract.evaluate_gate(
                1_000, metrics, prior_gates_passed=False
            )["passed"]
        )

    def test_one_attempt_failure_chain_and_downstream_denials(self) -> None:
        control = contract.CONTROL_UPDATE_400_FAIL
        chain = {
            "metrics": control,
            "artifact": control,
            "result": control,
            "completion": control,
        }
        self.assertEqual(contract.validate_failure_status_chain(chain), chain)
        bad = dict(chain)
        bad["result"] = contract.CONTROL_PASS
        with self.assertRaises(ValueError):
            contract.validate_failure_status_chain(bad)
        self.assertTrue(contract.EXECUTION_AUTHORITY["one_exact_fresh_attempt_authorized"])
        self.assertFalse(contract.EXECUTION_AUTHORITY["phase_b_authorized"])
        for field in (
            "navigation_authorized",
            "heldout_authorized",
            "sealed_authorized",
            "retry_resume_repair_recovery_replacement_or_second_seed_authorized",
        ):
            self.assertFalse(contract.EXECUTION_AUTHORITY[field])

    def test_exact_recursive_v11_source_reuse_is_forced(self) -> None:
        self.assertEqual(len(contract.REUSED_SOURCE_PATHS), 55)
        self.assertEqual(len(contract.ADDITIVE_SOURCE_PATHS), 9)
        self.assertEqual(len(contract.SOURCE_PATHS), 64)
        self.assertEqual(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES, contract.SOURCE_PATHS)
        self.assertIn(contract.FROZEN_V11_CONTRACT_RELATIVE_PATH, contract.SOURCE_PATHS)
        self.assertIn(contract.FROZEN_V11_RUNNER_RELATIVE_PATH, contract.SOURCE_PATHS)
        self.assertIn(contract.FROZEN_V11_LAUNCHER_RELATIVE_PATH, contract.SOURCE_PATHS)

    def test_canonical_helpers_require_one_line_and_reject_duplicates(self) -> None:
        value = contract.with_content_sha256({"schema": "fixture", "status": "PASS"})
        raw = contract.canonical_json_bytes(value) + b"\n"
        self.assertEqual(contract.parse_canonical_json(raw, name="fixture"), value)
        with self.assertRaises(ValueError):
            contract.parse_canonical_json(raw[:-1], name="fixture")
        with self.assertRaises(ValueError):
            contract.parse_canonical_json(
                b'{"content_sha256":"' + b"0" * 64 + b'","x":1,"x":2}\n',
                name="duplicate fixture",
            )

    def test_review_and_authorization_enforce_identity_separation(self) -> None:
        manifest_binding = {
            "path": contract.SOURCE_MANIFEST_RELATIVE_PATH,
            "file_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "byte_count": 100,
        }
        sources = {
            contract.SOURCE_MANIFEST_RELATIVE_PATH: "a" * 64,
            contract.PREREGISTRATION_RELATIVE_PATH: contract.PREREGISTRATION_FILE_SHA256,
            contract.PREREGISTRATION_INDEPENDENT_REVIEW_RELATIVE_PATH: (
                contract.PREREGISTRATION_INDEPENDENT_REVIEW_FILE_SHA256
            ),
            contract.ARCHITECTURE_DECISION_RELATIVE_PATH: (
                contract.ARCHITECTURE_DECISION_FILE_SHA256
            ),
            contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH: (
                contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
            ),
        }
        review_core = {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS_SOURCE_AND_SCIENCE",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_source_reviewer",
            "reviewed_sources": sources,
            "source_manifest": manifest_binding,
            "preregistration": contract.preregistration_binding(),
            "preregistration_independent_review": (
                contract.preregistration_independent_review_binding()
            ),
            "architecture_decision": contract.architecture_decision_binding(),
            "prior_terminal_audit": contract.prior_terminal_audit_binding(),
            "science_contract": contract.science_contract(),
            "source_only_checks": {
                "stdlib_only_contract_import": True,
                "generated_inputs_opened": [],
                "checkpoints_or_tensors_opened": [],
                "runtime_outputs_or_traces_opened": [],
                "sealed_or_heldout_opened": [],
            },
            "scientific_checks": contract.SCIENTIFIC_REVIEW_CHECKS,
            "findings": [],
            "authority": contract.REVIEW_AUTHORITY,
        }
        review = contract.with_content_sha256(review_core)
        self.assertEqual(
            contract.validate_review(
                review,
                expected_sources=sources,
                source_manifest_binding=manifest_binding,
            ),
            review,
        )
        bad_review = deepcopy(review_core)
        bad_review["reviewer"] = contract.IMPLEMENTATION_AUTHOR
        with self.assertRaises(PermissionError):
            contract.validate_review(
                contract.with_content_sha256(bad_review),
                expected_sources=sources,
                source_manifest_binding=manifest_binding,
            )

        review_binding = {
            "path": contract.REVIEW_RELATIVE_PATH,
            "file_sha256": "c" * 64,
            "content_sha256": review["content_sha256"],
            "byte_count": 1_000,
        }
        authorization_core = {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": contract.AUTHORIZATION_STATUS,
            "authorizer": "/root/separate_authorizer",
            "independent_source_review": review_binding,
            "preregistration": contract.preregistration_binding(),
            "architecture_decision": contract.architecture_decision_binding(),
            "prior_terminal_audit": contract.prior_terminal_audit_binding(),
            "runtime_inputs": contract.runtime_authorization_template(),
            "experiment": contract.science_contract(),
            "authority": contract.EXECUTION_AUTHORITY,
        }
        authorization = contract.with_content_sha256(authorization_core)
        self.assertEqual(
            contract.validate_authorization(
                authorization,
                review_binding=review_binding,
                reviewer="/root/independent_source_reviewer",
            ),
            authorization,
        )
        bad_authorization = deepcopy(authorization_core)
        bad_authorization["authorizer"] = "/root/independent_source_reviewer"
        with self.assertRaises(PermissionError):
            contract.validate_authorization(
                contract.with_content_sha256(bad_authorization),
                review_binding=review_binding,
                reviewer="/root/independent_source_reviewer",
            )


if __name__ == "__main__":
    unittest.main()
