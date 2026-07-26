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


_ACTION_ROW_COUNTS = {
    action: (60 if action == "hold" else (55 if index < 3 else 54))
    for index, action in enumerate(contract.ACTION_VOCABULARY)
}
_FAMILY_ROW_COUNTS = tuple(
    contract.SELECTION_FAMILY_BINDINGS[family]["row_count"]
    for family in contract.SCENE_FAMILIES
)
_FAMILY_SAME_ACTION_COUNTS = tuple(
    contract.SELECTION_FAMILY_BINDINGS[family]["same_action_row_count"]
    for family in contract.SCENE_FAMILIES
)
_FAMILY_NON_HOLD_COUNTS = tuple(
    contract.SELECTION_FAMILY_BINDINGS[family]["non_hold_row_count"]
    for family in contract.SCENE_FAMILIES
)


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


def _retrieval(
    *,
    action_nll: float,
    target_nll: float,
    same_action_target_nll: float,
    two_target_nll: float,
    correct_per_action: int | list[int],
    energy_ratio: float,
    strict_win_count: int,
    positive_family_count: int,
) -> dict[str, Any]:
    if type(correct_per_action) is int:
        correct_counts = [correct_per_action] * len(contract.ACTION_VOCABULARY)
    else:
        correct_counts = list(correct_per_action)
    per_action: dict[str, dict[str, int | float]] = {}
    recalls: list[float] = []
    total_correct = 0
    for action, correct in zip(contract.ACTION_VOCABULARY, correct_counts):
        count = _ACTION_ROW_COUNTS[action]
        recall = correct / count
        per_action[action] = {
            "row_count": count,
            "mean_nll": action_nll,
            "recall": recall,
        }
        recalls.append(recall)
        total_correct += correct

    families: dict[str, dict[str, int | float | bool]] = {}
    for index, family in enumerate(contract.SCENE_FAMILIES):
        margin = 0.2 if index < positive_family_count else 0.0
        families[family] = {
            "scene_id":
                contract.SELECTION_FAMILY_BINDINGS[family]["scene_id"],
            "row_count": _FAMILY_ROW_COUNTS[index],
            "same_action_row_count": _FAMILY_SAME_ACTION_COUNTS[index],
            "non_hold_row_count": _FAMILY_NON_HOLD_COUNTS[index],
            "deranged_minus_correct_energy": margin,
            "current_target_minus_correct_energy": margin,
            "cyclic_wrong_minus_executed_energy": margin,
            "hardest_wrong_minus_executed_energy": margin,
            "hold_minus_non_hold_executed_energy": margin,
            "permuted_minus_executed_energy": margin,
            "hold_action_rows_match_non_hold_rows": True,
        }

    return {
        "all_values_finite": True,
        "energy_values_within_closed_zero_four": True,
        "target_candidate_order_and_counts_exact": True,
        "same_action_target_mapping_exact": True,
        "selection_action_permutation_exact": True,
        "reference_values_immutable": True,
        "action_equal_logit_reference": math.log(9.0),
        "two_target_equal_logit_reference": math.log(2.0),
        "action_retrieval_nll": action_nll,
        "action_retrieval_top1_accuracy": total_correct / 495,
        "per_executed_action_action_retrieval": per_action,
        "action_retrieval_macro_balanced_accuracy": sum(recalls) / 9,
        "target_retrieval_nll": target_nll,
        "same_action_target_retrieval_nll": same_action_target_nll,
        "hold_target_retrieval_nll": target_nll,
        "non_hold_target_retrieval_nll": target_nll,
        "same_action_two_target_nll": two_target_nll,
        "target_retrieval_top1_count": 300,
        "target_retrieval_top1_accuracy": 300 / 495,
        "same_action_strict_win_count": strict_win_count,
        "same_action_strict_win_rate": strict_win_count / 494,
        "same_action_correct_energy": energy_ratio,
        "same_action_deranged_energy": 1.0,
        "same_action_correct_to_deranged_ratio": energy_ratio,
        "non_hold_correct_energy": energy_ratio,
        "non_hold_current_target_energy": 1.0,
        "non_hold_correct_to_current_ratio": energy_ratio,
        "executed_action_energy": energy_ratio,
        "cyclic_wrong_action_energy": 1.0,
        "hardest_wrong_action_energy": 1.0,
        "permuted_action_energy": 1.0,
        "non_hold_executed_action_energy": energy_ratio,
        "non_hold_hold_action_energy": 1.0,
        "executed_to_cyclic_ratio": energy_ratio,
        "executed_to_hardest_wrong_ratio": energy_ratio,
        "executed_to_permuted_ratio": energy_ratio,
        "non_hold_executed_to_hold_ratio": energy_ratio,
        "all_row_count": 495,
        "same_action_row_count": 494,
        "fallback_row_count": 1,
        "hold_row_count": 60,
        "non_hold_row_count": 435,
        "target_candidate_count": 1_425,
        "action_candidate_count": 9,
        "all_wrong_action_candidate_count": 3_960,
        "selection_target_mapping_sha256":
            contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
            ["mapping_sha256"],
        "selection_action_permutation_sha256":
            contract.SELECTION_ACTION_PERMUTATION_BINDING["mapping_sha256"],
        "per_family": families,
        "deranged_positive_family_margin_count": positive_family_count,
        "current_target_positive_family_margin_count": positive_family_count,
        "cyclic_positive_family_margin_count": positive_family_count,
        "hold_positive_family_margin_count": positive_family_count,
        "permuted_positive_family_margin_count": positive_family_count,
    }


def _metrics(
    *,
    action_nll: float,
    target_nll: float,
    same_action_target_nll: float,
    two_target_nll: float,
    raw_rank: float,
    projected_rank: float,
    correct_per_action: int | list[int],
    energy_ratio: float = 0.8,
    strict_win_count: int = 300,
    positive_family_count: int = 6,
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
        "non_hold_pair_count": 435,
        "shuffled_current_mse": 1.0,
        "latent_flow": _latent_flow(),
        "factorized_retrieval": _retrieval(
            action_nll=action_nll,
            target_nll=target_nll,
            same_action_target_nll=same_action_target_nll,
            two_target_nll=two_target_nll,
            correct_per_action=correct_per_action,
            energy_ratio=energy_ratio,
            strict_win_count=strict_win_count,
            positive_family_count=positive_family_count,
        ),
    }


def _update0_metrics() -> dict[str, Any]:
    return {
        "raw_cross_sample_variance": 4.0,
        "content_residual_spatial_diversity": 8.0,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
        "latent_flow": _latent_flow(update_zero=True),
        "factorized_retrieval": _retrieval(
            action_nll=math.log(9.0),
            target_nll=contract.SELECTION_EQUAL_LOGIT_REFERENCE_BINARY64,
            same_action_target_nll=
                contract.SELECTION_SAME_ACTION_EQUAL_LOGIT_REFERENCE_BINARY64,
            two_target_nll=math.log(2.0),
            correct_per_action=[55, 0, 0, 0, 0, 0, 0, 0, 0],
            energy_ratio=1.0,
            strict_win_count=0,
            positive_family_count=0,
        ),
    }


def _update100_metrics() -> dict[str, Any]:
    return _metrics(
        action_nll=2.1,
        target_nll=0.95,
        same_action_target_nll=0.94,
        two_target_nll=0.65,
        raw_rank=30.0,
        projected_rank=20.0,
        correct_per_action=12,
    )


def _update400_metrics() -> dict[str, Any]:
    return _metrics(
        action_nll=2.0,
        target_nll=0.85,
        same_action_target_nll=0.84,
        two_target_nll=0.60,
        raw_rank=38.0,
        projected_rank=33.0,
        correct_per_action=14,
    )


def _terminal_metrics(*, energy_ratio: float = 0.8) -> dict[str, Any]:
    return _metrics(
        action_nll=1.9,
        target_nll=0.75,
        same_action_target_nll=0.74,
        two_target_nll=0.55,
        raw_rank=48.0,
        projected_rank=48.0,
        correct_per_action=15,
        energy_ratio=energy_ratio,
    )


def _set_action_nll(metrics: dict[str, Any], value: float) -> None:
    retrieval = metrics["factorized_retrieval"]
    retrieval["action_retrieval_nll"] = value
    for row in retrieval["per_executed_action_action_retrieval"].values():
        row["mean_nll"] = value


def _set_action_correct_counts(
    metrics: dict[str, Any],
    correct_counts: list[int],
) -> None:
    retrieval = metrics["factorized_retrieval"]
    recalls: list[float] = []
    total = 0
    for action, correct in zip(contract.ACTION_VOCABULARY, correct_counts):
        row = retrieval["per_executed_action_action_retrieval"][action]
        row["recall"] = correct / row["row_count"]
        recalls.append(row["recall"])
        total += correct
    retrieval["action_retrieval_top1_accuracy"] = total / 495
    retrieval["action_retrieval_macro_balanced_accuracy"] = sum(recalls) / 9


def _selection_metadata_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_row = 4_262
    for scene_index, scene_size in enumerate(_FAMILY_ROW_COUNTS):
        if scene_size == 47:
            remaining = [
                action for action in contract.ACTION_VOCABULARY
                if action != "forward_slow"
            ]
            counts = dict(zip(remaining, (13, 5, 5, 5, 5, 5, 4, 4)))
            counts["forward_slow"] = 1
        else:
            counts = {
                action: (8 if index == 0 else 7)
                for index, action in enumerate(contract.ACTION_VOCABULARY)
            }
        local_row = 0
        for action in contract.ACTION_VOCABULARY:
            for _ in range(counts[action]):
                identity = f"scene-{scene_index}-row-{local_row}"
                rows.append({
                    "dataset_role": "checkpoint_selection",
                    "global_row": global_row,
                    "scene_id": f"scene-{scene_index}",
                    "primitive": action,
                    "content_sha256":
                        hashlib.sha256(identity.encode("ascii")).hexdigest(),
                    "next_endpoint_sha256": hashlib.sha256(
                        f"next-{identity}".encode("ascii")
                    ).hexdigest(),
                })
                global_row += 1
                local_row += 1
    return rows


def _integrity() -> dict[str, Any]:
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


class ActionConditionedRetrievalV10ContractTests(unittest.TestCase):
    def test_import_identity_and_frozen_evidence_are_exact(self) -> None:
        imported_roots = {
            name.partition(".")[0]
            for name in _MODULES_IMPORTED_BY_CONTRACT
        }
        self.assertTrue(imported_roots.isdisjoint({
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }))
        self.assertEqual(
            contract.SCHEMA_PREFIX,
            "lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10",
        )
        self.assertEqual(
            contract.PREREGISTRATION_COMMIT,
            "25b93c92fbfb2816d52f0dfc27603c759e7c3c68",
        )
        raw = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
        self.assertEqual(len(raw), 23_010)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "b199d396a9aa9196fa3acc50837f492edef9d4255634e44f399f470a7489785b",
        )
        self.assertEqual(contract.prior_terminal_audit_binding(), {
            "path":
                "docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_"
                "jepa_v9_terminal_audit_2026-07-25.json",
            "commit": "f02fdb02db328b339df5ec897424a42fe45a258b",
            "file_sha256":
                "a95b81c30e619c0fe5ef06c46e7cc60270ef27751c4588291482bdf9d0319ad8",
            "content_sha256":
                "82038aecd65d1d9b844903c768c7a0cee0750f981f4d824c05731fff95970120",
            "byte_count": 18_408,
        })

    def test_objective_has_exact_parameter_free_v10_retrieval_boundary(self) -> None:
        science = contract.science_contract()
        self.assertEqual(
            science["phase_a"]["reviewed_forward_base_commit"],
            "c93124b15387acf1fd440d281e9c4503a9e8355a",
        )
        initialization = science["initialization"]
        self.assertEqual(initialization["v10_new_parameter_count"], 0)
        self.assertEqual(initialization["v10_new_initialization_draw_count"], 0)
        objective = science["phase_a"]["objective"]
        retrieval = objective["factorized_retrieval"]
        self.assertEqual(retrieval["energy"], "mean_token(sum_feature((q-t)**2))")
        self.assertEqual(retrieval["energy_closed_bound"], [0.0, 4.0])
        self.assertEqual(retrieval["action_retrieval_loss_weight"], 1.0)
        self.assertEqual(retrieval["target_retrieval_loss_weight"], 1.0)
        self.assertEqual(retrieval["new_trainable_parameter_count"], 0)
        self.assertEqual(
            objective["total_loss"],
            "L_JEPA+L_action_retrieval+L_target_retrieval+"
            "0.50*(V_raw+V_projected)+0.02*(K_raw+K_projected)",
        )
        self.assertNotIn("dense_pairwise_inverse", contract.PHASE_A_METRIC_FIELDS)
        self.assertNotIn(
            "dense_pairwise_inverse_head.",
            contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES,
        )
        self.assertNotIn(
            "dense_pairwise_inverse_head",
            json.dumps(objective, sort_keys=True),
        )
        phase_b = science["phase_b"]
        self.assertEqual(
            phase_b["copied_state"],
            "phase_a_terminal_in_memory_online_encoder_only",
        )
        self.assertFalse(phase_b["factorized_retrieval_state_copied"])
        self.assertFalse(phase_b["latent_flow_predictor_or_projector_copied"])
        self.assertFalse(phase_b["optimizer_copied"])

    def test_mapping_bindings_and_population_constants_are_exact(self) -> None:
        self.assertEqual(contract.PAIR_INDEX_FILE_SHA256, (
            "5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d"
        ))
        self.assertEqual(contract.PAIR_INDEX_BYTE_COUNT, 6_207_286)
        self.assertEqual(
            contract.TARGET_MAPPING_BINDINGS["train"]["mapping_sha256"],
            "c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b",
        )
        self.assertEqual(
            contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
            ["mapping_sha256"],
            "95d42273a8319316ad68781cb2158146e7672eda529984c3aeddc0937d87a9c1",
        )
        self.assertEqual(
            contract.SELECTION_ACTION_PERMUTATION_BINDING["mapping_sha256"],
            "2740be362829c172a06aebae0d077e69ede8af80cbf6f00569eb460dc559bb0f",
        )
        self.assertEqual(
            contract.SELECTION_FAMILY_BINDINGS_SHA256,
            "c39efe48afd6d4c02a24af77f1f11e7f6cd5a69d571b0a9416924b07bbacbb11",
        )
        self.assertEqual(
            contract.canonical_json_sha256(contract.SELECTION_FAMILY_BINDINGS),
            contract.SELECTION_FAMILY_BINDINGS_SHA256,
        )
        self.assertEqual(
            contract.SELECTION_FAMILY_BINDINGS["small_enclosed_maze"],
            {
                "scene_id": "small_enclosed_maze_16b0fc2c449b",
                "row_count": 47,
                "same_action_row_count": 46,
                "non_hold_row_count": 37,
            },
        )
        self.assertEqual(contract.TRAIN_SAME_ACTION_ELIGIBLE_COUNT, 4_237)
        self.assertEqual(contract.TRAIN_SAME_ACTION_FALLBACK_COUNT, 25)
        self.assertEqual(contract.SELECTION_SAME_ACTION_PAIR_COUNT, 494)
        self.assertEqual(contract.SELECTION_FALLBACK_PAIR_COUNT, 1)
        self.assertEqual(contract.SELECTION_TARGET_CANDIDATE_COUNT, 1_425)
        self.assertEqual(
            contract.SELECTION_EQUAL_LOGIT_REFERENCE_BINARY64,
            1.049465002836817,
        )
        self.assertEqual(
            contract.SELECTION_SAME_ACTION_EQUAL_LOGIT_REFERENCE_BINARY64,
            1.0493655144039604,
        )

    def test_deterministic_target_and_action_mapping_semantics(self) -> None:
        rows = _selection_metadata_rows()
        target = contract.build_same_action_target_mapping(
            rows, role="checkpoint_selection"
        )
        self.assertEqual(target["binding"]["row_count"], 495)
        self.assertEqual(target["binding"]["same_action_eligible_count"], 494)
        self.assertEqual(target["binding"]["fallback_count"], 1)
        self.assertEqual(
            target["binding"]["non_singleton_primitive_group_count"], 71
        )
        self.assertEqual(target["binding"]["primitive_group_count"], 72)
        for index, negative_index in enumerate(target["negative_indices"]):
            self.assertEqual(
                rows[index]["scene_id"], rows[negative_index]["scene_id"]
            )
            if target["same_action_eligible"][index]:
                self.assertEqual(
                    rows[index]["primitive"], rows[negative_index]["primitive"]
                )
            else:
                self.assertNotEqual(
                    rows[index]["primitive"], rows[negative_index]["primitive"]
                )
        action = contract.build_selection_action_permutation(rows)
        self.assertEqual(action["binding"]["changed_action_count"], 495)
        self.assertEqual(
            action["binding"]["scene_size_histogram"], {"47": 1, "64": 7}
        )
        self.assertEqual(
            action["binding"]["shift_histogram"], {"8": 7, "13": 1}
        )
        self.assertCountEqual(action["control_actions"], [
            row["primitive"] for row in rows
        ])
        with self.assertRaises(PermissionError):
            contract.validate_same_action_target_mapping(
                rows, role="checkpoint_selection"
            )
        with self.assertRaises(PermissionError):
            contract.validate_selection_action_permutation(rows)

    def test_schedule_caps_roles_denials_and_phase_b_thresholds(self) -> None:
        self.assertEqual(contract.CHECKPOINT_UPDATES, (100, 400, 1_000))
        self.assertEqual(contract.PHASE_A_MAXIMUM_PRESENTATIONS, 16_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_PRESENTATIONS, 32_000)
        self.assertEqual(contract.CUMULATIVE_MAXIMUM_UPDATE, 2_000)
        self.assertEqual(contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        self.assertEqual(contract.PHASE_B_GPU_ACTIVE_TIME_CAP_MINUTES, 60)
        for phase in ("phase_a", "phase_b"):
            identity = contract.build_schedule_identity(phase)
            self.assertEqual(
                contract.validate_schedule_identity(identity, phase=phase),
                identity,
            )
        runtime = contract.runtime_authorization_template()
        self.assertEqual(contract.validate_runtime_inputs(runtime), runtime)
        grant = runtime["raw"]["phase_a_grant"]
        self.assertEqual(
            grant["target_mapping_bindings"],
            contract.TARGET_MAPPING_BINDINGS,
        )
        self.assertEqual(
            grant["selection_action_permutation_binding"],
            contract.SELECTION_ACTION_PERMUTATION_BINDING,
        )
        self.assertTrue(all(
            value is False for value in contract.DOWNSTREAM_DENIALS.values()
        ))
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

    def test_staged_update100_update400_and_terminal_pass(self) -> None:
        update0 = _update0_metrics()
        update100 = _update100_metrics()
        first = contract.evaluate_phase_a_continuation(
            100, update100, update0, _integrity()
        )
        self.assertTrue(first["passed"])
        self.assertEqual(first["control"], contract.CONTROL_CONTINUE)
        update400 = _update400_metrics()
        second = contract.evaluate_phase_a_continuation(
            400, update400, update0, _integrity(), update100
        )
        self.assertTrue(second["passed"])
        terminal = contract.evaluate_phase_a(
            _terminal_metrics(), update0, _integrity(), update400
        )
        self.assertTrue(terminal["passed"])
        self.assertEqual(terminal["control"], contract.CONTROL_PHASE_A_PASS)
        self.assertIn("factorized_retrieval", terminal)
        self.assertNotIn("dense_pairwise_inverse", terminal)

    def test_update100_strict_boundaries_fail(self) -> None:
        update0 = _update0_metrics()
        cases: list[tuple[str, dict[str, Any]]] = []

        action_reference = _update100_metrics()
        _set_action_nll(
            action_reference,
            action_reference["factorized_retrieval"]
            ["action_equal_logit_reference"],
        )
        cases.append((
            "action_retrieval_nll_strictly_below_equal_logit_reference",
            action_reference,
        ))

        macro = _update100_metrics()
        _set_action_correct_counts(macro, [55, 0, 0, 0, 0, 0, 0, 0, 0])
        cases.append((
            "action_macro_balanced_accuracy_strictly_above_one_ninth",
            macro,
        ))

        two_target = _update100_metrics()
        two_target["factorized_retrieval"]["same_action_two_target_nll"] = (
            math.log(2.0)
        )
        cases.append((
            "same_action_two_target_nll_strictly_below_reference",
            two_target,
        ))

        strict_win = _update100_metrics()
        strict_win["factorized_retrieval"]["same_action_strict_win_count"] = 247
        strict_win["factorized_retrieval"]["same_action_strict_win_rate"] = 0.5
        cases.append((
            "same_action_strict_win_rate_strictly_above_half", strict_win
        ))

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

    def test_update400_requires_progress_and_strict_mechanism_ratio(self) -> None:
        update0 = _update0_metrics()
        update100 = _update100_metrics()
        equal_action_nll = _update400_metrics()
        _set_action_nll(
            equal_action_nll,
            update100["factorized_retrieval"]["action_retrieval_nll"],
        )
        result = contract.evaluate_phase_a_continuation(
            400, equal_action_nll, update0, _integrity(), update100
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]
            ["action_retrieval_nll_strictly_lower_than_update100"]
        )

        ratio_boundary = _update400_metrics()
        retrieval = ratio_boundary["factorized_retrieval"]
        retrieval["same_action_correct_energy"] = 0.99
        retrieval["same_action_correct_to_deranged_ratio"] = 0.99
        result = contract.evaluate_phase_a_continuation(
            400, ratio_boundary, update0, _integrity(), update100
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]
            ["same_action_correct_to_deranged_strictly_below_point99"]
        )
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                400, _update400_metrics(), update0, _integrity()
            )

    def test_terminal_progress_is_strict_but_point95_is_inclusive(self) -> None:
        update0 = _update0_metrics()
        update400 = _update400_metrics()
        no_target_progress = _terminal_metrics()
        retrieval = no_target_progress["factorized_retrieval"]
        retrieval["target_retrieval_nll"] = 0.85
        retrieval["hold_target_retrieval_nll"] = 0.85
        retrieval["non_hold_target_retrieval_nll"] = 0.85
        result = contract.evaluate_phase_a(
            no_target_progress, update0, _integrity(), update400
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]
            ["target_retrieval_nll_strictly_lower_than_update400"]
        )
        boundary = contract.evaluate_phase_a(
            _terminal_metrics(energy_ratio=0.95),
            update0,
            _integrity(),
            update400,
        )
        self.assertTrue(boundary["passed"])

    def test_factorized_receipts_fail_closed(self) -> None:
        base = _update100_metrics()
        malformed = deepcopy(base)
        malformed["factorized_retrieval"]["target_candidate_count"] = 1_424
        with self.assertRaises(ValueError):
            contract.evaluate_phase_a_continuation(
                100, malformed, _update0_metrics(), _integrity()
            )
        malformed = deepcopy(base)
        malformed["factorized_retrieval"][
            "selection_target_mapping_sha256"
        ] = "0" * 64
        with self.assertRaises(PermissionError):
            contract.evaluate_phase_a_continuation(
                100, malformed, _update0_metrics(), _integrity()
            )
        unhealthy = deepcopy(base)
        unhealthy["factorized_retrieval"][
            "energy_values_within_closed_zero_four"
        ] = False
        result = contract.evaluate_phase_a_continuation(
            100, unhealthy, _update0_metrics(), _integrity()
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["conjuncts"]["factorized_retrieval_health_exact"]
        )

    def test_update0_action_symmetry_binds_chance_and_all_controls(self) -> None:
        base = _update100_metrics()
        cases: list[dict[str, Any]] = []

        changed_nll = _update0_metrics()
        retrieval = changed_nll["factorized_retrieval"]
        retrieval["action_retrieval_nll"] = 0.0
        for row in retrieval["per_executed_action_action_retrieval"].values():
            row["mean_nll"] = 0.0
        cases.append(changed_nll)

        changed_macro = _update0_metrics()
        _set_action_correct_counts(
            {"factorized_retrieval": changed_macro["factorized_retrieval"]},
            [1] * len(contract.ACTION_VOCABULARY),
        )
        cases.append(changed_macro)

        changed_ratios = _update0_metrics()
        retrieval = changed_ratios["factorized_retrieval"]
        retrieval["executed_action_energy"] = 0.5
        retrieval["non_hold_executed_action_energy"] = 0.5
        for field in (
            "executed_to_cyclic_ratio",
            "executed_to_hardest_wrong_ratio",
            "executed_to_permuted_ratio",
            "non_hold_executed_to_hold_ratio",
        ):
            retrieval[field] = 0.5
        cases.append(changed_ratios)

        for index, update0 in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaisesRegex(
                    ValueError, "update-zero action symmetry"
                ):
                    contract.evaluate_phase_a_continuation(
                        100, base, update0, _integrity()
                    )

    def test_family_scene_identity_and_exact_populations_fail_closed(self) -> None:
        base = _update100_metrics()
        per_family = base["factorized_retrieval"]["per_family"]
        first, second = contract.SCENE_FAMILIES[:2]
        per_family[first]["scene_id"], per_family[second]["scene_id"] = (
            per_family[second]["scene_id"],
            per_family[first]["scene_id"],
        )
        with self.assertRaisesRegex(PermissionError, "scene identity changed"):
            contract.evaluate_phase_a_continuation(
                100, base, _update0_metrics(), _integrity()
            )

        changed_count = _update100_metrics()
        changed_count["factorized_retrieval"]["per_family"][first][
            "row_count"
        ] -= 1
        with self.assertRaisesRegex(ValueError, "row_count changed"):
            contract.evaluate_phase_a_continuation(
                100, changed_count, _update0_metrics(), _integrity()
            )

    def test_receipt_status_and_forbidden_access_contracts(self) -> None:
        counters = {field: 0 for field in contract.ACCESS_ZERO_COUNTER_FIELDS}
        self.assertEqual(contract.validate_access_zero_counters(counters), counters)
        changed = dict(counters)
        changed["heldout_open_count"] = 1
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
                contract.validate_phase_a_failure_status_chain(chain), chain
            )

    def test_source_manifest_rejects_heldout_path_parts(self) -> None:
        for forbidden in (
            "lewm/heldout/synthetic.py",
            "lewm/heldout_future/synthetic.py",
        ):
            with self.subTest(path=forbidden):
                core = _source_manifest_core()
                bindings = [
                    *core["source_bindings"],
                    {
                        "path": forbidden,
                        "file_sha256": hashlib.sha256(
                            forbidden.encode("ascii")
                        ).hexdigest(),
                        "byte_count": len(forbidden),
                    },
                ]
                bindings.sort(key=lambda item: item["path"])
                core["source_bindings"] = bindings
                core["source_paths"] = [item["path"] for item in bindings]
                core["source_bindings_sha256"] = (
                    contract.canonical_json_sha256(bindings)
                )
                core["source_count"] = len(bindings)
                raw = contract.canonical_json_bytes(
                    contract.with_content_sha256(core)
                ) + b"\n"
                with self.assertRaises(PermissionError):
                    contract.validate_source_manifest(raw)

    def test_canonical_manifest_review_and_authorization_validators(self) -> None:
        manifest_raw = contract.canonical_json_bytes(
            contract.with_content_sha256(_source_manifest_core())
        ) + b"\n"
        manifest = contract.validate_source_manifest(manifest_raw)
        expected_sources = {
            binding["path"]: binding["file_sha256"]
            for binding in manifest["source_bindings"]
        }
        manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        expected_sources[contract.SOURCE_MANIFEST_RELATIVE_PATH] = (
            manifest_file_sha256
        )
        expected_sources[contract.PREREGISTRATION_RELATIVE_PATH] = (
            contract.PREREGISTRATION_FILE_SHA256
        )
        expected_sources[contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] = (
            contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
        )
        manifest_binding = contract.artifact_binding(
            contract.SOURCE_MANIFEST_RELATIVE_PATH,
            manifest_raw,
            content_sha256=manifest["content_sha256"],
        )
        review_core = {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS_SOURCE_AND_SCIENCE",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_v10_reviewer",
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
            contract.validate_review(
                review,
                expected_sources=expected_sources,
                source_manifest_raw=manifest_raw,
            ),
            review,
        )
        original_root = contract.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                contract.ROOT = Path(directory)
                manifest_path = (
                    contract.ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH
                )
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_bytes(manifest_raw)
                self.assertEqual(
                    contract.validate_review(
                        review,
                        expected_sources=expected_sources,
                    ),
                    review,
                )
        finally:
            contract.ROOT = original_root
        for field, changed_value in (
            ("content_sha256", "f" * 64),
            ("byte_count", len(manifest_raw) + 1),
        ):
            with self.subTest(manifest_binding_field=field):
                changed_core = deepcopy(review_core)
                changed_core["source_manifest"][field] = changed_value
                changed_review = contract.with_content_sha256(changed_core)
                with self.assertRaises(PermissionError):
                    contract.validate_review(
                        changed_review,
                        expected_sources=expected_sources,
                        source_manifest_raw=manifest_raw,
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
            "authorizer": "/root/independent_v10_authorizer",
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
                reviewer=review["reviewer"],
            ),
            authorization,
        )


if __name__ == "__main__":
    unittest.main()
