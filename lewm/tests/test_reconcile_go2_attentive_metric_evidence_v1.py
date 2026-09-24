from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "reconcile_go2_attentive_metric_evidence_v1",
    ROOT / "scripts/reconcile_go2_attentive_metric_evidence_v1.py")
assert SPEC is not None and SPEC.loader is not None
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def row(state: str, candidate: int, truth: float, score: float, *,
        family: str = "family_a", safety: float = 0.0,
        safety_prediction: float | None = None, completion: float = 0.0,
        completion_prediction: float | None = None) -> dict:
    return {
        "training_view_row_digest": f"row-{state}-{candidate}",
        "branch_identity_digest": f"branch-{state}-{candidate}",
        "state_id": state,
        "family": family,
        "stratum": "fixture",
        "candidate_index": candidate,
        "target": {
            "progress": truth,
            "safety": safety,
            "completion": completion,
            "utility": truth,
        },
        "prediction": {
            "progress": score,
            "safety": (safety_prediction if safety_prediction is not None
                       else safety),
            "completion": (completion_prediction
                           if completion_prediction is not None else completion),
            "utility": score,
        },
    }


def assert_consumers_agree(rows: list[dict]) -> tuple[dict, dict]:
    left = R.consumer_a(rows, project_component_targets_to_float32=False)
    right = R.consumer_b(rows, project_component_targets_to_float32=False)
    comparison = R.compare_consumers(left, right)
    assert comparison["passed"], comparison
    return left, right


def synthetic_closed_evidence(monkeypatch: pytest.MonkeyPatch) -> dict:
    rows = []
    for state_index in range(24):
        for candidate in range(12):
            progress = (0.1 if state_index == 0 and candidate == 0
                        else candidate / 16.0)
            safety = 0.25 if candidate % 2 else 0.0
            completion = float(candidate == 11)
            rows.append({
                "training_view_row_digest": f"row-{state_index:02d}-{candidate:02d}",
                "branch_identity_digest": f"branch-{state_index:02d}-{candidate:02d}",
                "state_id": f"state-{state_index:02d}",
                "family": f"family-{state_index // 3}",
                "stratum": f"stratum-{state_index % 3}",
                "candidate_index": candidate,
                "target": {"progress": progress, "safety": safety,
                           "completion": completion, "utility": progress},
                "prediction": {
                    "progress": float(R.np.float32(0.25 + candidate / 32.0)),
                    "safety": float(R.np.float32(0.75 if safety else 0.25)),
                    "completion": float(R.np.float32(
                        0.75 if completion else 0.25)),
                    "utility": float(R.np.float32(0.25 + candidate / 32.0)),
                },
            })
    order_digest = R.C.digest([row["training_view_row_digest"] for row in rows])
    row_set = R.C.digest(sorted(row["training_view_row_digest"] for row in rows))
    branch_set = R.C.digest(sorted(row["branch_identity_digest"] for row in rows))
    frozen = dict(R.C.FROZEN_EVIDENCE)
    frozen.update({
        "training_view_row_order_digest": order_digest,
        "training_view_row_identity_set_digest": row_set,
        "branch_identity_set_digest": branch_set,
    })
    monkeypatch.setattr(R.C, "FROZEN_EVIDENCE", frozen)
    expected = dict(R.C.EXPECTED_EVIDENCE_INVENTORY)
    delta = abs(0.1 - float(R.np.float32(0.1)))
    expected.update({
        "progress_targets_changed_by_online_float32_projection": 1,
        "safety_targets_changed_by_online_float32_projection": 0,
        "maximum_progress_target_projection_delta": delta,
        "maximum_safety_target_projection_delta": 0.0,
    })
    monkeypatch.setattr(R.C, "EXPECTED_EVIDENCE_INVENTORY", expected)
    source = copy.deepcopy(R.C.SOURCE_RECONSTRUCTION)
    source["first_divergent_row"] = {
        "row_index": 0, "training_view_row_digest": "row-00-00",
        "branch_identity_digest": "branch-00-00", "state_id": "state-00",
        "family": "family-0", "stratum": "stratum-0", "candidate_index": 0,
        "stored_target": 0.1,
        "direct_float32_projected_target": float(R.np.float32(0.1)),
        "prediction": 0.25,
        "replay_absolute_error": abs(0.25 - 0.1),
        "direct_absolute_error": abs(0.25 - float(R.np.float32(0.1))),
    }
    monkeypatch.setattr(R.C, "SOURCE_RECONSTRUCTION", source)
    payload = {
        "schema": frozen["schema"], "status": frozen["status"],
        "complete": True,
        "execution_bindings": R.C.FROZEN_EVIDENCE_EXECUTION_BINDINGS,
        "evaluation_authorisation_digest": frozen[
            "evaluation_authorisation_digest"],
        "final_checkpoint_sha256": frozen["final_checkpoint_sha256"],
        "final_state_digest": frozen["final_state_digest"],
        "row_count": 288, "training_view_row_order_digest": order_digest,
        "training_view_row_identity_set_digest": row_set,
        "branch_identity_set_digest": branch_set, "rows": rows,
        "calibration_evaluation_session_count": 1,
        "model_forward_batch_count": 72, "raw_latent_persisted": False,
        "predictor_material_accessed": False,
    }
    return R._signed(payload, "calibration_evidence_digest")


def test_perfect_and_reversed_hand_computed_rank_metrics() -> None:
    perfect = [row("perfect", index, float(index), float(index))
               for index in range(4)]
    metrics, _ = assert_consumers_agree(perfect)
    assert metrics["composite"]["pairwise_ordering_accuracy"] == 1.0
    assert metrics["composite"]["ranking_spearman"] == 1.0
    assert metrics["composite"]["absolute_rank_regret"] == 0.0
    assert metrics["composite"]["normalised_rank_regret"] == 0.0
    assert metrics["composite"]["top1_recovery"] == 1.0

    reversed_rows = [row("reversed", index, float(index), float(3 - index))
                     for index in range(4)]
    metrics, _ = assert_consumers_agree(reversed_rows)
    assert metrics["composite"]["pairwise_ordering_accuracy"] == 0.0
    assert metrics["composite"]["ranking_spearman"] == -1.0
    assert metrics["composite"]["absolute_rank_regret"] == 3.0
    assert metrics["composite"]["normalised_rank_regret"] == 1.0
    assert metrics["composite"]["realised_selected_utility"] == 0.0


def test_exact_tie_multiple_top_ties_and_nonzero_regret() -> None:
    rows = [row("tie", 0, 0.0, 1.0), row("tie", 1, 2.0, 1.0),
            row("tie", 2, 1.0, 0.0)]
    metrics, _ = assert_consumers_agree(rows)
    composite = metrics["composite"]
    assert composite["top_score_tie_rate"] == 1.0
    assert composite["all_pair_tie_rate"] == pytest.approx(1 / 3)
    assert composite["absolute_rank_regret"] == 2.0
    assert composite["normalised_rank_regret"] == 1.0

    three_way = [row("three", 0, 0.0, 1.0), row("three", 1, 1.0, 1.0),
                 row("three", 2, 2.0, 1.0), row("three", 3, 3.0, 0.0)]
    metrics, _ = assert_consumers_agree(three_way)
    assert metrics["composite"]["top_score_tie_rate"] == 1.0


def test_zero_oracle_spread_and_single_class_labels() -> None:
    rows = [row("flat", index, 1.0, float(index), safety=0.0,
                completion=0.0) for index in range(4)]
    metrics, _ = assert_consumers_agree(rows)
    assert metrics["composite"]["normalised_rank_regret"] == 0.0
    assert metrics["safety"]["auc"] is None
    assert metrics["completion"]["auc"] is None


def test_unequal_families_and_partial_final_batch_are_not_reweighted() -> None:
    rows = [row("a", 0, 0.0, 0.0, family="small"),
            row("a", 1, 1.0, 1.0, family="small")]
    rows += [row("b", index, float(index), float(index), family="large")
             for index in range(3)]
    rows[-1]["prediction"]["progress"] += 1.0
    metrics, _ = assert_consumers_agree(rows)
    assert metrics["rows"] == 5  # also exercises a partial final batch
    assert metrics["progress"]["mae"] == 0.2  # row-weighted, no batch mean
    assert metrics["per_family"]["small"]["rows"] == 2
    assert metrics["per_family"]["large"]["rows"] == 3


def test_logits_and_probabilities_are_equivalent() -> None:
    probability = 0.8
    logit = math.log(probability / (1.0 - probability))
    probability_row = row("p", 0, 1.0, 1.0, safety=1.0,
                          safety_prediction=probability,
                          completion=1.0,
                          completion_prediction=probability)
    logit_row = copy.deepcopy(probability_row)
    del logit_row["prediction"]["safety"]
    del logit_row["prediction"]["completion"]
    logit_row["prediction"]["safety_logit"] = logit
    logit_row["prediction"]["completion_logit"] = logit
    for consumer in (R.consumer_a, R.consumer_b):
        first = consumer([probability_row],
                         project_component_targets_to_float32=False)
        second = consumer([logit_row],
                          project_component_targets_to_float32=False)
        assert R.compare_consumers(first, second)["passed"]
    inconsistent = copy.deepcopy(probability_row)
    inconsistent["prediction"]["safety_logit"] = 0.0
    with pytest.raises(R.MetricReconciliationError):
        R.consumer_a([inconsistent],
                     project_component_targets_to_float32=False)


def test_baseline_reordering_is_identity_exact() -> None:
    rows = [row("state", index, float(index), float(index))
            for index in range(4)]
    reordered = R.align_baseline_predictions(rows, list(reversed(rows)))
    assert [item["candidate_index"] for item in reordered] == [0, 1, 2, 3]
    changed = copy.deepcopy(rows)
    changed[-1]["training_view_row_digest"] = "wrong"
    with pytest.raises(R.MetricReconciliationError):
        R.align_baseline_predictions(rows, changed)


def test_repaired_result_helper_is_conditional_on_complete_evidence() -> None:
    rows = []
    manifest = {}
    for state_index in range(24):
        state_id = f"state-{state_index:02d}"
        family = f"family-{state_index // 3}"
        provenance = {
            "family": family, "stratum": f"stratum-{state_index % 3}",
            "scene_id": f"scene-{state_index:02d}",
            "state_identity_digest": f"state-digest-{state_index:02d}",
        }
        manifest[state_id] = provenance
        for candidate in range(12):
            item = row(state_id, candidate, float(candidate), float(candidate),
                       family=family, completion=float(candidate == 11))
            item.update({
                "action_blocks": [[0.0] * 10 for _ in range(4)],
                "goal_binding_input": [0.0] * 3,
                "split_role": "calibration", **provenance,
            })
            rows.append(item)
    baseline = copy.deepcopy(list(reversed(rows)))
    direct = R.consumer_a(rows, project_component_targets_to_float32=True)
    result = R.reconcile_complete_evidence(
        rows, baseline, direct, completion_fit_nondegenerate=True,
        family_assignment_manifest=manifest)
    assert result["classification"] \
        == "POST_EVALUATION_METRIC_CONSUMER_DEFECT_RECOVERABLE"
    assert result["result_label"] == "POST_EVALUATION_CONSUMER_REPAIR"
    incomplete = copy.deepcopy(rows)
    del incomplete[0]["action_blocks"]
    with pytest.raises(R.MetricReconciliationError):
        R.reconcile_complete_evidence(
            incomplete, baseline, direct, completion_fit_nondegenerate=True,
            family_assignment_manifest=manifest)


def test_float_tolerance_and_discrete_mismatch_are_distinguished() -> None:
    assert R.compare_consumers({"x": 1.0}, {"x": 1.0 + 5e-11})["passed"]
    mismatch = R.compare_consumers({"rows": 2}, {"rows": 3})
    assert mismatch["discrete_identities_and_counts_exact"] is False
    assert mismatch["passed"] is False


def test_recoverability_fails_only_from_retained_evidence_gaps() -> None:
    gates = R.recoverability(R.C.EXPECTED_EVIDENCE_INVENTORY)
    assert gates["complete_action_blocks_present"] is False
    assert gates["complete_goal_binding_present"] is False
    assert gates["row_aligned_no_latent_predictions_present"] is False
    assert gates[
        "complete_split_scene_state_and_family_manifest_provenance_present"] is False
    assert gates[
        "source_and_evidence_sufficient_to_reconstruct_direct_and_replay_paths"] is True
    assert gates["consumer_a_and_b_discrete_outputs_exact"] is None
    assert gates["all_original_gate_inputs_reconstructable"] is None


def test_incomplete_run_never_calls_consumers_and_is_one_shot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    predecessor = tmp_path / "predecessor"
    runtime.mkdir()
    predecessor.mkdir()
    (runtime / "contract.json").write_text("{}")
    (runtime / "contract.json").chmod(0o444)
    evidence = R._signed({"fixture": True}, "calibration_evidence_digest")
    (predecessor / "calibration_evidence.json").write_bytes(
        R.C.canonical_bytes(evidence) + b"\n")
    contract = {
        R.C.CONTRACT_SELF_KEY: "c" * 64,
        "source_closure": {
            "source_repository_commit": "s" * 40,
            R.C.SOURCE_SELF_KEY: "d" * 64,
        },
        "lineage": {"artifact_set_digest": "a" * 64, "artifacts": {}},
    }
    inspection = {
        "inventory": dict(R.C.EXPECTED_EVIDENCE_INVENTORY),
        "observed_families": {},
        "family_assignment_manifest_verdict":
            "UNVERIFIABLE_FROM_THE_SEVEN_AUTHORISED_ARTIFACTS",
        "prediction_representation": {},
        "source_reconstruction": dict(R.C.SOURCE_RECONSTRUCTION),
        "first_component_target_projection_difference": {},
    }
    monkeypatch.setattr(R.C, "runtime_root", lambda root: runtime)
    monkeypatch.setattr(R.C, "predecessor_root", lambda root: predecessor)
    monkeypatch.setattr(R, "load_contract", lambda root: contract)
    monkeypatch.setattr(R.C, "source_closure",
                        lambda root: contract["source_closure"])
    monkeypatch.setattr(R.C, "lineage_binding",
                        lambda root: contract["lineage"])
    monkeypatch.setattr(R, "inspect_evidence", lambda value: inspection)
    monkeypatch.setattr(R, "consumer_a", lambda *args, **kwargs:
                        pytest.fail("Consumer A must not run"))
    monkeypatch.setattr(R, "consumer_b", lambda *args, **kwargs:
                        pytest.fail("Consumer B must not run"))
    terminal = R.run_once(tmp_path)
    assert terminal["classification"] \
        == "INVALID_TECHNICAL_UNRECOVERABLE_METRIC_EVIDENCE"
    assert terminal["consumers_scientifically_executed"] is False
    assert not (runtime / "repaired_result.json").exists()
    assert {path.name for path in runtime.iterdir()} \
        == {"contract.json", "attempt.json", "terminal.json"}
    assert all((path.stat().st_mode & 0o777) == 0o444
               for path in runtime.iterdir())
    with pytest.raises(R.MetricReconciliationError):
        R.run_once(tmp_path)


def test_evidence_structure_and_top_level_flags_fail_closed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = synthetic_closed_evidence(monkeypatch)
    inspection = R.inspect_evidence(evidence)
    assert inspection["exact_24_by_12_structure"] is True
    assert inspection["inventory"]["unique_state_candidate_pairs"] == 288

    changed = copy.deepcopy(evidence)
    changed["complete"] = False
    body = {key: value for key, value in changed.items()
            if key != "calibration_evidence_digest"}
    changed["calibration_evidence_digest"] = R.C.digest(body)
    with pytest.raises(R.MetricReconciliationError):
        R.inspect_evidence(changed)

    duplicate = copy.deepcopy(evidence)
    duplicate["rows"][1]["candidate_index"] = 0
    body = {key: value for key, value in duplicate.items()
            if key != "calibration_evidence_digest"}
    duplicate["calibration_evidence_digest"] = R.C.digest(body)
    with pytest.raises(R.MetricReconciliationError):
        R.inspect_evidence(duplicate)


def test_runner_has_no_tensor_model_training_or_predictor_route() -> None:
    source = (ROOT / "scripts/reconcile_go2_attentive_metric_evidence_v1.py").read_text()
    forbidden = (
        "import torch", "from torch", "torch.load(", "load_state_dict(",
        "optimizer.step(", ".backward(", "predictor checkpoint",
    )
    assert all(token not in source for token in forbidden)
