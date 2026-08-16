"""Focused source/synthetic tests for frozen V1.3 latent attribution."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import diagnose_go2_scorer_v1_3_latent_dependence_v1 as DIAG


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / (
    "scripts/diagnose_go2_scorer_v1_3_latent_dependence_v1.py")


def _source() -> str:
    return SOURCE_PATH.read_text()


def _function_source(name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    node = next(value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_source_closure(commit: str = "0" * 40):
    unsigned = {
        "schema": DIAG.CONTRACT.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": commit,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "files": {
            path: {"path": path, "sha256": _digest(path), "byte_count": 1}
            for path in DIAG.CONTRACT.SOURCE_CLOSURE_PATHS
        },
    }
    return DIAG.CONTRACT.validate_source_closure({
        **unsigned,
        DIAG.CONTRACT.SOURCE_CLOSURE_SELF_KEY:
            DIAG.CONTRACT.canonical_digest(unsigned),
    })


def _rows(states: int = 2, candidates: int = 12):
    return [{
        "state_id": f"state-{state}",
        "state_identity_digest": _digest(f"state:{state}"),
        "candidate_index": candidate,
        "branch_identity_digest": _digest(f"branch:{state}:{candidate}"),
        "training_view_row_digest": _digest(f"row:{state}:{candidate}"),
        "_latent_index": state * candidates + candidate,
    } for state in range(states) for candidate in range(candidates)]


def test_exact_A_to_G_suite_is_frozen_in_source() -> None:
    assert DIAG.VARIANT_IDS == (
        "A_matched",
        "B_within_state_candidate_derangement",
        "C_horizon_reversed",
        "D_fixed_token_permutation",
        "E_spatial_mean_repeated",
        "F_fit_mean_trajectory",
        "G_H1_only", "G_H2_only", "G_H3_only", "G_H4_only",
    )
    assert DIAG.RAW_SHAPE == (4, 768, 1024)
    assert DIAG.FIT_ROWS == 1152
    assert DIAG.CALIBRATION_ROWS == 288


def test_hash_sort_rotate_one_is_a_within_state_derangement() -> None:
    rows = _rows()
    first, receipt = DIAG.within_state_derangement(rows)
    second, repeated = DIAG.within_state_derangement(rows)
    assert np.array_equal(first, second)
    assert receipt == repeated
    assert sorted(first.tolist()) == list(range(len(rows)))
    for destination, source in enumerate(first):
        assert destination != int(source)
        assert rows[destination]["state_id"] == rows[int(source)]["state_id"]
        assert (rows[destination]["candidate_index"]
                != rows[int(source)]["candidate_index"])
    assert receipt["algorithm"] == DIAG.CONTRACT.TRANSFORMATION_SUITE[
        "B_WITHIN_STATE_CANDIDATE_DERANGEMENT"]["algorithm"]
    assert len(receipt["mapping_digest"]) == 64


def test_one_token_permutation_is_fixed_shared_and_nonidentity() -> None:
    first, receipt = DIAG.fixed_token_permutation()
    second, repeated = DIAG.fixed_token_permutation()
    assert np.array_equal(first, second)
    assert receipt == repeated
    assert sorted(first.tolist()) == list(range(768))
    assert not np.array_equal(first, np.arange(768))
    assert tuple(first) == DIAG.CONTRACT.SPATIAL_TOKEN_PERMUTATION
    assert receipt["contract_permutation_digest"] == (
        DIAG.CONTRACT.SPATIAL_TOKEN_PERMUTATION_DIGEST)
    assert receipt["namespace"] == DIAG.CONTRACT.TRANSFORMATION_SUITE[
        "D_FIXED_SPATIAL_TOKEN_PERMUTATION"]["namespace"]
    assert len(receipt["permutation_digest"]) == 64


def test_full_trajectory_transforms_precede_spatial_mean() -> None:
    raw = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    fit_mean = (1_000.0 + raw).astype(np.float32)
    permutation = np.asarray([2, 3, 0, 1, 5, 4], dtype=np.int64)

    matched = DIAG.apply_raw_transform(
        DIAG.A_MATCHED, raw, token_permutation=permutation,
        fit_mean_trajectory=fit_mean)
    reversed_h = DIAG.apply_raw_transform(
        DIAG.C_HORIZON_REVERSED, raw, token_permutation=permutation,
        fit_mean_trajectory=fit_mean)
    permuted = DIAG.apply_raw_transform(
        DIAG.D_TOKEN_PERMUTED, raw, token_permutation=permutation,
        fit_mean_trajectory=fit_mean)
    repeated = DIAG.apply_raw_transform(
        DIAG.E_SPATIAL_MEAN_REPEATED, raw,
        token_permutation=permutation, fit_mean_trajectory=fit_mean)
    constant = DIAG.apply_raw_transform(
        DIAG.F_FIT_MEAN_TRAJECTORY, raw,
        token_permutation=permutation, fit_mean_trajectory=fit_mean)

    assert matched.shape == raw.shape
    assert np.array_equal(reversed_h, raw[::-1])
    assert np.array_equal(permuted, raw[:, permutation, :])
    expected_mean = raw.mean(axis=1, dtype=np.float32)
    assert np.array_equal(
        repeated, np.repeat(expected_mean[:, None, :], 6, axis=1))
    assert np.array_equal(constant, fit_mean)
    assert np.allclose(DIAG.spatial_mean(permuted), expected_mean)
    assert np.allclose(DIAG.spatial_mean(repeated), expected_mean)


def test_G_keeps_exactly_one_matched_horizon() -> None:
    raw = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    fit_mean = np.full_like(raw, -7.0)
    permutation = np.asarray([2, 3, 0, 1, 4], dtype=np.int64)
    for keep, variant in enumerate(DIAG.G_SINGLE_HORIZON):
        transformed = DIAG.apply_raw_transform(
            variant, raw, token_permutation=permutation,
            fit_mean_trajectory=fit_mean)
        for horizon in range(4):
            expected = raw[horizon] if horizon == keep else fit_mean[horizon]
            assert np.array_equal(transformed[horizon], expected)


def test_fit_mean_is_streaming_float64_fit_only_and_sampled() -> None:
    shape = (4, 3, 2)
    arrays = [np.full(shape, value, dtype=np.float16)
              for value in (1.0, 2.0, 3.0, 6.0)]

    class Store:
        def __getitem__(self, index):
            return arrays[int(index)]

    rows = [{
        "_latent_index": index,
        "state_id": f"fit-state-{index // 2}",
        "candidate_index": index % 2,
        "branch_identity_digest": _digest(f"fit-branch:{index}"),
        "training_view_row_digest": _digest(f"fit:{index}"),
    } for index in range(4)]
    mean, receipt = DIAG.compute_fit_mean_trajectory(
        rows, Store(), raw_shape=shape, expected_rows=4,
        sample_ticks=(1, 3, 4))
    assert mean.dtype == np.float32
    assert np.array_equal(mean, np.full(shape, 3.0, dtype=np.float32))
    assert [sample["rows_accumulated"] for sample in receipt["samples"]] \
        == [1, 3, 4]
    assert receipt["fit_mean_trajectory_digest"] == DIAG.array_digest(mean)
    assert receipt["contract"] == DIAG.CONTRACT.TRANSFORMATION_SUITE[
        "F_FIT_SET_MEAN_TRAJECTORY"]["statistic"]
    assert len(receipt["fit_mean_receipt_digest"]) == 64


def test_installed_contract_must_exactly_bind_live_source_closure(
        tmp_path: Path) -> None:
    managed = tmp_path / DIAG.CONTRACT.GENERATED_ROOT
    managed.mkdir(parents=True)
    closure = _synthetic_source_closure()
    installed = DIAG.CONTRACT.contract(closure)
    (managed / "diagnostic_contract.json").write_text(json.dumps(installed))

    assert DIAG.managed_generated_root(tmp_path) == managed.absolute()
    assert DIAG.load_bound_contract(closure, root=tmp_path) == installed
    with pytest.raises(
            DIAG.LatentDependenceError,
            match="does not bind live source closure"):
        DIAG.load_bound_contract(
            _synthetic_source_closure("1" * 40), root=tmp_path)


def test_complete_metric_delta_aligns_per_state_rows() -> None:
    value = {
        "overall": {"rows": 2, "score": 0.7,
                    "per_state": [{"state_id": "b", "score": 0.8},
                                  {"state_id": "a", "score": 0.6}]},
        "label": "ignored",
    }
    reference = {
        "overall": {"rows": 2, "score": 0.5,
                    "per_state": [{"state_id": "a", "score": 0.1},
                                  {"state_id": "b", "score": 0.3}]},
        "label": "ignored",
    }
    delta = DIAG.metric_delta(value, reference)
    assert delta["overall"]["rows"] == 0.0
    assert np.isclose(delta["overall"]["score"], 0.2)
    assert delta["overall"]["per_state"] == [
        {"state_id": "b", "score": 0.5},
        {"state_id": "a", "score": 0.5},
    ]
    assert "label" not in delta


def test_invariance_receipt_reports_exact_head_errors() -> None:
    matched = {key: np.asarray([0.0, 1.0], dtype=np.float64)
               for key in ("progress", "safety", "completion", "utility")}
    changed = {key: value.copy() for key, value in matched.items()}
    changed["utility"][1] += 5e-7
    receipt = DIAG.prediction_invariance_error(
        matched, changed, atol=1e-6)
    assert receipt["all_within_absolute_tolerance"] is True
    assert np.isclose(receipt["heads"]["utility"]["max_abs_error"], 5e-7)


def test_matched_terminal_replay_is_closed_and_fail_closed() -> None:
    assert DIAG.MATCHED_TERMINAL_REPLAY_ATOL == 1e-6
    frozen = {
        "overall": {
            "score": 0.5,
            "undefined": None,
            "per_state": [{"state_id": "s0", "score": 0.25}],
        },
        "per_family": {"family-a": {"score": 0.75}},
        "per_stratum": {"stratum-a": {"score": 0.125}},
    }
    exact = DIAG.matched_terminal_replay(frozen, frozen)
    assert exact["matches_frozen_terminal"] is True
    assert exact["canonical_equal"] is True
    assert exact["max_abs_error"] == 0.0

    within = {
        **frozen,
        "overall": {**frozen["overall"], "score": 0.5 + 5e-7},
    }
    replay = DIAG.matched_terminal_replay(within, frozen, atol=1e-6)
    assert replay["matches_frozen_terminal"] is True
    assert replay["canonical_equal"] is False
    assert np.isclose(replay["max_abs_error"], 5e-7)

    outside = {
        **within,
        "overall": {**frozen["overall"], "score": 0.5 + 2e-6},
    }
    assert DIAG.matched_terminal_replay(
        outside, frozen, atol=1e-6)["matches_frozen_terminal"] is False
    extra = {**frozen, "unexpected": {}}
    assert DIAG.matched_terminal_replay(
        extra, frozen)["matches_frozen_terminal"] is False


def test_predecessor_bridge_is_narrow_and_never_weakens_v13() -> None:
    source = _function_source("load_frozen_predecessor")
    assert "load_preserved_encoded_training_view_for_replacement(" in source
    assert "load_and_validate_training_terminal_for_consumption(" not in source
    assert "qualification_report_digest" in source
    assert "latent_checkpoint_sha256" in source
    assert "state_dict_digest" in source
    assert "optimizer_state_digest" in source
    assert "rng_state_digest" in source


def test_transform_freeze_precedes_the_one_calibration_session() -> None:
    source = _function_source("run_once")
    contract_load = source.index("load_bound_contract(")
    authorisation = source.index(
        "label=\"latent-dependence evaluation authorisation\"")
    materialise = source.index("materialise_variant_spatial_means(")
    evaluate = source.index("evaluate_variants(")
    assert contract_load < authorisation < materialise < evaluate
    assert source.count("evaluate_variants(") == 1
    assert '"calibration_diagnostic_session_count": 1' in source


def test_result_carries_complete_grouped_metrics_deltas_and_invariances() -> None:
    evaluate = _function_source("evaluate_variants")
    run = _function_source("run_once")
    for field in (
        '"overall"', '"per_family"', '"per_stratum"',
        '"delta_vs_matched"',
        '"delta_vs_frozen_no_latent_baseline"',
    ):
        assert field in evaluate
    assert '"frozen_no_latent_baseline"' in run
    assert '"matched_condition_terminal_replay"' in run
    assert '"architecture_invariance_checks"' in run


def test_existing_result_reopens_through_live_terminal_validator() -> None:
    run = _function_source("run_once")
    validate = _function_source("validate_result_for_consumption")
    assert "return validate_result_for_consumption(" in run
    for field in (
        "build_source_closure(", "load_bound_contract(",
        "evaluation_authorisation_digest", "transformation_freeze_digest",
        "matched_condition_terminal_replay", "predictor_utility_shards_opened",
        "independently_recompute_terminal_evidence(",
        "validate_recomputed_terminal_evidence(",
    ):
        assert field in validate


def test_full_read_only_replay_rebuilds_A_to_G_and_every_metric_scope() -> None:
    replay = _function_source("independently_recompute_terminal_evidence")
    for operation in (
        "load_frozen_predecessor(",
        "within_state_derangement(",
        "fixed_token_permutation(",
        "compute_fit_mean_trajectory(",
        "materialise_variant_spatial_means(",
        "evaluate_variants(",
    ):
        assert operation in replay
    for field in (
        '"transformation_freeze"',
        '"results"',
        '"architecture_invariance_checks"',
        '"matched_condition_terminal_replay"',
        '"frozen_no_latent_baseline"',
    ):
        assert field in replay
    assert "publish_json_once(" not in replay


def test_resigned_B_to_G_metric_rewrite_fails_independent_replay() -> None:
    freeze = {
        "variant_ids": list(DIAG.VARIANT_IDS),
        "freeze_digest": _digest("freeze"),
    }
    results = {
        variant: {
            "metrics": {
                "overall": {"safety": {"auc_any_hazard": 0.5}},
                "per_family": {"family-a": {"score": 0.25}},
                "per_stratum": {"stratum-a": {"score": 0.125}},
            },
            "delta_vs_matched": {"overall": {"score": 0.0}},
            "delta_vs_frozen_no_latent_baseline": {
                "overall": {"score": 0.1}},
        }
        for variant in DIAG.VARIANT_IDS
    }
    evidence = {
        "transformation_freeze": freeze,
        "results": results,
        "architecture_invariance_checks": {
            DIAG.C_HORIZON_REVERSED: {"all_within_absolute_tolerance": True},
            DIAG.D_TOKEN_PERMUTED: {"all_within_absolute_tolerance": True},
            DIAG.E_SPATIAL_MEAN_REPEATED: {
                "all_within_absolute_tolerance": True},
        },
        "matched_condition_terminal_replay": {
            "matches_frozen_terminal": True, "verdict": "MATCH"},
        "frozen_no_latent_baseline": {"overall": {"score": 0.4}},
    }
    result = {field: copy.deepcopy(evidence[field])
              for field in DIAG.RECOMPUTED_RESULT_FIELDS}
    authorisation = {"transformation_freeze": copy.deepcopy(freeze)}
    receipt = DIAG.validate_recomputed_terminal_evidence(
        result=result, authorisation=authorisation, recomputed=evidence)
    assert receipt["writes"] == 0
    assert set(receipt["result_field_digests"]) == set(
        DIAG.RECOMPUTED_RESULT_FIELDS)

    # Model a maliciously rewritten and re-self-signed terminal by changing a
    # real B metric; the self-signature layer cannot make it match the replay.
    tampered = copy.deepcopy(result)
    tampered["results"][DIAG.B_WITHIN_STATE_DERANGEMENT]["metrics"][
        "overall"]["safety"]["auc_any_hazard"] = 0.875
    with pytest.raises(
            DIAG.LatentDependenceError,
            match="recorded results does not exactly replay frozen inputs"):
        DIAG.validate_recomputed_terminal_evidence(
            result=tampered, authorisation=authorisation,
            recomputed=evidence)


def test_attentive_prerequisite_calls_the_full_latent_validator() -> None:
    attentive = (ROOT / (
        "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py"
    )).read_text()
    assert "LATENT.validate_result_for_consumption(root=root)" in attentive


def test_source_has_no_training_predictor_package_or_raw_prediction_route() -> None:
    source = _source()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports
    for forbidden in (
        "train_registered_model(", "register_initialisation(",
        "torch.optim", "scorer_package_path(",
        '"raw_predictions_persisted": True',
        '"final_200_state_corpus_generated": True',
    ):
        assert forbidden not in source
    assert '"training_executions": 0' in source
    assert '"raw_predictions_persisted": False' in source
    assert '"scorer_package_published": False' in source


def test_contract_is_imported_without_redefining_custody_authority() -> None:
    source = _source()
    assert "go2_scorer_failure_attribution_v1_contract as CONTRACT" in source
    assert DIAG.STATUS == DIAG.CONTRACT.STATUS
    assert DIAG.TRANSFORMATIONS == DIAG.CONTRACT.TRANSFORMATION_SUITE
    assert DIAG.CONTRACT.GENERATED_ROOT.parts[-1] == (
        "go2_scorer_failure_attribution_v1")
    assert "sealed" not in source.lower()
