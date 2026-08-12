"""Pure synthetic tests for the bounded 20-state scorer-transfer consumer."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import apply_go2_utility_scorer_to_counterfactual_development_v1_2 as T


def synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state_index in range(T.EXPECTED_STATES):
        for candidate_index in range(T.EXPECTED_CANDIDATES):
            rows.append({
                "state_id": f"state-{state_index}",
                "family": T.FAMILIES[state_index % len(T.FAMILIES)],
                "candidate": f"candidate-{candidate_index}",
                "utility": float(candidate_index),
            })
    return rows


class DevelopmentTransferTests(unittest.TestCase):
    def test_downstream_uses_final_non_overwriting_selector_amendment(self):
        self.assertEqual(
            T.S.STATE_SELECTOR.AMENDMENT_VERSION,
            "completion_horizon_reachability_v2")
        self.assertNotEqual(
            T.S.STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME,
            "state_selector_feasibility_receipt.json")
        self.assertEqual(
            tuple(T.S.SELECTOR_BINDING_KEYS),
            tuple(T.S.STATE_SELECTOR.ACTIVE_SELECTOR_BINDING_KEYS))

    def test_frozen_stage_a_rows_need_no_successor_selector_keys(self):
        states = [
            SimpleNamespace(state_index=index, state_id=f"state-{index}")
            for index in range(T.EXPECTED_STATES)
        ]
        row_by_pair = {}
        for state in states:
            for candidate_index in range(T.EXPECTED_CANDIDATES):
                row_by_pair[(state.state_id, candidate_index)] = {
                    "state_id": state.state_id,
                    "candidate_index": candidate_index,
                    "valid": True,
                    "oracle_outcome_equal": True,
                    "action_blocks": [[0.0] * 10 for _ in range(T.HORIZONS)],
                    "goal_binding_input": [0.0, 1.0, 1.0],
                    "utility": 0.0,
                    "progress": 0.0,
                    "safety": 0.0,
                    "completion": 0.0,
                }
        rows = T.ordered_rows(SimpleNamespace(
            states=states, row_by_pair=row_by_pair))
        self.assertEqual(len(rows), T.EXPECTED_BRANCHES)
        self.assertTrue(all(
            not set(T.S.SELECTOR_BINDING_KEYS).intersection(row)
            for row in rows))

    def test_failed_qualification_refuses_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "schema": "go2_utility_scorer_v1_2_qualification",
                "criteria": {"progress_spearman_ge_0.50": False},
                "qualified": False,
            }
            report["qualification_report_digest"] = T.legacy_digest(report)
            (root / "qualification.json").write_text(json.dumps(report))
            with mock.patch.object(T.S, "PACKAGE_DIR", root), \
                    mock.patch.object(T.torch, "load") as load:
                with self.assertRaisesRegex(T.DevelopmentTransferRefused,
                                            "did not pass every frozen criterion"):
                    T.validate_qualified_scorer()
                load.assert_not_called()

    def test_missing_selector_successor_refuses_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "schema": "go2_utility_scorer_v1_2_qualification",
                "criteria": {"all_frozen_criteria": True},
                "qualified": True,
                "qualification_evaluations": 1,
                "epoch_selection_permitted": False,
                "scorer_contract_v1_2_digest": T.contract_digest(),
            }
            report["qualification_report_digest"] = T.legacy_digest(report)
            (root / "qualification.json").write_text(json.dumps(report))
            with mock.patch.object(T.S, "PACKAGE_DIR", root), \
                    mock.patch.object(T.torch, "load") as load:
                with self.assertRaisesRegex(
                        T.DevelopmentTransferRefused,
                        "qualified scorer has no state_selector_amendment_digest"):
                    T.validate_qualified_scorer()
                load.assert_not_called()

    def test_live_selector_chain_is_revalidated_and_bound_before_weight_load(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)
            paths = (
                pool / "pre_identity_allocation_validation.json",
                pool / "candidate_allocation_manifest.json",
                pool / "state_manifest.json",
                pool / T.S.STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME,
                pool / T.S.STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME,
                pool / T.S.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME,
            )
            pre_identity = {"pre_identity_validation_digest": "i" * 64}
            allocation = {
                "allocation_manifest_digest": "m" * 64,
                "post_identity_pre_outcome_validation": {
                    "post_identity_validation_digest": "o" * 64,
                },
            }
            state_manifest = {"states": []}
            state_manifest["state_manifest_digest"] = T.hashlib.sha256(
                json.dumps(state_manifest, sort_keys=True).encode()
            ).hexdigest()
            disposition = {
                "mixed_precontract_disposition_receipt_digest":
                    "p" * 64,
            }
            paths[0].write_text(json.dumps(pre_identity))
            paths[1].write_text(json.dumps(allocation))
            paths[2].write_text(json.dumps(state_manifest))
            paths[3].write_text(json.dumps({"record": 3}))
            paths[4].write_text(json.dumps(disposition))
            paths[5].write_text(json.dumps({"record": 5}))
            selector = {
                "state_selector_amendment_digest": "a" * 64,
                "state_selector_feasibility_receipt_digest": "f" * 64,
                "preserved_state_revalidation_receipt_digest": "e" * 64,
            }
            qualification = {
                **selector,
                "corpus_bindings": {
                    "candidate_allocation_manifest_digest": "m" * 64,
                    "candidate_allocation_post_identity_validation_digest":
                        "o" * 64,
                    "pre_identity_allocation_validation_digest": "i" * 64,
                    "state_manifest_digest":
                        state_manifest["state_manifest_digest"],
                    "state_manifest_file_sha256": T.sha256_file(paths[2]),
                },
            }
            launch = {
                "source_repository_commit": "c" * 40,
                "clean_source_binding_digest": "b" * 64,
                "bound_implementations_digest": "a" * 64,
                "mixed_precontract_disposition_receipt_digest":
                    "p" * 64,
            }
            with mock.patch.object(
                    T.S, "_validate_clean_source_launch",
                    return_value=launch) as validate_launch, \
                    mock.patch.object(
                        T.S.CORPUS_BUILDER,
                        "load_active_state_manifest_for_consumption",
                        return_value=state_manifest,
                    ) as validate_live_manifest, \
                    mock.patch.object(
                        T.S, "validate_pre_identity_structural_validation"), \
                    mock.patch.object(
                        T.S, "allocation_manifest_digest",
                        return_value="m" * 64), \
                    mock.patch.object(
                        T.S.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"), \
                    mock.patch.object(
                        T.S, "_validate_selector_successor",
                        return_value=selector) as validate_selector:
                result = T._validate_live_selector_provenance(
                    qualification, pool_dir=pool)
            validate_launch.assert_called_once()
            validate_live_manifest.assert_called_once_with(
                paths[2], pool=T.S.EXPECTED_POOL)
            validate_selector.assert_called_once()
            self.assertEqual(result["selector_bindings"], selector)
            self.assertEqual(
                result["mixed_precontract_disposition_receipt_digest"],
                "p" * 64)
            self.assertFalse(result["scorer_weights_opened_during_validation"])
            self.assertFalse(result["predictor_artifacts_opened_during_validation"])
            self.assertEqual(
                result["verification_digest"],
                T.legacy_digest(result, ("verification_digest",)))

            with mock.patch.object(
                    T.S, "_validate_clean_source_launch", return_value=launch), \
                    mock.patch.object(
                        T.S.CORPUS_BUILDER,
                        "load_active_state_manifest_for_consumption",
                        return_value=state_manifest), \
                    mock.patch.object(
                        T.S, "validate_pre_identity_structural_validation"), \
                    mock.patch.object(
                        T.S, "allocation_manifest_digest",
                        return_value="m" * 64), \
                    mock.patch.object(
                        T.S.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"), \
                    mock.patch.object(
                        T.S, "_validate_selector_successor",
                        return_value=selector):
                with self.assertRaisesRegex(
                        T.DevelopmentTransferRefused,
                        "differs at state_selector_feasibility_receipt_digest"):
                    T._validate_live_selector_provenance(
                        {**qualification,
                         "state_selector_feasibility_receipt_digest": "x" * 64},
                        pool_dir=pool)

    def test_live_selection_replay_failure_precedes_selector_and_weight_access(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)
            names = (
                "pre_identity_allocation_validation.json",
                "candidate_allocation_manifest.json",
                "state_manifest.json",
            )
            for name in names:
                (pool / name).write_text("{}")
            with mock.patch.object(
                    T.S.CORPUS_BUILDER,
                    "load_active_state_manifest_for_consumption",
                    side_effect=RuntimeError(
                        "later replacement capture prefix is not canonical"
                    )) as replay, \
                    mock.patch.object(
                        T.S, "_validate_selector_successor") as selector, \
                    mock.patch.object(T.torch, "load") as load:
                with self.assertRaisesRegex(
                        T.DevelopmentTransferRefused,
                        "later replacement capture prefix"):
                    T._validate_live_selector_provenance(
                        {}, pool_dir=pool)
            replay.assert_called_once()
            selector.assert_not_called()
            load.assert_not_called()

    def test_live_phase2_failure_refuses_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "schema": "go2_utility_scorer_v1_2_qualification",
                "criteria": {"all_frozen_criteria": True},
                "qualified": True,
                "qualification_evaluations": 1,
                "epoch_selection_permitted": False,
                "scorer_contract_v1_2_digest": T.contract_digest(),
                "state_selector_amendment_digest":
                    T.S.STATE_SELECTOR.state_selector_amendment_digest(),
                "state_selector_feasibility_receipt_digest": "f" * 64,
                "preserved_state_revalidation_receipt_digest": "e" * 64,
            }
            report["qualification_report_digest"] = T.legacy_digest(report)
            (root / "qualification.json").write_text(json.dumps(report))
            with mock.patch.object(T.S, "PACKAGE_DIR", root), \
                    mock.patch.object(
                        T, "_validate_live_selector_provenance",
                        side_effect=T.DevelopmentTransferRefused(
                            "phase-2 exact-mask reachability failed")), \
                    mock.patch.object(T.torch, "load") as load:
                with self.assertRaisesRegex(
                        T.DevelopmentTransferRefused,
                        "phase-2 exact-mask reachability failed"):
                    T.validate_qualified_scorer()
            load.assert_not_called()

    def test_perfect_state_rankings_and_equal_family_aggregation(self):
        rows = synthetic_rows()
        scores = np.asarray([row["utility"] for row in rows], dtype=np.float32)
        states = T.state_metrics(rows, scores)
        result = T.aggregate_metrics(states)
        for weighting in ("equal_family", "corpus_weighted"):
            self.assertEqual(result[weighting]["normalised_rank_regret"], 0.0)
            self.assertEqual(result[weighting]["absolute_rank_regret"], 0.0)
            self.assertEqual(result[weighting]["top1_recovery"], 1.0)
            self.assertEqual(result[weighting]["top3_recovery"], 1.0)
            self.assertEqual(result[weighting]["pairwise_ordering_accuracy"], 1.0)
            self.assertEqual(result[weighting]["spearman_rank_correlation"], 1.0)

    def test_score_receipt_directly_binds_live_selector_provenance(self):
        scorer = mock.Mock()
        scorer.package_sha256 = "p" * 64
        scorer.qualification = {
            "state_selector_amendment_digest": "a" * 64,
            "state_selector_feasibility_receipt_digest": "f" * 64,
            "preserved_state_revalidation_receipt_digest": "e" * 64,
        }
        scorer.selector_provenance = {"verification_digest": "v" * 64}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            score = out / "score.f32"
            score.write_bytes(b"\0" * (T.EXPECTED_BRANCHES * 4))
            with mock.patch.object(T, "OUT_DIR", out), \
                    mock.patch.object(T, "contract_digest",
                                      return_value="c" * 64):
                receipt = T._score_receipt(
                    "synthetic", "i" * 64, scorer, score)
        self.assertEqual(
            {key: receipt[key] for key in T.S.SELECTOR_BINDING_KEYS},
            scorer.qualification)
        self.assertEqual(
            receipt["selector_provenance_verification_digest"], "v" * 64)

    def test_rank_regret_effect_uses_one_step_minus_rollout(self):
        cells: dict[int, dict[str, dict[str, object]]] = {}
        for seed in T.SEEDS:
            cells[seed] = {}
            for cell, regret in (
                ("rgb_one_step", 0.4), ("rgb_rollout", 0.3),
                ("proprio_one_step", 0.5), ("proprio_rollout", 0.3),
            ):
                cells[seed][cell] = {
                    "equal_family": {"normalised_rank_regret": regret},
                    "corpus_weighted": {"normalised_rank_regret": regret},
                    "per_family": {family: {"normalised_rank_regret": regret}
                                   for family in T.FAMILIES},
                }
        result = T.paired_factorial(
            cells, "equal_family", "normalised_rank_regret")
        self.assertAlmostEqual(result["B_RGB"]["mean"], 0.1)
        self.assertAlmostEqual(result["B_prop"]["mean"], 0.2)
        self.assertAlmostEqual(result["M"]["mean"], 0.15)
        self.assertAlmostEqual(result["J"]["mean"], 0.1)

    def test_true_target_normalisation_is_not_the_predicted_path(self):
        # Frozen Stage-A target shards are raw encoder tokens. Frozen predicted
        # shards are already layer-normalised; the prospective spec must keep
        # those two paths explicit and distinct.
        scorer = mock.Mock()
        scorer.qualification = {
            "qualification_report_digest": "q" * 64,
            **{key: key[0] * (40 if key == "source_repository_commit" else 64)
               for key in T.S.SCORER_PROVENANCE_BINDING_KEYS},
        }
        scorer.package_sha256 = "p" * 64
        scorer.selector_provenance = {
            "status": "PASS_LIVE_PRE_WEIGHT_SELECTOR_PROVENANCE_REVALIDATION",
            "verification_digest": "v" * 64,
        }
        with mock.patch.object(T, "contract_digest", return_value="c" * 64), \
                mock.patch.object(T, "sha256_file", return_value="s" * 64):
            spec = T.prospective_spec(scorer)
        self.assertIn("F.layer_norm", spec["true_target_handling"])
        self.assertIn("no second", spec["predicted_handling"])
        self.assertEqual(
            set(spec["scorer_selector_successor_bindings"]),
            set(T.S.SELECTOR_BINDING_KEYS))


if __name__ == "__main__":
    unittest.main()
