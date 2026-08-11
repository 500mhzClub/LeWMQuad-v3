"""Pure synthetic tests for the bounded 20-state scorer-transfer consumer."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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
               for key in T.S.LAUNCH_BINDING_KEYS},
        }
        scorer.package_sha256 = "p" * 64
        with mock.patch.object(T, "contract_digest", return_value="c" * 64), \
                mock.patch.object(T, "sha256_file", return_value="s" * 64):
            spec = T.prospective_spec(scorer)
        self.assertIn("F.layer_norm", spec["true_target_handling"])
        self.assertIn("no second", spec["predicted_handling"])


if __name__ == "__main__":
    unittest.main()
