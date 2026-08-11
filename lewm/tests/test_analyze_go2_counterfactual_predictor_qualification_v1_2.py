#!/usr/bin/env python3
"""Synthetic-only tests for the counterfactual predictor assay."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as A


class CounterfactualPredictorAssayTest(unittest.TestCase):
    def test_legacy_pilot_digest_uses_published_default_json_encoding(self):
        payload = {"label": "two-sided ± interval"}
        self.assertNotEqual(A.legacy_json_digest(payload), A.json_digest(payload))

    def test_predictor_boundary_has_only_observed_state_and_candidate_plan(self):
        state = A.PlanningState(
            state_index=0,
            state_id="state",
            family=A.FAMILIES[0],
            scene_id="scene",
            episode_cluster_id="cluster",
            context_key="state",
            candidate_names=tuple(f"candidate_{i}" for i in range(12)),
            candidate_indices=tuple(range(12)),
            action_blocks=tuple(
                tuple(tuple(float(i + h) for _ in range(A.P.ACTION_DIM))
                      for h in range(4))
                for i in range(12)),
            proprio_history=tuple(tuple(0.0 for _ in range(5 * A.P.PROPRIO_DIM))
                                    for _ in range(3)),
            control_history=tuple(tuple(0.0 for _ in range(5 * A.P.CONTROL_DIM))
                                   for _ in range(3)),
        )
        captured = {}

        def fake_unroll(model, context, actions, proprio=None, control=None,
                        max_h=4):
            captured.update({"context": tuple(context.shape),
                             "actions": [tuple(value.shape) for value in actions],
                             "proprio": None if proprio is None else tuple(proprio.shape),
                             "control": tuple(control.shape), "max_h": max_h})
            return [torch.zeros(12, 2, 3) for _ in range(4)]

        stats = {
            "mean": [0.0] * A.P.PROPRIO_DIM,
            "std": [1.0] * A.P.PROPRIO_DIM,
            "control_mean": [0.0] * A.P.CONTROL_DIM,
            "control_std": [1.0] * A.P.CONTROL_DIM,
        }
        with mock.patch.object(A, "TOKENS", 2), \
                mock.patch.object(A, "TOKEN_DIM", 3), \
                mock.patch.object(A, "_read_f16_shard",
                                  return_value=np.zeros((3, 2, 3), np.float32)), \
                mock.patch.object(A.P, "unroll", side_effect=fake_unroll):
            predicted = A.predict_state(
                object(), state, {"observed": True}, stats, True,
                torch.device("cpu"))
        self.assertEqual(predicted.shape, (12, 4, 2, 3))
        self.assertEqual(captured["context"], (12, 3, 2, 3))
        self.assertEqual(captured["actions"], [(12, 10)] * 4)
        self.assertEqual(captured["proprio"], (12, 3, 5, 30))
        self.assertEqual(captured["control"], (12, 3, 5, 2))
        self.assertNotIn("target", A.predict_state.__code__.co_varnames[:7])

    def test_direct_metric_matches_frozen_ratio_and_zero_mask_is_available(self):
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prediction = target.clone()
        current = -target
        scored = A.direct_metrics(
            prediction, target, current, torch.tensor([True, True]))
        self.assertEqual(scored["changed_cosine"], 1.0)
        self.assertEqual(scored["persistence_changed_cosine"], -1.0)
        self.assertEqual(scored["advantage_over_persistence"], 2.0)
        self.assertEqual(scored["normalised_error_vs_persistence"], 0.0)

        unavailable = A.direct_metrics(
            prediction, target, current, torch.tensor([False, False]))
        self.assertFalse(unavailable["changed_metric_available"])
        self.assertIsNone(unavailable["changed_cosine"])
        self.assertEqual(unavailable["full_token_cosine"], 1.0)

    def test_retrieval_metrics_and_chance_references(self):
        names = [f"candidate_{index}" for index in range(A.EXPECTED_CANDIDATES)]
        result = A.retrieval_metrics(np.eye(A.EXPECTED_CANDIDATES), names)
        self.assertEqual(result["top1"], 1.0)
        self.assertEqual(result["top3"], 1.0)
        self.assertEqual(result["mean_reciprocal_rank"], 1.0)
        self.assertEqual(result["mean_rank"], 1.0)
        self.assertEqual(result["pairwise_accuracy"], 1.0)
        self.assertAlmostEqual(result["chance_references"]["top1"], 1 / 12)
        self.assertAlmostEqual(result["chance_references"]["top3"], 3 / 12)
        self.assertEqual(result["chance_references"]["mean_rank"], 6.5)

    def test_tie_order_is_frozen_candidate_index(self):
        names = [f"candidate_{index}" for index in range(A.EXPECTED_CANDIDATES)]
        result = A.retrieval_metrics(
            np.zeros((A.EXPECTED_CANDIDATES, A.EXPECTED_CANDIDATES)), names)
        self.assertEqual(result["winner_indices"], [0] * A.EXPECTED_CANDIDATES)
        self.assertEqual(result["ranks"], list(range(1, A.EXPECTED_CANDIDATES + 1)))
        self.assertEqual(result["own_wrong_exact_tie_rate"], 1.0)

    def test_equal_family_and_corpus_weighting_are_separate(self):
        names = [f"candidate_{index}" for index in range(A.EXPECTED_CANDIDATES)]
        family_sequence = []
        for index, family in enumerate(A.FAMILIES):
            family_sequence.extend([family] * (3 if index < 4 else 2))
        records = []
        for state_index, family in enumerate(family_sequence):
            value = A.FAMILIES.index(family) / 10.0
            direct = [{
                "changed_cosine": value,
                "normalised_error_vs_persistence": 1.0 - value,
                "persistence_changed_cosine": 0.0,
                "advantage_over_persistence": value,
                "prediction_mse": 1.0 - value,
                "persistence_mse": 1.0,
                "full_token_cosine": value,
                "full_token_persistence_cosine": 0.0,
                "full_token_normalised_error_vs_persistence": 1.0 - value,
                "changed_tokens": 10,
                "total_tokens": A.TOKENS,
                "changed_metric_available": True,
            } for _ in names]
            retrieval = A.retrieval_metrics(np.eye(A.EXPECTED_CANDIDATES), names)
            horizon = {
                "direct": direct,
                "retrieval_similarity_matrix":
                    np.eye(A.EXPECTED_CANDIDATES).tolist(),
                "retrieval": retrieval,
            }
            records.append({
                "state_index": state_index,
                "state_id": f"state_{state_index}",
                "family": family,
                "candidate_names": names,
                "per_horizon": {str(h): horizon for h in range(1, 5)},
            })
        aggregate = A.aggregate_records(records)["per_horizon"]["1"]
        self.assertAlmostEqual(
            aggregate["equal_family"]["direct"]["changed_cosine"], 0.35)
        self.assertAlmostEqual(
            aggregate["corpus_weighted"]["direct"]
            ["token_pooled_changed_cosine"], 0.31)

    def test_seed_interval_uses_df7(self):
        result = A.t_interval([0.0] * A.FROZEN_N)
        self.assertEqual(result["n"], 8)
        self.assertEqual(result["t_critical_df7"], 2.3646242510102993)
        self.assertEqual(result["two_sided_95_t_interval"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
