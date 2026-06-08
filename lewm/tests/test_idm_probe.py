import json
import unittest

import numpy as np

from scripts.probe_idm_decodability import _features, _fit_probe, _r2_report


class IdmProbeTest(unittest.TestCase):
    def test_true_pair_detects_transition_signal_beyond_state_and_shuffle(self) -> None:
        rng = np.random.default_rng(7)
        train_z = rng.normal(size=(800, 2, 8)).astype(np.float32)
        eval_z = rng.normal(size=(400, 2, 8)).astype(np.float32)
        train_y = (train_z[:, 1, :2] - train_z[:, 0, :2]).astype(np.float32)
        eval_y = (eval_z[:, 1, :2] - eval_z[:, 0, :2]).astype(np.float32)
        train_features = _features(train_z, seed=11)
        eval_features = _features(eval_z, seed=13)

        scores = {}
        for index, name in enumerate(("state", "true_pair", "shuffled_next")):
            prediction, _, _ = _fit_probe(
                train_features[name],
                train_y,
                eval_features[name],
                alphas=(1.0, 10.0, 100.0),
                seed=17 + index,
            )
            scores[name] = _r2_report(eval_y, prediction)["pooled_r2"]

        self.assertGreater(scores["true_pair"], 0.99)
        self.assertGreater(scores["true_pair"] - scores["state"], 0.4)
        self.assertGreater(scores["true_pair"] - scores["shuffled_next"], 0.4)

    def test_constant_channel_report_is_strict_json(self) -> None:
        truth = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        report = _r2_report(truth, truth)

        self.assertIsNone(report["per_channel_r2"][0])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
