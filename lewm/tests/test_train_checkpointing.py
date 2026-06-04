from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.train_lewm import (
    EpochRandomSampler,
    RunningAverages,
    _find_latest_checkpoint,
    _num_batches,
    _validate_partial_resume_config,
)


class TrainCheckpointingTests(unittest.TestCase):
    def test_epoch_random_sampler_resumes_at_sample_offset(self):
        sampler = EpochRandomSampler(list(range(20)), seed=7)
        sampler.set_epoch(3)
        full_order = list(sampler)

        sampler.set_epoch(3, start_sample=8)

        self.assertEqual(list(sampler), full_order[8:])
        self.assertEqual(len(sampler), 12)

    def test_latest_checkpoint_prefers_partial_checkpoint_in_newer_epoch(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            completed = tmp_path / "lewm_seq4_e5.pt"
            older_partial = tmp_path / "lewm_seq4_e5_b010000.pt"
            newer_partial = tmp_path / "lewm_seq4_e6_b005000.pt"
            for checkpoint in (completed, older_partial, newer_partial):
                checkpoint.touch()

            self.assertEqual(_find_latest_checkpoint(tmp_path, max_seq_len=4), newer_partial)

    def test_latest_checkpoint_prefers_completed_checkpoint_in_same_epoch(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            partial = tmp_path / "lewm_seq4_e6_b070000.pt"
            completed = tmp_path / "lewm_seq4_e6.pt"
            for checkpoint in (partial, completed):
                checkpoint.touch()

            self.assertEqual(_find_latest_checkpoint(tmp_path, max_seq_len=4), completed)

    def test_running_averages_round_trip(self):
        stats = RunningAverages()
        stats.update({"loss": 2.0}, weight=2)
        stats.update({"loss": 5.0}, weight=1)

        restored = RunningAverages.from_state_dict(stats.state_dict())

        self.assertEqual(restored.means(), {"loss": 3.0})

    def test_partial_resume_config_rejects_batch_size_change(self):
        checkpoint = {
            "data_loader_config": {
                "sampler": "epoch_random",
                "shuffle_seed": 0,
                "batch_size": 128,
                "drop_last": True,
                "num_samples": 1000,
            },
            "epoch_stats": {"totals": {}, "counts": {}},
        }
        args = Namespace(shuffle_seed=0, batch_size=64, drop_last=True)

        with self.assertRaisesRegex(RuntimeError, "batch_size"):
            _validate_partial_resume_config(checkpoint, list(range(1000)), args)

    def test_num_batches(self):
        cases = [
            (10, 4, True, 2),
            (10, 4, False, 3),
            (0, 4, False, 0),
        ]
        for num_samples, batch_size, drop_last, expected in cases:
            with self.subTest(
                num_samples=num_samples,
                batch_size=batch_size,
                drop_last=drop_last,
            ):
                self.assertEqual(
                    _num_batches(num_samples, batch_size, drop_last=drop_last),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
