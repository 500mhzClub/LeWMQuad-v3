from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from lewm.safety import occluded_goal_topological_belief_metrics_v2 as M
from lewm.safety import occluded_goal_topological_belief_v2_contract as C
from scripts import evaluate_occluded_goal_topological_belief_v2 as E
from scripts import run_occluded_goal_topological_belief_v2 as R


def _fake_source_freeze() -> dict[str, object]:
    return {
        "head_commit": "f" * 40,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            "independent_reducer": {
                "path": E.REDUCER_SOURCE_PATH,
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "metrics_module": {
                "path": E.METRICS_SOURCE_PATH,
                "bytes": 1,
                "sha256": "b" * 64,
            },
            "v1_pure_reducer_helpers": {
                "path": E.V1_REDUCER_SOURCE_PATH,
                "bytes": 1,
                "sha256": "c" * 64,
            },
        },
    }


def _fake_inference(*, signal: bool):
    def infer(
        *,
        episode: dict,
        query: dict,
        descriptors_by_observation: object,
        parameters: dict,
        condition_id: str,
    ) -> dict:
        del descriptors_by_observation, parameters
        node_ids = [str(node["node_id"]) for node in episode["nodes"]]
        count = len(node_ids)
        uniform = [1.0 / count] * count
        true_index = node_ids.index(str(query["true_node_id"]))
        true_mass = [1.0 if index == true_index else 0.0 for index in range(count)]
        persistent = condition_id in {
            "MAP_FILTER",
            "TOP_K_BELIEF",
            "FULL_BELIEF",
            "NO_ACTION_CONSISTENCY",
            "SHUFFLED_ACTION_HISTORY",
            "NO_OBSERVATION_LIKELIHOOD",
        }
        prior = true_mass if signal and persistent else uniform
        preprojection = list(prior)
        if condition_id in {
            "CURRENT_FRAME_NEAREST_NODE",
            "FIXED_WINDOW_SEQUENCE",
            "MAP_FILTER",
        }:
            best = min(
                range(count), key=lambda index: (-preprojection[index], node_ids[index])
            )
            posterior = [1.0 if index == best else 0.0 for index in range(count)]
        elif condition_id == "TOP_K_BELIEF":
            order = sorted(
                range(count), key=lambda index: (-preprojection[index], node_ids[index])
            )[: C.TOP_K]
            selected = set(order)
            total = sum(preprojection[index] for index in order)
            posterior = [
                preprojection[index] / total if index in selected else 0.0
                for index in range(count)
            ]
        elif condition_id == "ORACLE_PLACE_IDENTITY":
            posterior = true_mass
        else:
            posterior = list(preprojection)
        action_count = len(query["history_executed_action_labels"])
        mapping = list(range(action_count))
        if condition_id == "SHUFFLED_ACTION_HISTORY":
            mapping = list(
                M.deterministic_action_history_position_mapping(
                    str(query["query_id"]), action_count
                )
            )
        return {
            "node_ids": node_ids,
            "observation_similarities": [0.0] * count,
            "observation_likelihoods": uniform,
            "transition_prior_probabilities": prior,
            "preprojection_probabilities": preprojection,
            "posterior_probabilities": posterior,
            "action_history_position_mapping": mapping,
        }

    return infer


class RunnerV2PureTests(unittest.TestCase):
    def test_canonical_array_hash_includes_shape_dtype_layout_and_bytes(self) -> None:
        value = np.asarray([[1, 2]], dtype=np.uint8)
        expected = hashlib.sha256(
            b'{"dtype":"|u1","layout":"C","shape":[1,2]}\x00' + bytes((1, 2))
        ).hexdigest()
        self.assertEqual(R.canonical_array_sha256(value), expected)
        self.assertNotEqual(
            R.canonical_array_sha256(value),
            R.canonical_array_sha256(value.reshape(2, 1)),
        )

    def test_preoutcome_guard_rejects_extra_directory_and_broken_symlink(self) -> None:
        for kind in ("directory", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("contract.json", *R.REUSABLE_V1_LEAVES):
                    (root / name).write_bytes(b"x")
                if kind == "directory":
                    (root / "extra").mkdir()
                else:
                    (root / "extra").symlink_to(root / "missing")
                with mock.patch.object(R, "OUTPUT_ROOT", root), mock.patch.object(
                    R, "EXTERNAL_REGENERATION_RECEIPT", root.parent / "external.json"
                ):
                    with self.assertRaisesRegex(R.ExperimentError, "inventory drift"):
                        R._preoutcome_leaf_guard()

    def test_preprocessing_rejects_wrong_shape_and_dtype(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("supported torch runtime unavailable")
        import torch

        class BadShape:
            @staticmethod
            def preprocess_array(_image: np.ndarray) -> torch.Tensor:
                return torch.zeros((3, 4, 4), dtype=torch.float32)

        class BadDtype:
            @staticmethod
            def preprocess_array(_image: np.ndarray) -> torch.Tensor:
                return torch.zeros((3, 384, 512), dtype=torch.float64)

        with tempfile.TemporaryDirectory() as directory:
            for arm in (BadShape(), BadDtype()):
                with self.assertRaisesRegex(
                    R.ExperimentError, "preprocessing output contract drift"
                ):
                    R._preprocess_one(
                        arm, np.zeros((168, 224, 3), dtype=np.uint8), Path(directory)
                    )

    def test_singleton_repeat_mismatch_has_exact_terminal_disposition(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("supported torch runtime unavailable")
        import torch

        construction = {"count": 0}

        class DriftingArm:
            preprocessing_digest = C.PREPROCESSING_DIGEST

            def __init__(self) -> None:
                construction["count"] += 1
                self.serial = construction["count"]

            def build(self, _device: object, _dtype: object) -> None:
                return None

            @staticmethod
            def preprocess_array(_image: np.ndarray) -> torch.Tensor:
                return torch.zeros((3, 384, 512), dtype=torch.float32)

            def tokens(self, _pixels: torch.Tensor) -> torch.Tensor:
                return torch.full((1, 2, 2), float(self.serial), dtype=torch.float32)

        image = np.zeros((168, 224, 3), dtype=np.uint8)
        pixel_sha = R.canonical_array_sha256(image)
        with mock.patch.object(R, "TOKEN_SHAPE", (2, 2)):
            first = R._run_encoder_pass(
                pass_index=1,
                pixel_hashes=[pixel_sha],
                image_by_hash={pixel_sha: image},
                arm_factory=DriftingArm,
                fake_encoder=True,
            )
            with self.assertRaisesRegex(
                R.ExperimentError, C.CANONICAL_SINGLETON_ENCODER_NONDETERMINISM
            ):
                R._run_encoder_pass(
                    pass_index=2,
                    pixel_hashes=[pixel_sha],
                    image_by_hash={pixel_sha: image},
                    arm_factory=DriftingArm,
                    fake_encoder=True,
                    expected_raw=first["raw_tokens"],
                    expected_descriptors=first["spatial_descriptors"],
                    expected_preprocessed_hashes=first["preprocessed_sha256"],
                    prior_encoder_instance=first["_encoder_instance_guard"],
                )

    def test_stage_b_absent_branch_opens_no_cache_or_ranker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            R.atomic_json(root / "stage_a_metrics.json", {"minimal": True})
            with mock.patch.object(R, "OUTPUT_ROOT", root), mock.patch.object(
                R, "require_runtime", return_value="f" * 40
            ), mock.patch.object(
                M, "stage_a_authorizes_stage_b", return_value=False
            ), mock.patch.object(
                R, "validate_canonical_cache"
            ) as cache, mock.patch.object(
                R.V1, "FrozenCurrentVisualLocalRanker"
            ) as ranker:
                disposition = R.stage_b()
            self.assertFalse(disposition["stage_b_executed"])
            cache.assert_not_called()
            ranker.assert_not_called()


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "fake encoder needs the supported torch runtime",
)
class RunnerV2FakeEndToEndTests(unittest.TestCase):
    def test_bind_encode_science_reducer_and_report(self) -> None:
        import torch

        class FakeArm:
            preprocessing_digest = C.PREPROCESSING_DIGEST

            def build(self, device: object, _dtype: object) -> None:
                self.device = device

            @staticmethod
            def preprocess_array(image: np.ndarray) -> torch.Tensor:
                return torch.full(
                    (3, 384, 512),
                    float(image.mean() / 255.0),
                    dtype=torch.float32,
                )

            @staticmethod
            def tokens(pixels: torch.Tensor) -> torch.Tensor:
                means = pixels.mean(dim=(1, 2, 3))
                base = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
                return base + means.reshape(-1, 1, 1)

        def fake_local_ranker(
            current_tokens: np.ndarray,
            waypoint: list[float],
            *,
            previous_command: list[float],
            control_history: list[list[float]],
        ) -> dict[str, object]:
            self.assertEqual(tuple(current_tokens.shape), (4, 3))
            self.assertEqual(len(previous_command), 3)
            self.assertEqual(np.asarray(control_history).shape, (15, 2))
            selected = 3 if float(waypoint[2]) > 0.0 else 4
            scores = [0.0] * len(C.STAGE_B_LOCAL_CANDIDATE_IDS)
            scores[selected] = 1.0
            return {
                "candidate_names": list(C.STAGE_B_LOCAL_CANDIDATE_IDS),
                "candidate_scores": scores,
                "ranker_latency_ms": 0.1,
                "planning_latency_ms": 0.2,
            }

        with tempfile.TemporaryDirectory(dir="/home/andrewknowles") as directory:
            parent = Path(directory)
            root = parent / "official-v2"
            receipt_path = parent / "external-receipt.json"
            patches = (
                mock.patch.object(R, "OUTPUT_ROOT", root),
                mock.patch.object(R, "EXTERNAL_REGENERATION_RECEIPT", receipt_path),
                mock.patch.object(R, "require_source_freeze", return_value="f" * 40),
                mock.patch.object(C, "OUTPUT_ROOT", root),
                mock.patch.object(R, "TOKEN_SHAPE", (4, 3)),
                mock.patch.object(R.V1, "TOKEN_SHAPE", (4, 3)),
                mock.patch.object(
                    R.V1, "infer_query_belief", side_effect=_fake_inference(signal=True)
                ),
                mock.patch.object(E, "EXPECTED_TOKEN_SHAPE", (4, 3)),
                mock.patch.object(
                    E,
                    "_observe_source_freeze",
                    side_effect=lambda _module: _fake_source_freeze(),
                ),
            )
            with (
                patches[0], patches[1], patches[2], patches[3], patches[4],
                patches[5], patches[6], patches[7], patches[8]
            ):
                R.bind_inputs_stage()
                R.encode_stage(arm_factory=FakeArm)
                cache = R.validate_canonical_cache()
                self.assertTrue(cache["pass"])
                calibration = R.calibrate_stage()
                self.assertEqual(len(calibration["grid_query_results"]), 144 * 128)
                stage_a = R.stage_a()
                self.assertTrue(stage_a["decision"]["stage_b_authorized"])
                stage_b = R.stage_b(local_ranker=fake_local_ranker)
                self.assertIn(
                    stage_b["decision"]["primary_classification"],
                    C.STAGE_B_CLASSIFICATIONS,
                )
                receipt = E.build_regeneration_receipt(
                    root, source_freeze_observation=_fake_source_freeze()
                )
                E.emit_regeneration_receipt(root, receipt_path, receipt)
                E.validate_existing_regeneration_receipt(root, receipt_path)
                result = R.report_stage()
                self.assertTrue(result["stage_b_executed"])
                self.assertFalse(result["v1_invalid_science_reused"])
                self.assertEqual(
                    result["encoding_determinism_receipt"]["counts"][
                        "unique_pixel_rows"
                    ],
                    157,
                )
                report = (root / "result.md").read_text(encoding="utf-8")
                for condition in C.CONDITION_IDS:
                    self.assertIn(condition, report)
                self.assertIn("FULL versus MAP gate", report)
                self.assertIn("Fresh calibration selected grid index", report)
                self.assertIn("Prepublication runtime/storage observation", report)
                self.assertEqual(
                    {path.name for path in root.iterdir()},
                    set(C.RUNTIME_OUTPUT_PATHS.values()),
                )


if __name__ == "__main__":
    unittest.main()
