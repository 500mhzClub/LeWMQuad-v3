from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from lewm.safety import occluded_goal_topological_belief_metrics_v1 as M
from lewm.safety import occluded_goal_topological_belief_v1_contract as C
from scripts import evaluate_occluded_goal_topological_belief_v1 as E
from scripts import run_occluded_goal_topological_belief_v1 as R


PARAMETERS = {
    "observation_softmax_temperature": 0.01,
    "action_compatible_edge_probability": 0.70,
    "transition_noise_probability": 0.01,
    "normalized_entropy_abstention_threshold": 0.25,
}


def _fake_source_freeze() -> dict[str, object]:
    return {
        "head_commit": "a" * 40,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            "independent_reducer": {
                "path": E.REDUCER_SOURCE_PATH,
                "bytes": 101,
                "sha256": "b" * 64,
            },
            "metrics_module": {
                "path": E.METRICS_SOURCE_PATH,
                "bytes": 202,
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
            total = sum(preprojection[index] for index in order)
            posterior = [
                preprojection[index] / total if index in set(order) else 0.0
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


class _EvidenceStore:
    def observation_evidence(
        self,
        observation_id: str,
        reference_ids: list[str],
        *,
        temperature: float,
        uniform: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del observation_id, temperature, uniform
        count = len(reference_ids)
        return np.zeros(count), np.full(count, 1.0 / count)


class _LabelGuard(dict):
    FORBIDDEN = {
        "true_node_id",
        "true_next_edge_id",
        "true_next_port_label",
        "goal_node_id",
        "global_goal_coordinates",
        "shortest_path_labels",
    }

    def __getitem__(self, key: str):
        if key in self.FORBIDDEN:
            raise AssertionError(f"non-oracle inference opened {key}")
        return super().__getitem__(key)


class RunnerPureTests(unittest.TestCase):
    def test_canonical_json_and_jsonl_have_exactly_one_lf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "document.json"
            ledger = root / "ledger.jsonl"
            R.atomic_json(document, {"z": 2, "a": 1})
            R.write_jsonl(ledger, ({"row": 1}, {"row": 2}))
            self.assertEqual(document.read_bytes(), b'{"a":1,"z":2}\n')
            self.assertEqual(ledger.read_bytes(), b'{"row":1}\n{"row":2}\n')
            self.assertNotIn(b"\n\n", document.read_bytes() + ledger.read_bytes())
            self.assertEqual(R.load_json(document), {"a": 1, "z": 2})
            self.assertEqual(R.load_jsonl(ledger), [{"row": 1}, {"row": 2}])

    def test_teacher_paths_are_unique_and_bit_balanced(self) -> None:
        paths = R.teacher_side_paths()
        self.assertEqual(len(paths), 96)
        self.assertEqual(len(set(paths)), 96)
        self.assertEqual([sum(path[index] for path in paths) for index in range(8)], [48] * 8)

    def test_panel_and_locked_metric_authorities_agree(self) -> None:
        panel, graph, queries = R.build_panel_documents()
        M.validate_graph_manifest(graph)
        M.validate_query_rows(queries, graph)
        self.assertEqual(panel["episode_count"], 96)
        self.assertEqual(panel["query_count"], 768)
        self.assertEqual(panel["role_counts"], {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16})
        self.assertEqual(panel["family_counts"], {family: 24 for family in C.FAMILY_IDS})
        self.assertTrue(panel["adequacy"]["phase_a_all_graph_nodes_observed"])
        self.assertEqual(panel["adequacy"]["loop_alias_cycle_episode_count"], 24)
        self.assertEqual(
            panel["identity_disjointness"]["comparisons"],
            {field: 0 for field in M.IDENTITY_COMPARISON_FIELDS},
        )
        for episode in graph["graphs"]:
            self.assertEqual(episode["identity_domain"], C.IDENTITY_DOMAIN)
            self.assertTrue(episode["episode_id"].startswith(C.EPISODE_ID_PREFIX))
            self.assertTrue(episode["scene_id"].startswith(C.SCENE_ID_PREFIX))
            self.assertTrue(episode["episode_path_id"].startswith(C.EPISODE_PATH_ID_PREFIX))
            self.assertEqual(episode["phase_a_traversal"][-1]["node_id"], episode["goal_node_id"])
            self.assertEqual(episode["stage_b_start_node_id"], episode["goal_node_id"])
            phase_a_ids = {row["observation_id"] for row in episode["phase_a_traversal"]}
            episode_queries = [row for row in queries if row["episode_id"] == episode["episode_id"]]
            episode_queries.sort(key=lambda row: row["query_index"])
            self.assertEqual(
                episode["stage_b_decision_start_node_id"],
                episode_queries[0]["true_node_id"],
            )
            self.assertTrue(episode["stage_b_prelude_edge_ids"])
            self.assertTrue(all(row["query_observation_id"] not in phase_a_ids for row in episode_queries))
            for query in episode_queries:
                self.assertEqual(
                    set(query["stage_b_choice_macros"]),
                    set(query["oracle_admissible_edge_ids"]),
                )
                self.assertEqual(
                    [row["candidate_index"] for row in query["stage_b_local_candidate_outcomes"]],
                    list(range(12)),
                )
                self.assertTrue(
                    any(row["oracle_admissible"] for row in query["stage_b_local_candidate_outcomes"])
                )
            if episode["family"] == "LOOP_ALIAS":
                witness = episode["family_adequacy_witness"]
                self.assertEqual(len(witness["cycle_node_ids"]), len(witness["cycle_edge_ids"]))

    def test_authority_failure_precedes_official_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "official-output"
            with mock.patch.object(R, "OUTPUT_ROOT", output), mock.patch.object(
                R, "require_source_freeze", return_value="f" * 40
            ), mock.patch.object(
                R,
                "_load_bound_prior_authorities",
                side_effect=R.ExperimentError("bound authority drift"),
            ):
                with self.assertRaisesRegex(R.ExperimentError, "bound authority drift"):
                    R.panel_stage()
            self.assertFalse(output.exists())

    def test_source_freeze_preflight_rejects_dirty_or_wrong_diff(self) -> None:
        head = "f" * 40

        def dirty_git(*arguments: str) -> str:
            values = {
                ("rev-parse", "HEAD"): head,
                ("rev-parse", "HEAD^"): R.PARENT_COMMIT,
                ("show", "-s", "--format=%s", "HEAD"): C.CONTRACT_FREEZE_COMMIT_SUBJECT,
                ("status", "--porcelain=v1", "--untracked-files=all"): " M changed.py",
            }
            return values[arguments]

        with mock.patch.object(R, "git", side_effect=dirty_git):
            with self.assertRaisesRegex(R.ExperimentError, "not clean"):
                R.require_source_freeze()

        def wrong_diff_git(*arguments: str) -> str:
            values = {
                ("rev-parse", "HEAD"): head,
                ("rev-parse", "HEAD^"): R.PARENT_COMMIT,
                ("show", "-s", "--format=%s", "HEAD"): C.CONTRACT_FREEZE_COMMIT_SUBJECT,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"): "unexpected.py",
            }
            return values[arguments]

        with mock.patch.object(R, "git", side_effect=wrong_diff_git):
            with self.assertRaisesRegex(R.ExperimentError, "changed-path"):
                R.require_source_freeze()

    def test_published_source_closure_rejects_byte_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"frozen = True\n")
            docs = dict(R.DOC_PATHS)
            docs["source_closure"] = root / "source_closure.json"
            closure = C.attach_content_digest(
                {
                    "schema": "occluded_goal_topological_belief_v1.source_closure.v1",
                    "parent_commit": R.PARENT_COMMIT,
                    "row_count": 1,
                    "rows": [
                        {
                            "path": "source.py",
                            "bytes": source.stat().st_size,
                            "sha256": R.sha256_file(source),
                        }
                    ],
                }
            )
            R.atomic_json(docs["source_closure"], closure)
            with mock.patch.object(R, "REPO_ROOT", root), mock.patch.object(
                R, "SOURCE_PATHS", ("source.py",)
            ), mock.patch.object(R, "DOC_PATHS", docs):
                R._validate_published_source_closure()
                source.write_bytes(b"frozen = False\n")
                with self.assertRaisesRegex(R.ExperimentError, "differs from closure"):
                    R._validate_published_source_closure()

    def test_runtime_contract_rejects_recomputed_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scientific = {
                "schema": "fake.scientific.v1",
                "source_parent_commit": R.PARENT_COMMIT,
                "output": {"root": str(root)},
            }
            runtime = C.attach_content_digest(
                {
                    "schema": "occluded_goal_topological_belief_v1.runtime_contract.v1",
                    "source_freeze_commit": "f" * 40,
                    "parent_commit": R.PARENT_COMMIT,
                    "scientific_contract": scientific,
                }
            )
            R.atomic_json(root / "contract.json", runtime)
            with mock.patch.object(R, "OUTPUT_ROOT", root), mock.patch.object(
                C, "build_contract", return_value=scientific
            ), mock.patch.object(C, "validate_contract", return_value=None):
                R._validate_runtime_contract("f" * 40)
                tampered_scientific = {**scientific, "schema": "fake.tampered.v1"}
                R.atomic_json(
                    root / "contract.json",
                    C.attach_content_digest(
                        {
                            "schema": "occluded_goal_topological_belief_v1.runtime_contract.v1",
                            "source_freeze_commit": "f" * 40,
                            "parent_commit": R.PARENT_COMMIT,
                            "scientific_contract": tampered_scientific,
                        }
                    ),
                )
                with self.assertRaisesRegex(R.ExperimentError, "differs from frozen module"):
                    R._validate_runtime_contract("f" * 40)

    def test_runtime_contract_accepts_json_round_trip_of_tuple_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scientific = {
                "schema": "fake.scientific.v1",
                "source_parent_commit": R.PARENT_COMMIT,
                "output": {"root": str(root)},
                "claims": {"does_not_establish": ("safety", "deployment")},
            }
            runtime = C.attach_content_digest(
                {
                    "schema": "occluded_goal_topological_belief_v1.runtime_contract.v1",
                    "source_freeze_commit": "f" * 40,
                    "parent_commit": R.PARENT_COMMIT,
                    "scientific_contract": scientific,
                }
            )
            R.atomic_json(root / "contract.json", runtime)
            with mock.patch.object(R, "OUTPUT_ROOT", root), mock.patch.object(
                C, "build_contract", return_value=scientific
            ), mock.patch.object(C, "validate_contract", return_value=None):
                R._validate_runtime_contract("f" * 40)

    def test_renderer_preserves_exact_aliases_and_goal_scope(self) -> None:
        left = R.observation_recipe(family="MIRRORED_JUNCTION", kind="QUERY", module=2, side=0, step=4)
        right = R.observation_recipe(family="MIRRORED_JUNCTION", kind="QUERY", module=2, side=1, step=4)
        self.assertEqual(left, right)
        np.testing.assert_array_equal(R.render_observation(left), R.render_observation(right))
        left_marker = R.render_observation(R.observation_recipe(family="MIRRORED_JUNCTION", kind="BRANCH_MARKER", module=2, side=0))
        right_marker = R.render_observation(R.observation_recipe(family="MIRRORED_JUNCTION", kind="BRANCH_MARKER", module=2, side=1))
        self.assertFalse(np.array_equal(left_marker, right_marker))
        goal = R.render_observation(R.observation_recipe(family="MIRRORED_JUNCTION", kind="GOAL_KEYFRAME"))
        self.assertFalse(np.array_equal(goal, R.render_observation(left)))

    def test_exact_token_descriptor_and_aligned_cosine(self) -> None:
        raw = np.asarray(
            [[1.0, 2.0, 4.0], [2.0, 5.0, 9.0], [3.0, 7.0, 8.0], [4.0, 6.0, 10.0]],
            dtype=np.float32,
        )
        with mock.patch.object(R, "TOKEN_SHAPE", (4, 3)):
            descriptor = R.token_layernorm_l2(raw)
            np.testing.assert_allclose(np.linalg.norm(descriptor, axis=-1), 1.0, atol=1e-6)
            store = R.SpatialDescriptorStore(
                descriptors_by_template={"a": descriptor, "b": -descriptor},
                template_by_capture={"qa": "a", "ra": "a", "rb": "b"},
            )
            similarity, likelihood = store.observation_evidence(
                "qa", ["ra", "rb"], temperature=0.1, uniform=False
            )
        np.testing.assert_allclose(similarity, [1.0, -1.0], atol=1e-6)
        self.assertGreater(likelihood[0], 0.999)

    def test_transition_and_query_bound_action_derangement(self) -> None:
        node_ids = ["a", "b", "c"]
        outgoing = {
            "a": [
                {"target_node_id": "b", "executed_action_label": "GO"},
                {"target_node_id": "c", "executed_action_label": "TURN"},
            ]
        }
        belief = np.asarray([1.0, 0.0, 0.0])
        predicted = R._transition(
            belief,
            "GO",
            node_ids=node_ids,
            node_index={value: index for index, value in enumerate(node_ids)},
            outgoing=outgoing,
            compatibility=0.8,
            noise=0.0,
        )
        np.testing.assert_allclose(predicted, [0.0, 0.8, 0.2])
        no_action = R._transition(
            belief,
            "GO",
            node_ids=node_ids,
            node_index={value: index for index, value in enumerate(node_ids)},
            outgoing=outgoing,
            compatibility=0.8,
            noise=0.0,
            action_consistency=False,
        )
        np.testing.assert_allclose(no_action, [0.0, 0.5, 0.5])
        actions = ["a", "b", "c", "d", "e"]
        shuffled, mapping = R._shuffled_actions("query-bound-id", actions)
        expected = list(M.deterministic_action_history_position_mapping("query-bound-id", len(actions)))
        self.assertEqual(mapping, expected)
        self.assertEqual(shuffled, [actions[index] for index in expected])
        self.assertTrue(all(index != source for index, source in enumerate(mapping)))

    def test_nonoracle_inference_never_opens_label_fields(self) -> None:
        episode, queries = R.build_episode_graph(
            family="REPEATED_CORRIDOR", family_rank=0, teacher_path=(0, 1, 0, 1, 0, 1, 0, 1)
        )
        guarded = _LabelGuard(queries[0])
        for condition in C.CONDITION_IDS:
            if condition == "ORACLE_PLACE_IDENTITY":
                continue
            R.infer_query_belief(
                episode=episode,
                query=guarded,
                descriptors_by_observation=_EvidenceStore(),
                parameters=PARAMETERS,
                condition_id=condition,
            )

    def test_tied_belief_projection_uses_frozen_node_id_order(self) -> None:
        tied = np.full(4, 0.25, dtype=np.float64)
        node_ids = ["node-z", "node-a", "node-y", "node-b"]
        map_projection = R._truncate(tied, "MAP_FILTER", node_ids)
        self.assertEqual(node_ids[int(np.argmax(map_projection))], "node-a")
        top_k_projection = R._truncate(tied, "TOP_K_BELIEF", node_ids)
        self.assertEqual(
            {node_ids[index] for index in np.flatnonzero(top_k_projection)},
            {"node-a", "node-b", "node-y"},
        )

        episode, queries = R.build_episode_graph(
            family="REPEATED_CORRIDOR",
            family_rank=0,
            teacher_path=(0, 1, 0, 1, 0, 1, 0, 1),
        )
        evidence = R.infer_query_belief(
            episode=episode,
            query=queries[0],
            descriptors_by_observation=_EvidenceStore(),
            parameters=PARAMETERS,
            condition_id="CURRENT_FRAME_NEAREST_NODE",
        )
        selected_index = int(np.argmax(evidence["posterior_probabilities"]))
        self.assertEqual(evidence["node_ids"][selected_index], min(evidence["node_ids"]))

    def test_stage_b_old_ranker_feature_shapes(self) -> None:
        from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as OLD

        previous = [0.1, 0.0, -0.2]
        history = [[0.05, 0.1] for _ in range(15)]
        names, base, query, anchor = R.build_stage_b_ranker_inputs(
            [0.2, 0.0, 0.45],
            OLD,
            previous_command=previous,
            control_history=history,
        )
        self.assertEqual(len(names), 12)
        self.assertEqual(base.shape, (12, OLD.BASE_FEATURE_DIM))
        self.assertEqual(query.shape, (12, OLD.QUERY_FEATURE_DIM))
        self.assertEqual(anchor.shape, (12,))
        self.assertTrue(np.isfinite(base).all() and np.isfinite(query).all() and np.isfinite(anchor).all())
        np.testing.assert_allclose(base[:, 94:97], np.tile(previous, (12, 1)))
        np.testing.assert_allclose(
            base[:, 97:127], np.tile(np.asarray(history).reshape(1, -1), (12, 1))
        )

    def test_stage_b_absent_branch_creates_no_conditional_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = C.attach_content_digest(
                {
                    "schema": "occluded_goal_topological_belief_v1.stage_a_metrics.v1",
                    "experiment_id": C.EXPERIMENT_ID,
                    "decision": {"stage_b_authorized": False},
                }
            )
            R.atomic_json(root / "stage_a_metrics.json", metrics)
            with mock.patch.object(R, "OUTPUT_ROOT", root), mock.patch.object(
                R, "require_source_freeze", return_value="f" * 40
            ):
                disposition = R.stage_b(local_ranker=lambda *_args: {})
            self.assertFalse(disposition["stage_b_executed"])
            self.assertFalse((root / "stage_b_trace.jsonl").exists())
            self.assertFalse((root / "stage_b_metrics.json").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "fake encoder needs the supported torch runtime")
class RunnerFakeEndToEndTests(unittest.TestCase):
    def test_fake_encoder_full_stage_sequence_and_independent_reducer(self) -> None:
        import torch

        class FakeArm:
            preprocessing_digest = "FAKE_TEST_ENCODER_PREPROCESSING_V1"

            def build(self, device: object, dtype: object) -> None:
                self.device = device
                self.dtype = dtype

            @staticmethod
            def preprocess_array(image: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0

            @staticmethod
            def tokens(pixels: torch.Tensor) -> torch.Tensor:
                means = pixels.mean(dim=(1, 2, 3), keepdim=False)
                base = torch.arange(12, dtype=torch.float32, device=pixels.device).reshape(1, 4, 3)
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
                "ranker_latency_ms": 0.25,
                "planning_latency_ms": 0.30,
            }

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "official-output"
            receipt_path = parent / "external-regeneration.json"
            patches = (
                mock.patch.object(R, "OUTPUT_ROOT", root),
                mock.patch.object(R, "EXTERNAL_REGENERATION_RECEIPT", receipt_path),
                mock.patch.object(R, "require_source_freeze", return_value="f" * 40),
                mock.patch.object(R, "TOKEN_SHAPE", (4, 3)),
                mock.patch.object(R, "infer_query_belief", side_effect=_fake_inference(signal=True)),
                mock.patch.object(E, "_observe_source_freeze", return_value=_fake_source_freeze()),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                R.panel_stage()
                R.render_stage()
                R.encode_stage(batch_size=64, arm_factory=FakeArm)
                calibration = R.calibrate_stage()
                self.assertEqual(len(calibration["grid_results"]), 144)
                self.assertEqual(len(calibration["grid_query_results"]), 144 * 128)
                metrics_a = R.stage_a()
                self.assertEqual(metrics_a["decision"]["primary_classification"], "TOPOLOGICAL_MAP_SUFFICIENT")
                self.assertTrue(metrics_a["decision"]["stage_b_authorized"])
                metrics_b = R.stage_b(local_ranker=fake_local_ranker)
                trace_rows = R.load_jsonl(root / "stage_b_trace.jsonl")
                self.assertGreaterEqual(len(trace_rows), 3 * 16 * 2)
                self.assertLessEqual(len(trace_rows), 3 * 16 * 16)
                self.assertEqual(
                    sum(row["terminal"] for row in trace_rows), 3 * 16
                )
                self.assertIn(metrics_b["decision"]["primary_classification"], C.STAGE_B_CLASSIFICATIONS)
                receipt = E.build_regeneration_receipt(root)
                E.emit_regeneration_receipt(root, receipt_path, receipt)
                result = R.report_stage()
                self.assertTrue(result["independent_regeneration"]["validated_mapping"]["pass"])
                self.assertTrue(result["stage_b_executed"])
                self.assertEqual(result["secondary_classifications"], ["TOPOLOGICAL_MAP_SUFFICIENT"])
                expected = set(C.RUNTIME_OUTPUT_PATHS.values())
                self.assertEqual({path.name for path in root.iterdir()}, expected)
                E.validate_existing_regeneration_receipt(root, receipt_path)
                report = (root / "result.md").read_text(encoding="utf-8")
                self.assertIn(C.CONSTRUCTED_SET_CAVEAT, report)
                self.assertIn("REQUIREMENTS_ACQUISITION_REQUIRED", report)


if __name__ == "__main__":
    unittest.main()
