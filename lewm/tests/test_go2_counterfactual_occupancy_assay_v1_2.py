#!/usr/bin/env python3
"""Focused tests for the frozen-probe counterfactual occupancy diagnostic.

The lightweight CI interpreter does not install torch.  Pure estimator
functions are therefore compiled directly from the assay's AST; this still
tests the implementation that will run in the scientific environment without
inventing a second copy of its formulas.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py"
PACKAGE = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "factorial_v1/spatial_retention/probe_package.json"
)
WEIGHTS = PACKAGE.with_name("probe_final_epoch.pt")


def _pure_namespace() -> dict[str, object]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    names = {
        "OccupancyAssayRefused", "_require", "occupied_counts", "pooled_iou",
        "episode_then_family", "t_summary", "_cell_effects",
    }
    selected = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
        *[node for node in tree.body
          if ((isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
              or (isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name)
                          and target.id in {"FAMILIES", "CELLS", "FROZEN_SEED_COUNT"}
                          for target in node.targets)))],
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "np": np,
        "math": math,
        "collections": __import__("collections"),
        "ray_v4": types.SimpleNamespace(UNKNOWN_CLASS=0, OCCUPIED_CLASS=2),
        "Mapping": dict,
        "Sequence": list,
        "Any": object,
    }
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


class CounterfactualOccupancyAssayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = _pure_namespace()

    def test_frozen_probe_package_and_weights_validate(self):
        self.assertEqual(
            hashlib.sha256(PACKAGE.read_bytes()).hexdigest(),
            "3d216f4e60851861d521705397ae0f43f783a8ceb1852685f42ab27ff0260c75",
        )
        package = json.loads(PACKAGE.read_text(encoding="utf-8"))
        stored = package.pop("package_digest")
        self.assertEqual(
            hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest(),
            stored,
        )
        self.assertEqual(
            stored,
            "b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686",
        )
        self.assertEqual(
            hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            "95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322",
        )
        self.assertTrue(package["qualification"]["qualified"])
        self.assertGreaterEqual(package["qualification"]["observed_iou"], 0.35)

    def test_observable_occupied_iou_ignores_unknown_target_cells(self):
        truth = np.asarray([[0, 2, 1], [0, 2, 1]], dtype=np.uint8)
        prediction = np.asarray([[2, 2, 2], [2, 1, 1]], dtype=np.uint8)
        result = self.ns["occupied_counts"](prediction, truth)
        self.assertEqual(result["observable_cells"], 4)
        self.assertEqual(result["occupied_intersection"], 1)
        self.assertEqual(result["occupied_union"], 3)
        self.assertAlmostEqual(result["observable_occupied_iou"], 1 / 3)

    def test_equal_family_and_corpus_weighting_remain_distinct(self):
        families = self.ns["FAMILIES"]
        rows, values = [], []
        for family_index, family in enumerate(families):
            repetitions = 4 if family_index == 0 else 1
            for candidate in range(repetitions):
                rows.append({
                    "episode_cluster_id": f"{family}-state",
                    "family": family,
                })
                values.append(float(family_index == 0))
        result = self.ns["episode_then_family"](values, rows)
        self.assertAlmostEqual(result["equal_family"], 1 / 8)
        self.assertAlmostEqual(result["corpus_weighted"], 4 / 11)

    def test_undefined_family_never_becomes_a_seven_family_average(self):
        families = self.ns["FAMILIES"]
        rows = [{
            "episode_cluster_id": f"{family}-state",
            "family": family,
        } for family in families]
        values = [None if family == "rough_local_dynamics" else 0.5
                  for family in families]
        result = self.ns["episode_then_family"](values, rows)
        self.assertIsNone(result["equal_family"])
        self.assertFalse(result["equal_family_available"])
        self.assertEqual(result["equal_family_defined_family_count"], 7)
        self.assertEqual(
            result["equal_family_missing_families"], ["rough_local_dynamics"])
        self.assertIsNone(result["per_family"]["rough_local_dynamics"])
        self.assertAlmostEqual(result["corpus_weighted"], 0.5)
        self.assertEqual(result["defined_rows"], 7)
        self.assertEqual(result["undefined_rows"], 1)
        self.assertEqual(result["episode_clusters"], 8)
        self.assertEqual(result["clusters_per_family"]["rough_local_dynamics"], 1)
        self.assertEqual(
            result["per_family_defined_cluster_count"]["rough_local_dynamics"],
            0,
        )

    def test_seed_effects_use_paired_quadruplets_and_df7(self):
        values = {
            "rgb_one_step": [0.4] * 8,
            "rgb_rollout": [0.5] * 8,
            "proprio_one_step": [0.45] * 8,
            "proprio_rollout": [0.57] * 8,
        }
        effects = self.ns["_cell_effects"](values)
        self.assertAlmostEqual(effects["B_RGB"]["mean"], 0.1)
        self.assertAlmostEqual(effects["B_prop"]["mean"], 0.12)
        self.assertAlmostEqual(effects["M_main_rollout"]["mean"], 0.11)
        self.assertAlmostEqual(
            effects["J_proprioception_by_rollout"]["mean"], 0.02)
        self.assertEqual(effects["B_RGB"]["t_critical_df7"], 2.3646242510102993)

    def test_true_target_gate_precedes_any_predicted_latent_access(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run_assay")
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        call_names = []
        for call in calls:
            if isinstance(call.func, ast.Name):
                call_names.append((call.func.id, call.lineno))
        gate_line = next(line for name, line in call_names
                         if name == "freeze_true_target_gate")
        prediction_line = next(line for name, line in call_names
                               if name == "load_prediction_indices")
        self.assertLess(gate_line, prediction_line)

        torch_loads = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == "torch"
                       and node.func.attr == "load"]
        self.assertEqual(len(torch_loads), 1)
        self.assertIsInstance(torch_loads[0].args[0], ast.Name)
        self.assertEqual(torch_loads[0].args[0].id, "PROBE_WEIGHTS_PATH")
        true_function = next(node for node in tree.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name == "score_true_targets")
        predicted_function = next(node for node in tree.body
                                  if isinstance(node, ast.FunctionDef)
                                  and node.name == "score_prediction_index")
        def layer_norm_calls(node):
            return [call for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "F"
                    and call.func.attr == "layer_norm"]
        self.assertEqual(len(layer_norm_calls(true_function)), 1)
        self.assertEqual(layer_norm_calls(predicted_function), [])
        self.assertIn(
            "raw final-block tokens rounded to float16; consumers reload float16 as ",
            source,
        )
        self.assertIn(
            "raster.output_labels, dtype=LABEL_DTYPE",
            source,
        )

    def test_labels_stage_returns_before_probe_or_prediction_access(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        labels_only_if = next(
            node for node in main.body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "stage"
            and any(isinstance(item, ast.Constant) and item.value == "labels"
                    for item in node.test.comparators)
        )
        labels_return = next(node.lineno for node in ast.walk(labels_only_if)
                             if isinstance(node, ast.Return))
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)]
        self.assertLess(
            labels_return,
            next(node.lineno for node in calls if node.func.id == "load_labels"),
        )
        self.assertLess(
            labels_return,
            next(node.lineno for node in calls if node.func.id == "run_assay"),
        )
        self.assertLess(
            labels_return,
            next(node.lineno for node in calls if node.func.id == "resolve_device"),
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a",
            source,
        )
        self.assertIn(
            "39545af7599da2f2a1bf171c050489eea9f8637137bc1a9c0af3a193d1aaaf3a",
            source,
        )
        self.assertIn(
            "0536504c46422a69733853786e45f906a0fa63defa9af7e4f7a63f1789fa1365",
            source,
        )
        self.assertIn(
            "seven-family averaging and imputation are forbidden",
            source,
        )


if __name__ == "__main__":
    unittest.main()
