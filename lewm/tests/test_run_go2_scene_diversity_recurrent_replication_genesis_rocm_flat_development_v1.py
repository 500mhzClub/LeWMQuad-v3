from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_flat_development_v1_plan as plan_builder
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_flat_development_v1 as harness


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_PLAN = plan_builder.QUALIFICATION_PLAN_OUTPUT
QUALIFICATION_PLAN_SHA256 = (
    "87a400d8d2688ed58c1a0dd61e4121dfa35374381d10d00b44738d544e3853b2"
)
QUALIFICATION_PLAN_BYTE_COUNT = 367_782


def _fake_cached_box_obj(
    size_xyz_m, *, tiles_per_m=0.7, cache_dir=None
):
    return str(Path(cache_dir or ".") / f"{tuple(size_xyz_m)}-{tiles_per_m}.obj")


cached_box_obj = _fake_cached_box_obj


def _fake_renderer(size_xyz_m):
    return cached_box_obj(size_xyz_m)


class _FakeRolloutConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FlatDevelopmentQualificationHarnessTest(unittest.TestCase):
    def test_source_has_no_imported_module_writes_or_adapter_context_calls(self):
        path = Path(harness.__file__).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(alias.asname or alias.name for alias in node.names)
        writes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = (
                    list(node.targets)
                    if isinstance(node, ast.Assign)
                    else list(node.targets)
                    if isinstance(node, ast.Delete)
                    else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in imported_names
                    ):
                        writes.append(f"{target.value.id}.{target.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"setattr", "delattr"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(node.func.attr.startswith("_configured_"))
        self.assertEqual(writes, [])
        self.assertNotIn("kernel._build_rollout_runner =", source)
        execute_source = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "execute_qualification"
            ),
        )
        self.assertIsNotNone(execute_source)
        self.assertLess(
            execute_source.index("_reserve_qualification("),
            execute_source.index("_run_rocm_egl_preflight("),
        )

    def test_exact_static_plan_validation_precedes_fresh_root(self):
        self.assertFalse(plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists())
        plan, binding, scenes = harness._read_and_validate_plan(
            QUALIFICATION_PLAN,
            expected_sha256=QUALIFICATION_PLAN_SHA256,
            expected_byte_count=QUALIFICATION_PLAN_BYTE_COUNT,
        )
        self.assertEqual(plan["attempt_id"], plan_builder.QUALIFICATION_ATTEMPT_ID)
        self.assertEqual(binding["sha256"], QUALIFICATION_PLAN_SHA256)
        self.assertEqual([scenes[index]["scene_index"] for index in harness.PROBE_ORDER], [12, 0])
        self.assertFalse(plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists())

    def test_worker_environment_accepts_only_cpython_locale_coercion(self):
        plan = {
            "execution_contract": {
                "environment": plan_builder.rocm_execution_environment("qualification")
            }
        }
        exact = dict(plan["execution_contract"]["environment"])
        with mock.patch.dict(os.environ, {**exact, "LC_CTYPE": "C.UTF-8"}, clear=True):
            harness._require_worker_environment(plan)
        with mock.patch.dict(os.environ, {**exact, "LANG": "en_GB.UTF-8"}, clear=True):
            with self.assertRaises(harness.FlatQualificationError):
                harness._require_worker_environment(plan)

    def test_worker_local_runtime_does_not_mutate_renderer_or_input_map(self):
        original_global = _fake_renderer.__globals__["cached_box_obj"]
        original_runtime = {
            "cached_box_obj": _fake_cached_box_obj,
            "build_textured_v03_scene": _fake_renderer,
            "RolloutConfig": _FakeRolloutConfig,
        }
        with tempfile.TemporaryDirectory() as temporary:
            scene_root = Path(temporary) / "scene"
            local, cache_root, audit = harness._clone_worker_runtime(
                original_runtime, scene_root=scene_root
            )
            self.assertIs(original_runtime["build_textured_v03_scene"], _fake_renderer)
            self.assertIs(
                _fake_renderer.__globals__["cached_box_obj"], original_global
            )
            self.assertIsNot(local, original_runtime)
            self.assertIsNot(local["build_textured_v03_scene"], _fake_renderer)
            self.assertEqual(
                local["build_textured_v03_scene"]((1.0, 2.0, 3.0)),
                str(cache_root / "(1.0, 2.0, 3.0)-0.7.obj"),
            )
            config = local["RolloutConfig"](seed=7)
            self.assertEqual(config.kwargs, {"foot_contact_source": "zero", "seed": 7})
            with self.assertRaises(harness.FlatQualificationError):
                local["RolloutConfig"](foot_contact_source="native")
            self.assertTrue(audit["original_renderer_globals_unchanged"])

    def test_worker_plan_loader_is_post_reservation_and_does_not_call_parity(self):
        raw = json.loads(QUALIFICATION_PLAN.read_bytes())
        with tempfile.TemporaryDirectory(dir=ROOT / ".generated/dev") as temporary:
            namespace = Path(temporary)
            attempt = namespace / "attempt_v1"
            collection = attempt / "collection"
            collection.mkdir(parents=True)
            raw["output_root"] = str(collection.resolve())
            environment = copy.deepcopy(raw["execution_contract"]["environment"])
            environment["GS_CACHE_FILE_PATH"] = str((attempt / "quadrants_cache").resolve())
            raw["execution_contract"]["environment"] = environment
            plan_path = namespace / "plan.json"
            encoded = json.dumps(raw, indent=2, sort_keys=True).encode() + b"\n"
            plan_path.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            with (
                mock.patch.object(plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt),
                mock.patch.object(plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection),
                mock.patch.object(
                    plan_builder,
                    "validate_flat_plan",
                    side_effect=AssertionError("post-reservation parity called"),
                ),
            ):
                loaded, binding, scenes = harness._read_worker_plan(
                    plan_path,
                    expected_sha256=digest,
                    expected_byte_count=len(encoded),
                )
            self.assertEqual(loaded["output_root"], str(collection.resolve()))
            self.assertEqual(binding["sha256"], digest)
            self.assertEqual(len(scenes), 64)


if __name__ == "__main__":
    unittest.main()
