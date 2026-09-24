from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest import mock

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_plan as plan_builder
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3 as harness


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_PLAN = plan_builder.QUALIFICATION_PLAN_OUTPUT
QUALIFICATION_PLAN_SHA256 = (
    "6a055839ab9bb6fe45b9cb5864e8f3c87e75f468dd7e9c26e8c950e4a6fedb78"
)
QUALIFICATION_PLAN_BYTE_COUNT = 355_206


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


class CpuFlatDevelopmentQualificationHarnessTest(unittest.TestCase):
    def test_source_has_no_module_writes_or_adapter_context_calls(self):
        path = Path(harness.__file__).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(
                    alias.asname or alias.name.split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(
                    alias.asname or alias.name for alias in node.names
                )
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
            execute_source.index("_run_graphics_preflight("),
        )
        worker_argv_source = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_worker_argv"
            ),
        )
        self.assertNotIn("_validate_python_invocation", worker_argv_source)

    def test_exact_static_plan_validation_precedes_fresh_root(self):
        self.assertFalse(plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists())
        plan, binding, scenes = harness._read_and_validate_plan(
            QUALIFICATION_PLAN,
            expected_sha256=QUALIFICATION_PLAN_SHA256,
            expected_byte_count=QUALIFICATION_PLAN_BYTE_COUNT,
        )
        self.assertEqual(plan["attempt_id"], plan_builder.QUALIFICATION_ATTEMPT_ID)
        self.assertEqual(binding["sha256"], QUALIFICATION_PLAN_SHA256)
        self.assertEqual(plan["branch_mechanism"], "parallel_lockstep_envs_no_restore")
        self.assertEqual(
            [scenes[index]["scene_index"] for index in harness.PROBE_ORDER],
            [12, 0],
        )
        self.assertFalse(plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists())

    def test_child_environment_is_exact_eleven_keys_with_fixed_home_and_path(self):
        plan = {
            "execution_contract": {
                "environment": copy.deepcopy(plan_builder.CPU_EXECUTION_ENVIRONMENT)
            }
        }
        with mock.patch.dict(os.environ, {"HOME": "/tmp/ambient"}, clear=True):
            child = harness._child_environment(plan)
        self.assertEqual(child, plan_builder.CPU_EXECUTION_ENVIRONMENT)
        self.assertEqual(len(child), 11)
        self.assertEqual(child["HOME"], "/home/andrewknowles")
        self.assertEqual(child["PATH"], "/usr/bin:/bin")
        exact = dict(plan_builder.CPU_EXECUTION_ENVIRONMENT)
        with mock.patch.dict(
            os.environ, {**exact, "LC_CTYPE": "C.UTF-8"}, clear=True
        ):
            harness._require_worker_environment(plan)
        with mock.patch.dict(os.environ, {**exact, "HOME": "/tmp"}, clear=True):
            with self.assertRaises(harness.CpuFlatQualificationError):
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
            self.assertIs(_fake_renderer.__globals__["cached_box_obj"], original_global)
            self.assertIsNot(local, original_runtime)
            self.assertEqual(
                local["build_textured_v03_scene"]((1.0, 2.0, 3.0)),
                str(cache_root / "(1.0, 2.0, 3.0)-0.7.obj"),
            )
            config = local["RolloutConfig"](seed=7)
            self.assertEqual(config.kwargs, {"foot_contact_source": "zero", "seed": 7})
            with self.assertRaises(RuntimeError):
                local["RolloutConfig"](foot_contact_source="native")
            self.assertTrue(audit["original_renderer_globals_unchanged"])

    def test_worker_plan_loader_does_not_repeat_historical_validation(self):
        raw = json.loads(QUALIFICATION_PLAN.read_bytes())
        with tempfile.TemporaryDirectory(dir=ROOT / ".generated/dev") as temporary:
            namespace = Path(temporary)
            attempt = namespace / "attempt_v1"
            collection = attempt / "collection"
            collection.mkdir(parents=True)
            raw["output_root"] = str(collection.resolve())
            plan_path = namespace / "plan.json"
            encoded = json.dumps(raw, indent=2, sort_keys=True).encode() + b"\n"
            plan_path.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            with (
                mock.patch.object(plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt),
                mock.patch.object(plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection),
                mock.patch.object(
                    plan_builder,
                    "validate_qualification_plan",
                    side_effect=AssertionError("post-reservation validation called"),
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

    def test_pilot_diagnostics_are_persisted_in_worker_failure(self):
        diagnostics = {
            "phase": "common_history_policy_step",
            "history_index": 0,
            "command_tick_index": 0,
            "policy_step_index": 0,
            "synchronization_audits": [
                {
                    "state_id": "state-a",
                    "passed": False,
                    "lane_state_sha256s": ["a" * 64, "b" * 64],
                }
            ],
        }
        error = pilot.PilotDiagnosticError(
            "first-step divergence", diagnostics=diagnostics
        )
        with tempfile.TemporaryDirectory(dir=ROOT / ".generated/dev") as temporary:
            attempt = Path(temporary) / "attempt_v1"
            collection = attempt / "collection"
            (collection / "scene_results").mkdir(parents=True)
            with (
                mock.patch.object(plan_builder, "QUALIFICATION_ATTEMPT_ROOT", attempt),
                mock.patch.object(plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection),
            ):
                harness._write_worker_failure(
                    error,
                    scene_index=12,
                    plan_binding={"path": "/plan", "sha256": "1" * 64, "byte_count": 1},
                    reservation_binding={
                        "path": "/reservation",
                        "sha256": "2" * 64,
                        "byte_count": 2,
                    },
                    orchestrator_pid=123,
                )
                failure = json.loads(
                    (collection / "scene_results/012.failure.json").read_bytes()
                )
        self.assertEqual(failure["failure"]["type"], "PilotDiagnosticError")
        self.assertEqual(failure["failure"]["diagnostics"], diagnostics)
        self.assertTrue(failure["diagnostics_persisted_if_present"])
        self.assertFalse(failure["authorizes_scientific_plan_release"])

    def test_png_receipt_is_rehashed_decoded_and_unique(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            collection = Path(temporary)
            relative = PurePosixPath("scenes/train/scene-a/rgb/frame.png")
            path = collection.joinpath(*relative.parts)
            path.parent.mkdir(parents=True)
            image = Image.new("RGB", (224, 224), color=(1, 2, 3))
            image.save(path, format="PNG")
            raw = path.read_bytes()
            frame = {
                "artifact_id": "frame-a",
                "frame_identity": "frame-a",
                "path": str(relative),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "width": 224,
                "height": 224,
                "mode": "RGB",
                "format": "PNG",
                "camera_valid": True,
                "low_information": False,
                "low_info_reasons": [],
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
            paths: set[str] = set()
            identities: set[str] = set()
            checked = harness._validate_png_frame_receipt(
                frame,
                collection_root=collection,
                expected_prefix=PurePosixPath("scenes/train/scene-a/rgb"),
                seen_paths=paths,
                seen_identities=identities,
            )
            self.assertEqual(checked, frame)
            with self.assertRaises(harness.CpuFlatQualificationError):
                harness._validate_png_frame_receipt(
                    frame,
                    collection_root=collection,
                    expected_prefix=PurePosixPath("scenes/train/scene-a/rgb"),
                    seen_paths=paths,
                    seen_identities=identities,
                )


if __name__ == "__main__":
    unittest.main()
