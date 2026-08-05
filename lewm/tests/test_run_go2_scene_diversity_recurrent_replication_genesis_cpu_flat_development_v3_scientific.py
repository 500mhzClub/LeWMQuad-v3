from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_plan as plan_builder
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific as runner


ROOT = Path(__file__).resolve().parents[2]
PLAN = plan_builder.SCIENTIFIC_PLAN_OUTPUT
PLAN_SHA256 = (
    "0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29"
)
PLAN_BYTE_COUNT = 359_692


class CpuFlatV3ScientificRunnerTest(unittest.TestCase):
    def test_source_has_no_imported_module_writes_or_monkeypatch_context(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.asname or alias.name.split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imported.update(alias.asname or alias.name for alias in node.names)
        writes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = (
                    list(node.targets)
                    if isinstance(node, (ast.Assign, ast.Delete))
                    else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in imported
                    ):
                        writes.append(f"{target.value.id}.{target.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"setattr", "delattr"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(node.func.attr.startswith("_configured_"))
        self.assertEqual(writes, [])
        self.assertNotIn("qualification_result.json", source)
        self.assertNotIn("qualification_decision.json", source)
        self.assertNotIn("torch.cuda.empty_cache", source)
        self.assertNotIn("torch.cuda.get_device_name", source)

    def test_static_plan_is_exact_and_does_not_reserve_root(self):
        self.assertFalse(plan_builder.SCIENTIFIC_ATTEMPT_ROOT.exists())
        with mock.patch.object(
            runner,
            "_validate_dino_inputs",
            return_value=runner.frozen_runner.expected_dino_v1(),
        ):
            plan, binding, scenes = runner._read_and_validate_plan(
                PLAN,
                expected_sha256=PLAN_SHA256,
                expected_byte_count=PLAN_BYTE_COUNT,
            )
        self.assertEqual(plan["attempt_id"], plan_builder.SCIENTIFIC_ATTEMPT_ID)
        self.assertEqual(binding["sha256"], PLAN_SHA256)
        self.assertEqual([row["scene_index"] for row in scenes], list(range(64)))
        self.assertFalse(plan_builder.SCIENTIFIC_ATTEMPT_ROOT.exists())

    def test_review_validation_is_before_reservation_in_parent(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(source, main)
        self.assertIsNotNone(main_source)
        self.assertLess(
            main_source.index("_read_and_validate_source_review("),
            main_source.index("execute_scientific("),
        )
        execute = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute_scientific"
        )
        execute_source = ast.get_source_segment(source, execute)
        self.assertIsNotNone(execute_source)
        self.assertLess(
            execute_source.index("_reserve_scientific("),
            execute_source.index("_run_graphics_preflight("),
        )

    def test_source_review_requires_exact_one_shot_decision(self):
        plan_binding = {
            "path": str(PLAN.resolve()),
            "sha256": PLAN_SHA256,
            "byte_count": PLAN_BYTE_COUNT,
        }
        decisions = {
            "source_review_passed": True,
            "exact_scientific_payload_identity_verified": True,
            "exact_eleven_key_environment_verified": True,
            "exact_64_fresh_scene_process_contract_verified": True,
            "frozen_three_arm_protocol_and_gates_verified": True,
            "minimal_qualification_decision_and_terminal_review_only": True,
            "qualification_payload_reuse_authorized": False,
            "fresh_scientific_root_verified": True,
            "exactly_one_scientific_invocation_eligible_under_user_authorization": True,
            "scientific_execution_authority_created_by_review": False,
            "retry_resume_refill_overwrite_repair_or_second_invocation_authorized": False,
        }
        sources = {
            "scientific_plan_builder": {"builder": True},
            "scientific_runner": {"runner": True},
            "scientific_runner_test": {"test": True},
        }
        dino = {"repository_path": "/dino"}
        prereg = {"path": "/prereg", "sha256": "2" * 64, "byte_count": 2}
        decision_binding = {"path": "/decision", "sha256": "3" * 64, "byte_count": 3}
        terminal_binding = {"path": "/terminal", "sha256": "4" * 64, "byte_count": 4}
        review = {
            "schema": runner.SOURCE_REVIEW_SCHEMA,
            "status": runner.SOURCE_REVIEW_STATUS,
            "protected_material_opened": False,
            "findings": [],
            "preregistration_binding": prereg,
            "scientific_plan_binding": plan_binding,
            "scientific_plan_builder_binding": {"builder": True},
            "scientific_runner_binding": {"runner": True},
            "focused_test_binding": {"test": True},
            "source_bindings": sources,
            "qualification_pass_decision_binding": decision_binding,
            "independent_qualification_terminal_review_binding": terminal_binding,
            "dino": dino,
            "decision": decisions,
        }
        fake_binding = {
            "path": "/review",
            "file_sha256": "5" * 64,
            "byte_count": 5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            review_path = Path(temporary) / "review.json"
            review_path.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(runner, "SOURCE_REVIEW", review_path),
                mock.patch.object(runner.pilot, "read_bound_json", return_value=(review, fake_binding)),
                mock.patch.object(runner, "_source_bindings", return_value=sources),
                mock.patch.object(runner, "_validate_dino_inputs", return_value=dino),
                mock.patch.object(runner, "_preregistration_binding", return_value=prereg),
                mock.patch.object(runner.plan_builder, "qualification_pass_decision_binding", return_value=decision_binding),
                mock.patch.object(runner.plan_builder, "independent_qualification_terminal_review_binding", return_value=terminal_binding),
            ):
                loaded, binding = runner._read_and_validate_source_review(
                    review_path,
                    expected_sha256="5" * 64,
                    expected_byte_count=5,
                    plan_binding=plan_binding,
                )
                self.assertEqual(loaded["decision"], decisions)
                self.assertEqual(binding["sha256"], "5" * 64)
                review["decision"] = {**decisions, "qualification_payload_reuse_authorized": True}
                with self.assertRaises(runner.CpuFlatScientificError):
                    runner._read_and_validate_source_review(
                        review_path,
                        expected_sha256="5" * 64,
                        expected_byte_count=5,
                        plan_binding=plan_binding,
                    )

    def test_child_environment_is_exact_eleven_key_cpu_render_contract(self):
        plan = {
            "execution_contract": {
                "environment": copy.deepcopy(plan_builder.CPU_EXECUTION_ENVIRONMENT)
            }
        }
        child = runner._child_environment(plan)
        self.assertEqual(child, plan_builder.CPU_EXECUTION_ENVIRONMENT)
        self.assertEqual(len(child), 11)
        self.assertEqual(child["GS_BACKEND"], "cpu")
        self.assertEqual(child["HOME"], "/home/andrewknowles")
        self.assertEqual(child["PATH"], "/usr/bin:/bin")
        self.assertEqual(child["MESA_VK_DEVICE_SELECT"], "1002:7551!")

    def test_join_emits_frozen_loader_shape(self):
        scenes = []
        results = []
        result_bindings = []
        workers = []
        releases = []
        for index in range(64):
            role = "train" if index < 32 else "eval"
            scene_id = f"scene-{index}"
            scenes.append({"scene_index": index, "role": role, "scene_id": scene_id})
            state_bindings = [
                {"path": f"state/{index}/{slot}", "file_sha256": "1" * 64, "byte_count": 1}
                for slot in range(4)
            ]
            metric = {
                "native_render_calls": 48,
                "rgb_render_calls": 48,
                "auxiliary_depth_render_calls": 48,
                "stored_rgb_frames": 48,
            }
            results.append(
                {
                    "scene_index": index,
                    "role": role,
                    "scene_id": scene_id,
                    "state_receipt_bindings": state_bindings,
                    "render_receipt_binding": {"path": f"render/{index}", "file_sha256": "2" * 64, "byte_count": 2},
                    "scene_metric": metric,
                    "scene_local_mesh_cache": {},
                    "runtime_versions": {"genesis": "0.3.14"},
                    "stored_rgb_bytes": 1,
                }
            )
            result_bindings.append({"path": f"result/{index}", "file_sha256": "3" * 64, "byte_count": 3})
            workers.append({"scene_index": index, "pid": 10_000 + index})
            releases.append({"status": "PASSED"})
        plan_binding = {"path": "/plan", "sha256": "4" * 64, "byte_count": 4}
        review_binding = {"path": "/review", "sha256": "5" * 64, "byte_count": 5}
        source_bindings = {"runner": {"path": "/runner", "sha256": "6" * 64, "byte_count": 6}}
        with tempfile.TemporaryDirectory() as temporary:
            collection = Path(temporary)
            with (
                mock.patch.object(plan_builder, "SCIENTIFIC_OUTPUT_ROOT", collection),
                mock.patch.object(plan_builder, "SCIENTIFIC_ATTEMPT_ROOT", collection.parent),
                mock.patch.object(runner.scene_core, "_rehash_relative_binding_v2"),
                mock.patch.object(runner.scene_core, "_validate_scene_local_mesh_bindings_v2", return_value={}),
                mock.patch.object(runner, "_source_bindings", return_value=source_bindings),
                mock.patch.object(runner, "_preregistration_binding", return_value={"path": "/prereg", "sha256": "7" * 64, "byte_count": 7}),
                mock.patch.object(runner, "_validate_dino_inputs", return_value={"repository_path": "/dino"}),
                mock.patch.object(runner.pilot, "write_json_exclusive") as write,
            ):
                authority, physics = runner._join_collection(
                    plan={"purpose": "science", "execution_contract": {}, "runtime_bindings": {}},
                    plan_binding=plan_binding,
                    plan_receipt_binding={"path": "authorized_plan.json", "file_sha256": "4" * 64, "byte_count": 4},
                    source_review_binding=review_binding,
                    reservation_binding={"path": "/reservation", "sha256": "8" * 64, "byte_count": 8},
                    scenes=scenes,
                    scene_results=results,
                    scene_result_bindings=result_bindings,
                    worker_receipts=workers,
                    release_barriers=releases,
                    collection_wall_seconds=1.0,
                )
        self.assertEqual(len(physics["state_receipt_bindings"]), 256)
        self.assertEqual(len(physics["render_receipt_bindings"]), 64)
        self.assertEqual(len(physics["scene_metrics"]), 64)
        self.assertEqual(physics["plan_binding"]["file_sha256"], "4" * 64)
        self.assertEqual(physics["authority_binding"], review_binding)
        self.assertEqual(physics["expected_counts"], runner.scene_core.EXPECTED_COUNTS)
        self.assertEqual(physics["caps"], runner.scene_core.EXPECTED_CAPS)
        self.assertEqual(authority["source_bindings"], source_bindings)
        write.assert_called_once()

    def test_model_stage_source_pins_cpu_and_checkpoint_before_eval(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        model = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_frozen_model_stage"
        )
        segment = ast.get_source_segment(source, model)
        self.assertIsNotNone(segment)
        self.assertIn('torch.device("cpu")', segment)
        self.assertIn("torch.cuda.is_available() is not False", segment)
        self.assertLess(segment.index('role="train"'), segment.index("load_dino_trunk_v1("))
        self.assertLess(segment.index("_save_checkpoint_exclusive("), segment.index("ledger.checkpoint()"))
        self.assertLess(segment.index("ledger.checkpoint()"), segment.index('ledger.load_receipts("eval")'))
        self.assertEqual(segment.count("benchmark.evaluate_checkpoint_v1("), 2)
        self.assertNotIn("execute_v1(", segment)
        self.assertNotIn("empty_cache", segment)

    def test_failure_after_reservation_always_writes_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            with mock.patch.object(plan_builder, "SCIENTIFIC_ATTEMPT_ROOT", attempt):
                runner._write_failure_terminal(RuntimeError("boom"))
            terminal = json.loads((attempt / "terminal.json").read_bytes())
        self.assertEqual(terminal["schema"], runner.TERMINAL_SCHEMA)
        self.assertEqual(terminal["status"], runner.FAIL_STATUS)
        self.assertIsNone(terminal["result_binding"])
        self.assertFalse(terminal["authorizes_retry_or_resume"])
        self.assertFalse(terminal["authorizes_navigation_claim"])


if __name__ == "__main__":
    unittest.main()
