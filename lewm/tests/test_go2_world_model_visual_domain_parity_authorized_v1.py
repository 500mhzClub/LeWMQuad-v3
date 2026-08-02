from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_world_model_visual_domain_parity_authority_v1 as authority
from scripts import build_go2_world_model_visual_domain_parity_plan_v1 as plan_builder
from scripts import evaluate_go2_world_model_visual_domain_parity_v1 as evaluator
from scripts import run_go2_world_model_visual_domain_parity_authorized_v1 as runner


REPO_ROOT = Path(__file__).resolve().parents[2]


def _function_ast(module_path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_parity_review_source_closure_only_names_existing_files():
    paths = authority.canonical_source_paths_v1()
    assert paths["external_supervisor"] == (
        "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py"
    )
    assert paths["parity_runtime_boundary_test"] == (
        "lewm/tests/test_go2_world_model_visual_domain_parity_authorized_v1.py"
    )
    assert paths["calibration_analyzer"] == (
        "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
    )
    assert all((REPO_ROOT / relative).is_file() for relative in paths.values())


def test_fresh_parity_supervisor_imports_are_inside_canonical_source_closure():
    probe = r'''
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(repo_root))
import scripts.run_go2_world_model_visual_domain_parity_authorized_v1  # noqa: F401,E402

loaded = set()
for module in tuple(sys.modules.values()):
    module_file = getattr(module, "__file__", None)
    if not module_file:
        continue
    candidate = Path(module_file)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        candidate = candidate.resolve(strict=True)
        relative = candidate.relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        continue
    parts = tuple(part.lower() for part in Path(relative).parts)
    if relative.startswith(".generated/") or any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out", "protected"}
        or part.startswith(("heldout_", "held_out_", "held-out-", "protected_"))
        for part in parts
    ):
        continue
    if relative.endswith(".py"):
        loaded.add(relative)
print(json.dumps(sorted(loaded)))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    closure = set(authority.canonical_source_paths_v1().values())
    assert loaded <= closure, sorted(loaded - closure)


def test_authority_caps_bound_rgb_total_pipeline_and_margin():
    fake_plan = {"expected_counts": dict(plan_builder.EXPECTED_COUNTS)}
    caps = authority._expected_caps(fake_plan, wall_seconds=900.0)
    assert caps["maximum_stored_rgb_bytes"] == 512 * 1024**2
    assert caps["maximum_parity_output_bytes"] == 1024**3
    assert caps["projected_pipeline_new_bytes"] == 3 * 1024**3
    assert caps["free_space_margin_bytes"] == 1024**3
    assert caps["required_preflight_free_bytes"] == 4 * 1024**3
    assert caps["wall_seconds"] == 900.0


def test_disk_preflight_fails_before_reservation_when_cap_not_available(
    tmp_path, monkeypatch
):
    caps = authority._expected_caps(
        {"expected_counts": dict(plan_builder.EXPECTED_COUNTS)}, wall_seconds=900.0
    )
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=(caps["required_preflight_free_bytes"] // 4096) - 1,
            f_frsize=4096,
        ),
    )
    with pytest.raises(
        runner.VisualDomainParitySupervisionError,
        match="insufficient pre-reservation disk",
    ):
        runner._disk_preflight(output_parent=tmp_path, authority={"caps": caps})


def test_disk_preflight_records_exact_four_gib_budget(tmp_path, monkeypatch):
    caps = authority._expected_caps(
        {"expected_counts": dict(plan_builder.EXPECTED_COUNTS)}, wall_seconds=900.0
    )
    monkeypatch.setattr(
        runner.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=2_000_000, f_frsize=4096),
    )
    observed = runner._disk_preflight(
        output_parent=tmp_path, authority={"caps": caps}
    )
    assert observed["passed"] is True
    assert observed["required_preflight_free_bytes"] == 4 * 1024**3


def test_atomic_root_creation_is_the_one_shot_consumption_boundary(
    tmp_path, monkeypatch
):
    development = tmp_path / ".generated" / "dev"
    development.mkdir(parents=True)
    consumed = development / "parity-attempt"
    consumed.mkdir()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(
        runner.VisualDomainParitySupervisionError,
        match="attempt root is not fresh",
    ):
        runner._fresh_attempt_root(consumed)
    authority_source = Path(authority.__file__).read_text(encoding="utf-8")
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"root_creation_consumes_attempt": True' in authority_source
    assert '"reservation_records_consumed_attempt": True' in authority_source
    assert "reservation_consumes_attempt" not in authority_source
    assert "reservation_consumes_attempt" not in runner_source


def test_chain_rehash_distinguishes_fresh_and_reserved_output_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development = tmp_path / ".generated" / "dev"
    development.mkdir(parents=True)
    output_root = development / "parity-attempt"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}\n", encoding="utf-8")
    authority_binding = pilot.file_binding(authority_path)
    observed: list[bool] = []

    monkeypatch.setattr(plan_builder, "DEVELOPMENT_ROOT", development)

    def validate_authority(
        value,
        *,
        plan,
        plan_binding,
        review,
        review_binding,
        require_fresh_output,
    ):
        del plan_binding, review, review_binding
        observed.append(require_fresh_output)
        plan_builder._validate_output_root(  # noqa: SLF001
            Path(plan["output_root"]), require_fresh=require_fresh_output
        )
        return value

    monkeypatch.setattr(authority, "validate_authority_v1", validate_authority)
    chain = {
        "plan": {"output_root": str(output_root)},
        "plan_binding": {},
        "authority": {},
        "authority_binding": authority_binding,
        "review": {},
        "review_binding": {},
    }

    runner._rehash_chain(**chain, require_fresh_output=True)  # noqa: SLF001
    assert not output_root.exists()
    output_root.mkdir()
    runner._rehash_chain(**chain, require_fresh_output=False)  # noqa: SLF001
    assert observed == [True, False]


def test_every_chain_rehash_declares_its_output_root_lifecycle_state() -> None:
    expected = {
        "_render_scene_worker": [False, False],
        "_terminal_revalidate": [False],
        "supervise_v1": [True, False],
    }
    for function_name, expected_flags in expected.items():
        function = _function_ast(Path(runner.__file__), function_name)
        calls = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_rehash_chain"
            ),
            key=lambda node: node.lineno,
        )
        observed_flags = []
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            flag = keywords.get("require_fresh_output")
            assert isinstance(flag, ast.Constant)
            assert type(flag.value) is bool
            observed_flags.append(flag.value)
        assert observed_flags == expected_flags


def test_scene_worker_calls_shared_rgb_helper_exactly_twice_per_pose():
    function = _function_ast(Path(runner.__file__), "_render_scene_worker")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_render_textured_v03_rgb_from_base_pose"
    ]
    assert len(calls) == 2
    source = ast.unparse(function)
    assert "camera.render" not in source
    assert "depth=True" not in source
    assert ".step(" not in source


def test_terminal_revalidation_occurs_only_after_result_binding_is_written():
    function = _function_ast(Path(runner.__file__), "supervise_v1")
    statements = list(ast.walk(function))
    result_assignments = [
        node
        for node in statements
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result_binding"
            for target in node.targets
        )
    ]
    terminal_calls = [
        node
        for node in statements
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_terminal_revalidate"
    ]
    assert len(result_assignments) == 1
    assert len(terminal_calls) == 1
    assert result_assignments[0].lineno < terminal_calls[0].lineno


def test_evaluator_uses_dedicated_parity_plan_not_generic_pilot_plan():
    function = _function_ast(Path(evaluator.__file__), "_validate_candidate_lineage")
    source = ast.unparse(function)
    assert "parity_plan.validate_plan_v1" in source
    assert "pilot.validate_plan" not in source
    assert "pilot.require_plan_bindings" not in source
    assert "sizing_calibration_textured_v03_v3" not in source


def test_candidate_renderer_provenance_names_scene_builder_and_pose_helper():
    assert evaluator.CANDIDATE_RENDERER == REPO_ROOT / "scripts/render_replay_v03.py"
    assert evaluator.CANDIDATE_CAMERA_POSE_HELPER == (
        REPO_ROOT / "lewm_genesis/lewm_genesis/render_replay.py"
    )
    assert evaluator.COMPARISON_CONTRACT["candidate_renderer_path"] == (
        "scripts/render_replay_v03.py"
    )
    assert evaluator.COMPARISON_CONTRACT[
        "candidate_camera_pose_helper_path"
    ] == "lewm_genesis/lewm_genesis/render_replay.py"


def test_preterminal_revalidation_rejects_recomputable_fail_result(monkeypatch):
    candidate_binding = {
        "path": "/tmp/candidate.json",
        "file_sha256": "1" * 64,
        "byte_count": 1,
    }
    result_binding = {
        "path": "/tmp/result.json",
        "file_sha256": "2" * 64,
        "byte_count": 1,
    }
    source_binding = {
        "path": "/tmp/source.json",
        "file_sha256": "3" * 64,
        "byte_count": 1,
    }
    fail_result = {
        "schema": evaluator.RESULT_SCHEMA,
        "status": evaluator.FAIL_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "measurements": {},
    }

    def read(binding, *, label):
        if label == "terminal candidate panel":
            return {}, candidate_binding
        if label == "terminal parity result":
            return fail_result, result_binding
        if label == "terminal historical source panel":
            return {}, source_binding
        raise AssertionError(label)

    monkeypatch.setattr(runner, "_rehash_chain", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_read_binding_document", read)
    monkeypatch.setattr(
        runner.evaluator, "evaluate_v1", lambda **_kwargs: dict(fail_result)
    )
    with pytest.raises(
        runner.VisualDomainParitySupervisionError,
        match="evaluator did not pass exactly",
    ):
        runner._terminal_revalidate(
            plan={"source_panel_binding": source_binding},
            plan_binding={},
            authority={},
            authority_binding={},
            review={},
            review_binding={},
            candidate_binding=candidate_binding,
            result_binding=result_binding,
            scene_result_bindings=[],
        )


def test_public_deep_validator_rejects_shallow_valid_fail_terminal(
    tmp_path, monkeypatch
):
    root = tmp_path / "attempt"
    root.mkdir()

    def write(path: Path, value: object) -> dict[str, object]:
        return pilot.write_json_exclusive(path, value)

    plan_binding = write(tmp_path / "plan.json", {})
    authority_binding = write(tmp_path / "authority.json", {})
    source_review_binding = write(tmp_path / "source-review.json", {})
    reservation_binding = write(root / "reservation.json", {})
    generation_binding = write(root / "generation_receipt.json", {})
    candidate_binding = write(root / "candidate_panel.json", {})
    fail_result = {
        "schema": evaluator.RESULT_SCHEMA,
        "status": evaluator.FAIL_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "measurements": {},
    }
    result_binding = write(root / "parity_result.json", fail_result)
    scene_bindings = [dict(reservation_binding) for _ in range(8)]
    plan = {"attempt_id": "attempt", "output_root": str(root)}
    authority_document = {
        "source_commit": "a" * 40,
        "caps": {
            "wall_seconds": 100.0,
            "maximum_parity_output_bytes": 1024,
        },
        "external_supervisor": {"terminal_reviewer": "reviewer"},
    }
    terminal = {
        "schema": runner.TERMINAL_SCHEMA,
        "status": runner.TERMINAL_SUCCESS_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "authorizes_retry_or_resume": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "attempt_id": "attempt",
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "reservation_binding": reservation_binding,
        "source_review_binding": source_review_binding,
        "source_commit": "a" * 40,
        "scene_result_bindings": scene_bindings,
        "generation_receipt_binding": generation_binding,
        "candidate_panel_binding": candidate_binding,
        "parity_result_binding": result_binding,
        "graphics_preflight": {},
        "disk_preflight": {},
        "wall_seconds": 10.0,
        "wall_ceiling_seconds": 100.0,
        "total_output_bytes_before_terminal": 123,
        "completed_at": "2026-08-02T12:00:00Z",
        "terminal_reviewer": "reviewer",
    }
    terminal_binding = write(root / "terminal.json", terminal)
    monkeypatch.setattr(
        runner,
        "load_and_validate_chain_v1",
        lambda **_kwargs: (
            plan,
            plan_binding,
            authority_document,
            authority_binding,
            {},
            source_review_binding,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_validate_reservation",
        lambda **_kwargs: reservation_binding,
    )
    monkeypatch.setattr(
        runner,
        "_terminal_revalidate",
        lambda **_kwargs: (
            {
                "candidate_panel_binding": candidate_binding,
                "parity_result_binding": result_binding,
                "generation_receipt_binding": generation_binding,
                "scene_result_bindings": scene_bindings,
            },
            123,
        ),
    )
    with pytest.raises(
        runner.VisualDomainParitySupervisionError,
        match="evaluator did not pass exactly",
    ):
        runner.validate_success_terminal_v1(
            terminal_binding=terminal_binding,
            expected_result_binding=result_binding,
        )


def _write_json_document(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = pilot.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return pilot.file_binding(path)


def _write_file(path: Path, raw: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return pilot.file_binding(path)


@pytest.fixture
def complete_parity_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a complete runtime tree with only three declared semantic stubs.

    The legacy source-panel lineage and the focused plan/authority semantic
    validators are synthetic here; their real validators have separate focused
    tests.  Inventory, bindings, render receipts, decoded RGB, generation,
    evaluation, reservation, terminal byte accounting, and deep terminal
    revalidation all execute their production implementations.
    """
    synthetic_repo = tmp_path / "synthetic-repo"
    source_rgb_root = synthetic_repo / ".generated/datagen_full/render_textured_v03"
    candidate_root = synthetic_repo / ".generated/dev"
    output_root = candidate_root / "complete-parity-attempt"
    output_root.mkdir(parents=True)
    source_rgb_root.mkdir(parents=True)

    source_paths = {
        "collector": synthetic_repo / "scripts/collector.py",
        "genesis_render_replay": synthetic_repo / "lewm_genesis/render_replay.py",
        "historical_textured_v03_renderer": synthetic_repo / "scripts/render_v03.py",
        "textures": synthetic_repo / "lewm_genesis/textures.py",
    }
    for name, path in source_paths.items():
        _write_file(path, f"# synthetic {name}\n".encode())
    runtime_binding = _write_file(
        synthetic_repo / "runtime/runtime.json", b"synthetic runtime\n"
    )
    corpus_binding = _write_file(
        synthetic_repo / "corpus/corpus.json", b"synthetic corpus\n"
    )

    monkeypatch.setattr(evaluator, "REPO_ROOT", synthetic_repo)
    monkeypatch.setattr(evaluator, "SOURCE_RGB_ROOT", source_rgb_root)
    monkeypatch.setattr(evaluator, "CANDIDATE_ROOT", candidate_root)
    monkeypatch.setattr(
        evaluator, "REFERENCE_RENDERER", source_paths["historical_textured_v03_renderer"]
    )
    monkeypatch.setattr(
        evaluator, "CANDIDATE_RENDERER", source_paths["historical_textured_v03_renderer"]
    )
    monkeypatch.setattr(evaluator, "CANDIDATE_COLLECTOR", source_paths["collector"])
    monkeypatch.setattr(
        evaluator,
        "CANDIDATE_CAMERA_POSE_HELPER",
        source_paths["genesis_render_replay"],
    )
    monkeypatch.setattr(
        evaluator, "REFERENCE_TEXTURE_SOURCE", source_paths["textures"]
    )

    texture_paths = {}
    texture_bindings = {}
    for index, category in enumerate(evaluator.TEXTURE_CATEGORIES):
        path = synthetic_repo / "assets/textures" / category / "selected.png"
        texture_paths[category] = path
        texture_bindings[category] = _write_file(
            path, f"synthetic texture {index}\n".encode()
        )
    monkeypatch.setattr(
        evaluator.reference_renderer,
        "select_scene_textures",
        lambda *, visual_seed, scene_id: {
            name: str(path) for name, path in texture_paths.items()
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_source_lineage",
        lambda panel: {
            "schema": evaluator.SOURCE_LINEAGE_SCHEMA,
            "synthetic_complete_tree": True,
        },
    )

    scenes = []
    source_rows = []
    source_arrays = {}
    texture_map = {}
    mesh_map = {}
    all_mesh_bindings = []
    for scene_index, family in enumerate(pilot.FAMILIES):
        scene_id = f"{scene_index:02d}_{family}"
        scene_source_root = synthetic_repo / "source-scenes" / scene_id
        manifest = {
            "scene_id": scene_id,
            "family": family,
            "split": "train",
            "visual_seed": scene_index + 1,
        }
        manifest_binding = _write_json_document(
            scene_source_root / "manifest.json", manifest
        )
        genesis_binding = _write_json_document(
            scene_source_root / "genesis_scene.json", {"scene_id": scene_id}
        )
        mesh_binding = _write_file(
            synthetic_repo / ".generated/box_meshes" / f"{scene_id}.obj",
            f"o {scene_id}\n".encode(),
        )
        all_mesh_bindings.append(mesh_binding)
        texture_map[scene_id] = copy.deepcopy(texture_bindings)
        mesh_map[scene_id] = [mesh_binding]
        poses = []
        for pose_index in range(evaluator.POSES_PER_SCENE):
            pair_id = f"{scene_id}/pose_{pose_index:02d}"
            camera_pose = {
                "position": [float(pose_index), float(scene_index), 0.4],
                "lookat": [float(pose_index) + 1.0, float(scene_index), 0.4],
                "up": [0.0, 0.0, 1.0],
            }
            rgb = np.empty((224, 224, 3), dtype=np.uint8)
            rgb[:, :, 0] = scene_index * 17
            rgb[:, :, 1] = pose_index * 31
            rgb[:, :, 2] = 101
            source_path = (
                source_rgb_root / scene_id / "rgb" / f"source_{pose_index:02d}.png"
            )
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_binding = runner._write_png_exclusive(source_path, rgb)  # noqa: SLF001
            raw_sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            source_arrays[pair_id] = rgb
            source_rows.append({
                "pair_id": pair_id,
                "scene_id": scene_id,
                "family": family,
                "pose_index": pose_index,
                "camera_pose_world": camera_pose,
                "scene_manifest_binding": manifest_binding,
                "producer_frame_identity": f"source-{scene_id}-{pose_index}",
                "rgb_binding": source_binding,
                "raw_rgb_sha256": raw_sha,
            })
            poses.append({
                "pair_id": pair_id,
                "pose_index": pose_index,
                "base_position_xyz_m": [float(pose_index), 0.0, 0.3],
                "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "historical_camera_pose_world": camera_pose,
            })
        scenes.append({
            "family": family,
            "scene_id": scene_id,
            "scene_manifest_binding": manifest_binding,
            "scene_genesis_binding": genesis_binding,
            "selected_texture_asset_bindings": copy.deepcopy(texture_bindings),
            "mesh_asset_bindings": [mesh_binding],
            "poses": poses,
        })

    source_rows.sort(key=lambda row: row["pair_id"])
    reference_renderer_binding = pilot.file_binding(
        source_paths["historical_textured_v03_renderer"]
    )
    texture_source_binding = pilot.file_binding(source_paths["textures"])
    source_panel = {
        "schema": evaluator.PANEL_SCHEMA,
        "domain": evaluator.SOURCE_DOMAIN,
        "rgb_root": str(source_rgb_root),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "producer_source_binding": reference_renderer_binding,
        "renderer_source_binding": reference_renderer_binding,
        "texture_source_binding": texture_source_binding,
        "selected_texture_asset_bindings_by_scene": texture_map,
        "mesh_asset_bindings_by_scene": mesh_map,
        "producer_lineage": {"synthetic": True},
        "rows": source_rows,
    }
    source_panel_path = synthetic_repo / "source_panel.json"
    source_panel_binding = _write_json_document(source_panel_path, source_panel)

    plan = {
        "schema": plan_builder.PLAN_SCHEMA,
        "status": plan_builder.PLAN_STATUS,
        "attempt_id": "complete-parity-attempt",
        "purpose": plan_builder.PURPOSE,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "output_root": str(output_root),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "comparison_contract": dict(plan_builder.COMPARISON_CONTRACT),
        "expected_counts": dict(plan_builder.EXPECTED_COUNTS),
        "runtime_bindings": {"runtime": runtime_binding},
        "execution_contract": {
            "environment": {},
            "graphics_preflight": {"eglinfo_expected_exit_code": 2},
        },
        "texture_asset_bindings": list(texture_bindings.values()),
        "source_panel_binding": source_panel_binding,
        "scene_corpus_manifest_bindings": [corpus_binding],
        "scenes": scenes,
        "mesh_asset_bindings": all_mesh_bindings,
    }
    plan_path = synthetic_repo / "plan.json"
    plan_binding = _write_json_document(plan_path, plan)
    review = {"schema": "synthetic-source-review", "status": "PASS"}
    review_path = synthetic_repo / "source_review.json"
    review_binding = _write_json_document(review_path, review)
    source_bindings = [
        {"name": name, "binding": pilot.file_binding(path)}
        for name, path in source_paths.items()
    ]
    authority_document = {
        "schema": authority.AUTHORITY_SCHEMA,
        "status": authority.AUTHORITY_STATUS,
        "plan_binding": plan_binding,
        "review_binding": review_binding,
        "source_commit": "a" * 40,
        "source_bindings": source_bindings,
        "caps": authority._expected_caps(plan, wall_seconds=100.0),
        "external_supervisor": {"terminal_reviewer": "synthetic-reviewer"},
    }
    authority_path = synthetic_repo / "authority.json"
    authority_binding = _write_json_document(authority_path, authority_document)

    def validate_plan(value, *, require_fresh_output=True):
        if (
            not isinstance(value, dict)
            or value.get("schema") != plan_builder.PLAN_SCHEMA
            or value.get("output_root") != str(output_root)
            or value.get("source_panel_binding") != source_panel_binding
        ):
            raise plan_builder.VisualDomainParityPlanError("synthetic plan changed")
        return copy.deepcopy(value)

    def validate_authority(
        value,
        *,
        plan,
        plan_binding: dict,
        review,
        review_binding: dict,
        require_fresh_output=True,
    ):
        if (
            not isinstance(value, dict)
            or value.get("schema") != authority.AUTHORITY_SCHEMA
            or value.get("plan_binding") != plan_binding
            or value.get("review_binding") != review_binding
            or value.get("source_bindings") != source_bindings
        ):
            raise authority.VisualDomainParityAuthorityError(
                "synthetic authority changed"
            )
        return copy.deepcopy(value)

    monkeypatch.setattr(plan_builder, "validate_plan_v1", validate_plan)
    monkeypatch.setattr(authority, "validate_authority_v1", validate_authority)

    reservation = {
        "schema": runner.RESERVATION_SCHEMA,
        "status": runner.RESERVATION_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "attempt_id": plan["attempt_id"],
        "output_root": plan["output_root"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "review_binding": review_binding,
        "source_commit": authority_document["source_commit"],
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry_or_resume_allowed": False,
        "reserved_at": "2026-08-02T12:00:00Z",
        "worker_capability_sha256": "b" * 64,
    }
    reservation_path = output_root / "reservation.json"
    reservation_binding = _write_json_document(reservation_path, reservation)
    scenes_root = output_root / "scenes"
    scenes_root.mkdir()

    candidate_rows = []
    generation_rows = []
    scene_result_bindings = []
    candidate_collector_binding = pilot.file_binding(source_paths["collector"])
    camera_helper_binding = pilot.file_binding(source_paths["genesis_render_replay"])
    for scene_index, scene in enumerate(scenes):
        scene_root = scenes_root / f"{scene_index:02d}_{scene['scene_id']}"
        rows_root = scene_root / "rows"
        rows_root.mkdir(parents=True)
        scene_rows = []
        for pose in scene["poses"]:
            pair_id = pose["pair_id"]
            pose_index = pose["pose_index"]
            pose_root = rows_root / f"pose_{pose_index:02d}"
            pose_root.mkdir()
            candidate_binding = runner._write_png_exclusive(  # noqa: SLF001
                pose_root / "candidate.png", source_arrays[pair_id]
            )
            duplicate_binding = runner._write_png_exclusive(  # noqa: SLF001
                pose_root / "duplicate.png", source_arrays[pair_id]
            )
            raw_sha = hashlib.sha256(
                np.ascontiguousarray(source_arrays[pair_id]).tobytes()
            ).hexdigest()
            candidate_identity = f"{pair_id}:candidate"
            duplicate_identity = f"{pair_id}:duplicate"
            common_receipt = {
                "schema": evaluator.RENDER_RECEIPT_SCHEMA,
                "status": evaluator.RENDER_RECEIPT_STATUS,
                "authority_granted_by_this_document": False,
                "scientific_claim_granted_by_this_document": False,
                "development_only": True,
                "protected_material_opened": False,
                "attempt_id": plan["attempt_id"],
                "pair_id": pair_id,
                "scene_id": scene["scene_id"],
                "family": scene["family"],
                "pose_index": pose_index,
                "base_position_xyz_m": pose["base_position_xyz_m"],
                "base_quaternion_wxyz": pose["base_quaternion_wxyz"],
                "historical_camera_pose_world": pose["historical_camera_pose_world"],
                "computed_camera_pose_world": pose["historical_camera_pose_world"],
                "scene_manifest_binding": scene["scene_manifest_binding"],
                "scene_genesis_binding": scene["scene_genesis_binding"],
                "source_panel_binding": source_panel_binding,
                "plan_binding": plan_binding,
                "authority_binding": authority_binding,
                "source_commit": authority_document["source_commit"],
                "source_bindings": source_bindings,
                "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
                "runtime_bindings": plan["runtime_bindings"],
                "execution_contract": plan["execution_contract"],
                "producer_source_binding": candidate_collector_binding,
                "renderer_source_binding": reference_renderer_binding,
                "camera_pose_helper_source_binding": camera_helper_binding,
                "texture_source_binding": texture_source_binding,
                "selected_texture_asset_bindings": scene[
                    "selected_texture_asset_bindings"
                ],
                "mesh_asset_bindings": scene["mesh_asset_bindings"],
                "rgb_render_call": {"rgb": True, "depth": False},
                "physics_steps": 0,
                "rgb_render_wall_seconds": 0.01,
            }
            receipt_bindings = {}
            for ordinal, identity, rgb_binding in (
                ("candidate", candidate_identity, candidate_binding),
                ("duplicate", duplicate_identity, duplicate_binding),
            ):
                receipt = {
                    **common_receipt,
                    "render_ordinal": ordinal,
                    "producer_frame_identity": identity,
                    "rgb_binding": rgb_binding,
                    "raw_rgb_sha256": raw_sha,
                }
                receipt_bindings[ordinal] = _write_json_document(
                    pose_root / f"{ordinal}_receipt.json", receipt
                )
            row = {
                "pair_id": pair_id,
                "scene_id": scene["scene_id"],
                "family": scene["family"],
                "pose_index": pose_index,
                "base_position_xyz_m": pose["base_position_xyz_m"],
                "base_quaternion_wxyz": pose["base_quaternion_wxyz"],
                "camera_pose_world": pose["historical_camera_pose_world"],
                "scene_manifest_binding": scene["scene_manifest_binding"],
                "scene_genesis_binding": scene["scene_genesis_binding"],
                "selected_texture_asset_bindings": scene[
                    "selected_texture_asset_bindings"
                ],
                "mesh_asset_bindings": scene["mesh_asset_bindings"],
                "candidate_producer_frame_identity": candidate_identity,
                "duplicate_producer_frame_identity": duplicate_identity,
                "candidate_rgb_binding": candidate_binding,
                "duplicate_rgb_binding": duplicate_binding,
                "candidate_raw_rgb_sha256": raw_sha,
                "duplicate_raw_rgb_sha256": raw_sha,
                "candidate_render_receipt_binding": receipt_bindings["candidate"],
                "duplicate_render_receipt_binding": receipt_bindings["duplicate"],
            }
            scene_rows.append(row)
            generation_rows.append(row)
            candidate_rows.append({
                "pair_id": pair_id,
                "scene_id": scene["scene_id"],
                "family": scene["family"],
                "pose_index": pose_index,
                "camera_pose_world": pose["historical_camera_pose_world"],
                "scene_manifest_binding": scene["scene_manifest_binding"],
                "producer_frame_identity": candidate_identity,
                "rgb_binding": candidate_binding,
                "raw_rgb_sha256": raw_sha,
                "duplicate_producer_frame_identity": duplicate_identity,
                "duplicate_rgb_binding": duplicate_binding,
                "duplicate_raw_rgb_sha256": raw_sha,
            })
        scene_result = {
            "schema": runner.SCENE_RESULT_SCHEMA,
            "status": runner.SCENE_RESULT_STATUS,
            "authority_granted_by_this_document": False,
            "scientific_claim_granted_by_this_document": False,
            "attempt_id": plan["attempt_id"],
            "scene_index": scene_index,
            "scene_id": scene["scene_id"],
            "family": scene["family"],
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "render_rows": scene_rows,
            "observed_counts": {
                "poses": 4,
                "candidate_rgb_frames": 4,
                "duplicate_rgb_frames": 4,
                "rgb_render_calls": 8,
                "auxiliary_depth_render_calls": 0,
                "physics_steps": 0,
            },
        }
        scene_result_bindings.append(
            _write_json_document(scene_root / "scene_result.json", scene_result)
        )

    generation_rows.sort(key=lambda row: row["pair_id"])
    candidate_rows.sort(key=lambda row: row["pair_id"])
    stored_rgb_bytes = sum(
        int(row[name]["byte_count"])
        for row in generation_rows
        for name in ("candidate_rgb_binding", "duplicate_rgb_binding")
    )
    generation = {
        "schema": evaluator.CANDIDATE_GENERATION_RECEIPT_SCHEMA,
        "status": evaluator.CANDIDATE_GENERATION_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "attempt_id": plan["attempt_id"],
        "output_root": plan["output_root"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "source_review_binding": review_binding,
        "source_commit": authority_document["source_commit"],
        "source_panel_binding": source_panel_binding,
        "render_contract": plan["render_contract"],
        "comparison_contract": plan["comparison_contract"],
        "expected_counts": plan["expected_counts"],
        "runtime_bindings": plan["runtime_bindings"],
        "execution_contract": plan["execution_contract"],
        "scene_corpus_manifest_bindings": plan["scene_corpus_manifest_bindings"],
        "texture_asset_bindings": plan["texture_asset_bindings"],
        "mesh_asset_bindings": plan["mesh_asset_bindings"],
        "producer_source_binding": candidate_collector_binding,
        "renderer_source_binding": reference_renderer_binding,
        "camera_pose_helper_source_binding": camera_helper_binding,
        "texture_source_binding": texture_source_binding,
        "selected_texture_asset_bindings_by_scene": texture_map,
        "mesh_asset_bindings_by_scene": mesh_map,
        "source_bindings": source_bindings,
        "render_rows": generation_rows,
        "observed_counts": {
            "scenes": 8,
            "poses": 32,
            "candidate_rgb_frames": 32,
            "duplicate_rgb_frames": 32,
            "rgb_render_calls": 64,
            "auxiliary_depth_render_calls": 0,
            "physics_steps": 0,
            "stored_rgb_bytes": stored_rgb_bytes,
        },
        "wall_seconds": 1.0,
    }
    generation_path = output_root / "generation_receipt.json"
    generation_binding = _write_json_document(generation_path, generation)
    candidate_panel = {
        "schema": evaluator.PANEL_SCHEMA,
        "domain": evaluator.CANDIDATE_DOMAIN,
        "rgb_root": str(scenes_root),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "producer_source_binding": candidate_collector_binding,
        "renderer_source_binding": reference_renderer_binding,
        "texture_source_binding": texture_source_binding,
        "selected_texture_asset_bindings_by_scene": texture_map,
        "mesh_asset_bindings_by_scene": mesh_map,
        "producer_lineage": {
            "schema": evaluator.CANDIDATE_LINEAGE_SCHEMA,
            "generation_receipt_binding": generation_binding,
        },
        "rows": candidate_rows,
    }
    candidate_panel_path = output_root / "candidate_panel.json"
    candidate_binding = _write_json_document(candidate_panel_path, candidate_panel)
    result = evaluator.evaluate_v1(
        source_panel=source_panel,
        source_panel_binding=source_panel_binding,
        candidate_panel=candidate_panel,
        candidate_panel_binding=candidate_binding,
    )
    assert result["status"] == evaluator.PASS_STATUS
    result_path = output_root / "parity_result.json"
    result_binding = _write_json_document(result_path, result)
    total_bytes = runner._validate_completed_inventory(  # noqa: SLF001
        plan, allow_terminal=False
    )
    caps = authority_document["caps"]
    terminal = {
        "schema": runner.TERMINAL_SCHEMA,
        "status": runner.TERMINAL_SUCCESS_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "authorizes_retry_or_resume": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "attempt_id": plan["attempt_id"],
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "reservation_binding": reservation_binding,
        "source_review_binding": review_binding,
        "source_commit": authority_document["source_commit"],
        "scene_result_bindings": scene_result_bindings,
        "generation_receipt_binding": generation_binding,
        "candidate_panel_binding": candidate_binding,
        "parity_result_binding": result_binding,
        "graphics_preflight": {
            "phase": "graphics_preflight",
            "status": "PASS",
            "environment": plan["execution_contract"]["environment"],
            "expectation": plan["execution_contract"]["graphics_preflight"],
            "vulkan_stdout_sha256": "c" * 64,
            "egl_stdout_sha256": "d" * 64,
            "egl_stderr_sha256": "e" * 64,
            "egl_exit_code": 2,
        },
        "disk_preflight": {
            "filesystem_path": str(output_root.parent.resolve(strict=True)),
            "available_bytes_before_reservation": caps[
                "required_preflight_free_bytes"
            ],
            "projected_pipeline_new_bytes": caps[
                "projected_pipeline_new_bytes"
            ],
            "free_space_margin_bytes": caps["free_space_margin_bytes"],
            "required_preflight_free_bytes": caps[
                "required_preflight_free_bytes"
            ],
            "maximum_parity_output_bytes": caps["maximum_parity_output_bytes"],
            "passed": True,
        },
        "wall_seconds": 1.0,
        "wall_ceiling_seconds": caps["wall_seconds"],
        "total_output_bytes_before_terminal": total_bytes,
        "completed_at": "2026-08-02T12:10:00Z",
        "terminal_reviewer": authority_document["external_supervisor"][
            "terminal_reviewer"
        ],
    }
    terminal_path = output_root / "terminal.json"
    terminal_binding = _write_json_document(terminal_path, terminal)
    parity_review = {
        "schema": pilot.TEXTURED_V03_PARITY_REVIEW_SCHEMA,
        "status": pilot.TEXTURED_V03_PARITY_REVIEW_PASS_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "result_binding": result_binding,
        "terminal_binding": terminal_binding,
        "reviewer": {
            "identity": "synthetic-independent-runtime-reviewer",
            "independence_basis": (
                "test fixture author is distinct from the runtime producer"
            ),
        },
        "reviewed_at": "2026-08-02T12:20:00Z",
        "checks": {
            name: True for name in pilot.TEXTURED_V03_PARITY_REVIEW_CHECKS
        },
        "remaining_findings": [],
    }
    parity_review_path = synthetic_repo / "parity_independent_review.json"
    parity_review_binding = _write_json_document(
        parity_review_path, parity_review
    )
    return {
        "terminal_binding": terminal_binding,
        "result_binding": result_binding,
        "parity_review_binding": parity_review_binding,
        "synthetic_semantic_stubs": (
            "source_panel_lineage",
            "plan_validation",
            "authority_validation",
        ),
        "paths": {
            "terminal": terminal_path,
            "result": result_path,
            "source_review": review_path,
            "parity_review": parity_review_path,
            "plan": plan_path,
            "authority": authority_path,
            "reservation": reservation_path,
            "generation": generation_path,
            "candidate_panel": candidate_panel_path,
            "source_panel": source_panel_path,
            "scene_results": [
                output_root
                / "scenes"
                / f"{index:02d}_{scene['scene_id']}"
                / "scene_result.json"
                for index, scene in enumerate(scenes)
            ],
            "render_receipt": (
                output_root
                / "scenes"
                / f"00_{scenes[0]['scene_id']}"
                / "rows/pose_00/candidate_receipt.json"
            ),
            "rgb_leaf": (
                output_root
                / "scenes"
                / f"00_{scenes[0]['scene_id']}"
                / "rows/pose_00/candidate.png"
            ),
            "missing_leaf": (
                output_root
                / "scenes"
                / f"00_{scenes[0]['scene_id']}"
                / "rows/pose_00/duplicate.png"
            ),
            "unexpected": output_root / "unexpected.json",
        },
    }


def test_complete_parity_tree_passes_deep_revalidation_and_rejects_mutations(
    complete_parity_tree,
) -> None:
    terminal_binding = complete_parity_tree["terminal_binding"]
    result_binding = complete_parity_tree["result_binding"]
    validated = runner.validate_success_terminal_v1(
        terminal_binding=terminal_binding,
        expected_result_binding=result_binding,
    )
    assert validated["result_binding"] == result_binding

    paths = complete_parity_tree["paths"]
    mutated_files = [
        paths["result"],
        paths["source_review"],
        paths["plan"],
        paths["authority"],
        paths["reservation"],
        paths["generation"],
        paths["candidate_panel"],
        paths["source_panel"],
        paths["render_receipt"],
        paths["rgb_leaf"],
        *paths["scene_results"],
    ]
    for path in mutated_files:
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            with pytest.raises(
                (runner.VisualDomainParitySupervisionError, evaluator.VisualDomainParityError)
            ):
                runner.validate_success_terminal_v1(
                    terminal_binding=terminal_binding,
                    expected_result_binding=result_binding,
                )
        finally:
            path.write_bytes(original)

    paths["unexpected"].write_bytes(b"{}\n")
    try:
        with pytest.raises(
            runner.VisualDomainParitySupervisionError, match="inventory"
        ):
            runner.validate_success_terminal_v1(
                terminal_binding=terminal_binding,
                expected_result_binding=result_binding,
            )
    finally:
        paths["unexpected"].unlink()

    missing = paths["missing_leaf"]
    displaced = missing.with_name("duplicate.missing")
    missing.rename(displaced)
    try:
        with pytest.raises(
            (runner.VisualDomainParitySupervisionError, evaluator.VisualDomainParityError)
        ):
            runner.validate_success_terminal_v1(
                terminal_binding=terminal_binding,
                expected_result_binding=result_binding,
            )
    finally:
        displaced.rename(missing)

    terminal_path = paths["terminal"]
    terminal_raw = terminal_path.read_bytes()
    terminal = json.loads(terminal_raw)
    terminal["total_output_bytes_before_terminal"] += 1
    wrong_terminal_binding = _write_json_document(
        terminal_path.with_name("terminal-wrong-byte-count.json"), terminal
    )
    wrong_path = Path(str(wrong_terminal_binding["path"]))
    replacement_raw = wrong_path.read_bytes()
    wrong_path.unlink()
    terminal_path.write_bytes(replacement_raw)
    rebound_terminal = pilot.file_binding(terminal_path)
    try:
        with pytest.raises(
            runner.VisualDomainParitySupervisionError,
            match="recomputation changed",
        ):
            runner.validate_success_terminal_v1(
                terminal_binding=rebound_terminal,
                expected_result_binding=result_binding,
            )
    finally:
        terminal_path.write_bytes(terminal_raw)


def test_public_textured_parity_prerequisites_reopen_complete_runtime_triple(
    complete_parity_tree,
    tmp_path: Path,
) -> None:
    result_binding = complete_parity_tree["result_binding"]
    terminal_binding = complete_parity_tree["terminal_binding"]
    review_binding = complete_parity_tree["parity_review_binding"]
    assert complete_parity_tree["synthetic_semantic_stubs"] == (
        "source_panel_lineage",
        "plan_validation",
        "authority_validation",
    )

    validated = pilot.validate_textured_v03_parity_prerequisites(
        result_binding=result_binding,
        terminal_binding=terminal_binding,
        review_binding=review_binding,
    )
    assert validated == {
        "result_binding": result_binding,
        "terminal_binding": terminal_binding,
        "review_binding": review_binding,
    }

    paths = complete_parity_tree["paths"]
    mutations = []
    changed_result = json.loads(paths["result"].read_bytes())
    changed_result["status"] = "FAIL"
    mutations.append((
        _write_json_document(tmp_path / "changed-result.json", changed_result),
        terminal_binding,
        review_binding,
    ))
    changed_terminal = json.loads(paths["terminal"].read_bytes())
    changed_terminal["root_creation_consumes_attempt"] = False
    mutations.append((
        result_binding,
        _write_json_document(tmp_path / "changed-terminal.json", changed_terminal),
        review_binding,
    ))
    changed_review = json.loads(paths["parity_review"].read_bytes())
    changed_review["checks"][pilot.TEXTURED_V03_PARITY_REVIEW_CHECKS[0]] = False
    mutations.append((
        result_binding,
        terminal_binding,
        _write_json_document(tmp_path / "changed-review.json", changed_review),
    ))
    for selected_result, selected_terminal, selected_review in mutations:
        with pytest.raises(pilot.PilotContractError):
            pilot.validate_textured_v03_parity_prerequisites(
                result_binding=selected_result,
                terminal_binding=selected_terminal,
                review_binding=selected_review,
            )

    swapped_result = _write_json_document(
        tmp_path / "swapped-result.json", json.loads(paths["result"].read_bytes())
    )
    swapped_terminal = _write_json_document(
        tmp_path / "swapped-terminal.json",
        json.loads(paths["terminal"].read_bytes()),
    )
    swapped_review_document = json.loads(paths["parity_review"].read_bytes())
    swapped_review_document["terminal_binding"] = swapped_terminal
    swapped_review = _write_json_document(
        tmp_path / "swapped-review.json", swapped_review_document
    )
    for selected_result, selected_terminal, selected_review in (
        (swapped_result, terminal_binding, review_binding),
        (result_binding, swapped_terminal, review_binding),
        (result_binding, terminal_binding, swapped_review),
    ):
        with pytest.raises(pilot.PilotContractError):
            pilot.validate_textured_v03_parity_prerequisites(
                result_binding=selected_result,
                terminal_binding=selected_terminal,
                review_binding=selected_review,
            )

    for missing in ("result", "terminal", "review"):
        triple = {
            "result_binding": result_binding,
            "terminal_binding": terminal_binding,
            "review_binding": review_binding,
        }
        triple[f"{missing}_binding"] = None
        with pytest.raises(pilot.PilotContractError):
            pilot.validate_textured_v03_parity_prerequisites(**triple)
