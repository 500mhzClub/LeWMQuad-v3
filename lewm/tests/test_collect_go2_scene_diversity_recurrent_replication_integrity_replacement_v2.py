from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import (
    collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2
    as collector,
)


def _binding(path: str) -> dict[str, object]:
    return {"path": path, "file_sha256": "a" * 64, "byte_count": 1}


def _plan() -> dict[str, object]:
    states: list[dict[str, object]] = []
    for scene_index in range(collector.SCENE_COUNT):
        role = "train" if scene_index < collector.TRAIN_SCENE_COUNT else "eval"
        scene_id = f"{role}-scene-{scene_index:03d}"
        for state_index in range(collector.STATES_PER_SCENE):
            states.append(
                {
                    "role": role,
                    "scene_id": scene_id,
                    "state_id": f"{role}-state-{scene_index:03d}-{state_index}",
                }
            )
    return {
        "states": states,
        "execution_contract": {
            "seed": 20260802,
            "python_invocation_path": str(Path(sys.executable).resolve()),
        },
    }


def _barrier(
    *, scene: dict[str, object], worker_pid: int, baseline: int
) -> dict[str, object]:
    return {
        "scene_index": scene["scene_index"],
        "role": scene["role"],
        "scene_id": scene["scene_id"],
        "after_worker_pid": worker_pid,
        "status": "PASSED",
        "read_only": True,
        "counter_path": "/sys/mock/mem_info_vram_used",
        "baseline_used_bytes": baseline,
        "release_margin_bytes": collector.VRAM_RELEASE_MARGIN_BYTES,
        "release_ceiling_bytes": baseline + collector.VRAM_RELEASE_MARGIN_BYTES,
        "absolute_vram_ceiling_bytes": collector.EXPECTED_CAPS[
            "selected_device_vram_byte_ceiling"
        ],
        "required_consecutive_samples": collector.VRAM_RELEASE_CONSECUTIVE_SAMPLES,
        "sample_interval_seconds": collector.VRAM_RELEASE_POLL_SECONDS,
        "sample_count": 3,
        "minimum_used_bytes": baseline,
        "maximum_used_bytes": baseline + 1,
        "final_used_bytes": baseline,
        "final_consecutive_samples": 3,
        "elapsed_seconds": 0.1,
    }


def _valid_result() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    plan = _plan()
    scenes = collector._scene_slices_v2(plan)  # noqa: SLF001
    scene_results: list[dict[str, object]] = []
    scene_bindings: list[dict[str, object]] = []
    worker_receipts: list[dict[str, object]] = []
    barriers: list[dict[str, object]] = []
    for scene in scenes:
        index = int(scene["scene_index"])
        role = str(scene["role"])
        scene_id = str(scene["scene_id"])
        pid = 1000 + index
        baseline = 1_000_000_000 + index
        expected = collector._scene_expected_counts_v2(role)  # noqa: SLF001
        state_bindings = [
            _binding(
                f"scenes/{role}/{scene_id}/state_receipts/{state['state_id']}.json"
            )
            for state in scene["states"]
        ]
        render_binding = _binding(
            f"scenes/{role}/{scene_id}/live_render_receipt.json"
        )
        scene_results.append(
            {
                "scene_index": index,
                "role": role,
                "scene_id": scene_id,
                "execution_seed": 20260802,
                "scene_local_mesh_cache": {
                    "path": f"scenes/{role}/{scene_id}/derived_meshes",
                    "cross_scene_reuse_authorized": False,
                    "mesh_count": 1,
                    "bindings_identity_sha256": "d" * 64,
                },
                "expected_counts": expected,
                "observed_counts": copy.deepcopy(expected),
                "state_receipt_bindings": state_bindings,
                "render_receipt_binding": render_binding,
                "scene_metric": {
                    "role": role,
                    "scene_id": scene_id,
                    "states": collector.STATES_PER_SCENE,
                },
            }
        )
        scene_bindings.append(_binding(f"scene_results/{index:03d}.json"))
        worker_receipts.append(
            {
                "pid": pid,
                "parent_pid": 999,
                "sys_executable": str(Path(sys.executable).resolve()),
                "parent_process_group_id": 777,
                "child_process_group_id": 777,
                "prelaunch_baseline_used_bytes": baseline,
            }
        )
        barriers.append(_barrier(scene=scene, worker_pid=pid, baseline=baseline))
    evidence = collector._build_scene_process_evidence_v2(  # noqa: SLF001
        plan=plan,
        scene_results=scene_results,
        scene_result_bindings=scene_bindings,
        worker_receipts=worker_receipts,
        release_barriers=barriers,
    )
    authority_binding = {
        "path": "/authority",
        "sha256": "b" * 64,
        "byte_count": 1,
    }
    plan_binding = {
        "path": "/plan",
        "file_sha256": "c" * 64,
        "byte_count": 1,
    }
    result = {
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
        "state_receipt_bindings": [
            row
            for scene_result in scene_results
            for row in scene_result["state_receipt_bindings"]
        ],
        "render_receipt_bindings": [
            scene_result["render_receipt_binding"] for scene_result in scene_results
        ],
        "scene_metrics": [
            scene_result["scene_metric"] for scene_result in scene_results
        ],
        "scene_result_bindings": scene_bindings,
        "scene_process_evidence": evidence,
    }
    return result, plan, authority_binding


def test_plan_is_exactly_64_four_state_scenes_in_frozen_role_order() -> None:
    scenes = collector._scene_slices_v2(_plan())  # noqa: SLF001
    assert len(scenes) == 64
    assert [row["scene_index"] for row in scenes] == list(range(64))
    assert [row["role"] for row in scenes] == ["train"] * 32 + ["eval"] * 32
    assert all(len(row["states"]) == 4 for row in scenes)


def test_every_worker_initializes_from_exact_full_plan_first_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    binding = pilot.write_json_exclusive(
        manifest, {"physics_seed": collector.PLAN_FIRST_PHYSICS_SEED}
    )
    plan = {
        "states": [
            {
                "state_id": "scene-diversity-train-large_enclosed_maze-state-0",
                "scene_id": "large_enclosed_maze_8a6599d5327d",
                "scene_manifest_binding": binding,
            }
        ],
        "execution_contract": {"backend": "vulkan"},
    }
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector,
        "_initialize_genesis_v2",
        lambda *, backend, seed: calls.append((backend, seed)),
    )

    rows = [
        collector._initialize_from_plan_first_scene_v2(plan=plan)  # noqa: SLF001
        for _ in range(collector.SCENE_COUNT)
    ]

    assert calls == [
        ("vulkan", collector.PLAN_FIRST_PHYSICS_SEED)
    ] * collector.SCENE_COUNT
    assert {row["effective_genesis_seed"] for row in rows} == {315_871_188}


def test_process_reset_equivalence_audit_binds_deterministic_call_graph() -> None:
    audit = collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
    assert audit["status"] == "PASS_NO_OUTCOME_AFFECTING_POST_INIT_RANDOM_DRAW"
    assert audit["exact_clone_initialization_overwrites_dynamic_state"] is True
    assert audit["requested_action_blocks_are_plan_bound"] is True
    assert (
        audit["collector_scheduler_policy_and_assignment_route_not_consumed"]
        is True
    )
    assert "collector_scheduler_not_invoked" not in audit
    assert audit["texture_rng_is_scene_keyed_local_rng"] is True
    assert (
        audit["reachable_global_rng_draw_affects_only_unobserved_collision_debug_rgba"]
        is True
    )


def test_mesh_cache_is_role_and_scene_local(tmp_path: Path) -> None:
    calls: list[tuple[tuple[float, ...], float, Path]] = []

    def cached(
        size: tuple[float, ...], *, tiles_per_m: float, cache_dir: Path
    ) -> str:
        calls.append((size, tiles_per_m, Path(cache_dir)))
        return str(Path(cache_dir) / "mesh.obj")

    render_namespace: dict[str, object] = {"cached_box_obj": cached}
    exec("def build_scene(*args, **kwargs): pass", render_namespace)
    runtime: dict[str, object] = {
        "cached_box_obj": cached,
        "build_textured_v03_scene": render_namespace["build_scene"],
    }
    scene_root = tmp_path / "collection/scenes/eval/scene-063"
    cache_root = collector._install_scene_local_mesh_cache_v2(  # noqa: SLF001
        runtime, scene_root=scene_root
    )
    assert runtime["cached_box_obj"]((1.0, 2.0, 3.0)) == str(
        cache_root / "mesh.obj"
    )
    render_cached = render_namespace["build_scene"].__globals__["cached_box_obj"]
    assert render_cached((4.0, 5.0, 6.0)) == str(cache_root / "mesh.obj")
    assert calls == [
        ((1.0, 2.0, 3.0), 0.7, cache_root),
        ((4.0, 5.0, 6.0), 0.7, cache_root),
    ]
    assert cache_root == scene_root / "derived_meshes"


def test_worker_observes_live_process_group_and_fresh_prelaunch_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    counter = tmp_path / "vram_used"
    counter.write_text("123456")

    class Process:
        pid = 1234

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(collector.subprocess, "Popen", Process)
    monkeypatch.setattr(collector.os, "getpgrp", lambda: 88)
    monkeypatch.setattr(collector.os, "getpgid", lambda pid: 88 if pid == 1234 else -1)
    receipt = collector._run_worker_process_v2(  # noqa: SLF001
        [sys.executable, "/collector.py", "--worker-scene-index", "0"],
        scene_index=0,
        role="train",
        scene_id="scene-0",
        used_path=counter,
        ceiling_bytes=collector.EXPECTED_CAPS[
            "selected_device_vram_byte_ceiling"
        ],
    )
    assert captured["argv"][0] == sys.executable
    assert captured["kwargs"] == {"cwd": collector.REPO_ROOT}
    assert receipt["prelaunch_baseline_used_bytes"] == 123456
    assert receipt["parent_process_group_id"] == 88
    assert receipt["child_process_group_id"] == 88
    assert receipt["process_group_equality_observed"] is True


def test_worker_rejects_noninherited_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = tmp_path / "vram_used"
    counter.write_text("0")

    class Process:
        pid = 1234

        def __init__(self, _argv: list[str], **_kwargs: object) -> None:
            pass

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(collector.subprocess, "Popen", Process)
    monkeypatch.setattr(collector.os, "getpgrp", lambda: 88)
    monkeypatch.setattr(collector.os, "getpgid", lambda _pid: 89)
    receipt = collector._run_worker_process_v2(  # noqa: SLF001
        [sys.executable, "/collector.py", "--worker-scene-index", "0"],
        scene_index=0,
        role="train",
        scene_id="scene-0",
        used_path=counter,
        ceiling_bytes=collector.EXPECTED_CAPS[
            "selected_device_vram_byte_ceiling"
        ],
    )
    assert receipt["process_group_equality_observed"] is False
    with pytest.raises(collector.SceneProcessCollectionError, match="completion"):
        collector._validate_completed_worker_v2(receipt)  # noqa: SLF001


def test_release_barrier_requires_three_50ms_bounded_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = 1_000_000_000
    release = baseline + collector.VRAM_RELEASE_MARGIN_BYTES
    counter = tmp_path / "counter"
    counter.write_text(str(baseline))
    values = iter([release + 1, release, release - 1, release])
    monkeypatch.setattr(
        collector, "_read_vram_counter_v2", lambda _path: next(values)
    )
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    evidence = collector._wait_for_vram_release_v2(  # noqa: SLF001
        counter,
        baseline_used_bytes=baseline,
        ceiling_bytes=collector.EXPECTED_CAPS[
            "selected_device_vram_byte_ceiling"
        ],
    )
    assert evidence["status"] == "PASSED"
    assert evidence["sample_interval_seconds"] == 0.05
    assert evidence["sample_count"] == 4
    assert evidence["final_consecutive_samples"] == 3


def test_scene_process_evidence_accepts_exact_64_workers_and_barriers() -> None:
    result, plan, authority_binding = _valid_result()
    assert collector.validate_scene_process_evidence_v2(
        result,
        authority_binding=authority_binding,
        plan_binding=result["plan_binding"],
        plan=plan,
    ) == {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barriers_exact": True,
        "join_exact": True,
    }


def test_final_join_mesh_rehash_rejects_tamper_and_unbound_cache_entry(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "scenes/train/scene/derived_meshes"
    cache.mkdir(parents=True)
    mesh = cache / "box.obj"
    mesh.write_text("v 0 0 0\n")
    metric = {"derived_mesh_bindings": [pilot.file_binding(mesh)]}
    summary = collector._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
        metric, cache_root=cache, collection_root=tmp_path
    )
    assert summary["mesh_count"] == 1
    mesh.write_text("v 1 0 0\n")
    with pytest.raises(collector.SceneProcessCollectionError, match="derived mesh"):
        collector._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
            metric, cache_root=cache, collection_root=tmp_path
        )
    metric = {"derived_mesh_bindings": [pilot.file_binding(mesh)]}
    (cache / "unbound.obj").write_text("v 0 1 0\n")
    with pytest.raises(collector.SceneProcessCollectionError, match="closure"):
        collector._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
            metric, cache_root=cache, collection_root=tmp_path
        )


def _filesystem_closure_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    root = tmp_path / "collection"
    (root / "scene_results").mkdir(parents=True)
    plan = _plan()
    scenes = collector._scene_slices_v2(plan)  # noqa: SLF001
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "sha256": "b" * 64,
        "byte_count": 1,
    }
    plan_binding = {
        "path": str(tmp_path / "plan.json"),
        "file_sha256": "c" * 64,
        "byte_count": 1,
    }
    scene_results: list[dict[str, object]] = []
    scene_bindings: list[dict[str, object]] = []
    workers: list[dict[str, object]] = []
    barriers: list[dict[str, object]] = []
    for scene in scenes:
        index = int(scene["scene_index"])
        role = str(scene["role"])
        scene_id = str(scene["scene_id"])
        input_root = tmp_path / "inputs" / f"{index:03d}"
        manifest_binding = pilot.write_json_exclusive(
            input_root / "manifest.json", {"scene": scene_id}
        )
        genesis_binding = pilot.write_json_exclusive(
            input_root / "scene.genesis.json", {"scene": scene_id}
        )
        for state in scene["states"]:
            state["scene_manifest_binding"] = manifest_binding
            state["scene_genesis_binding"] = genesis_binding
        scene_root = root / "scenes" / role / scene_id
        cache = scene_root / "derived_meshes"
        cache.mkdir(parents=True)
        mesh = cache / "box.obj"
        mesh.write_text(f"# {scene_id}\n")
        metric = {
            "role": role,
            "scene_id": scene_id,
            "states": collector.STATES_PER_SCENE,
            "derived_mesh_bindings": [pilot.file_binding(mesh)],
        }
        cache_summary = collector._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
            metric, cache_root=cache, collection_root=root
        )
        state_bindings = []
        for state in scene["states"]:
            receipt = scene_root / "state_receipts" / f"{state['state_id']}.json"
            binding = pilot.write_json_exclusive(receipt, {"state": state["state_id"]})
            state_bindings.append(
                {
                    "path": Path(binding["path"]).relative_to(root).as_posix(),
                    "file_sha256": binding["file_sha256"],
                    "byte_count": binding["byte_count"],
                }
            )
        render = pilot.write_json_exclusive(
            scene_root / "live_render_receipt.json", {"scene": scene_id}
        )
        render_binding = {
            "path": Path(render["path"]).relative_to(root).as_posix(),
            "file_sha256": render["file_sha256"],
            "byte_count": render["byte_count"],
        }
        expected = collector._scene_expected_counts_v2(role)  # noqa: SLF001
        scene_result = {
            "schema": collector.SCENE_RESULT_SCHEMA,
            "status": collector.SCENE_RESULT_STATUS,
            "attempt_id": collector.ATTEMPT_ID,
            "scene_index": index,
            "role": role,
            "scene_id": scene_id,
            "execution_seed": plan["execution_contract"]["seed"],
            "process_reset_equivalence_audit": copy.deepcopy(
                collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
            ),
            "scene_local_mesh_cache": cache_summary,
            "plan_binding": plan_binding,
            "authority_binding": authority_binding,
            "expected_counts": expected,
            "observed_counts": copy.deepcopy(expected),
            "ordered_state_ids": [str(row["state_id"]) for row in scene["states"]],
            "state_receipt_bindings": state_bindings,
            "render_receipt_binding": render_binding,
            "scene_metric": metric,
            "failure": None,
            "authorizes_retry_or_resume": False,
            "allows_refill": False,
            "allows_overwrite": False,
            "allows_adaptive_batching": False,
        }
        absolute_scene_result = pilot.write_json_exclusive(
            root / "scene_results" / f"{index:03d}.json", scene_result
        )
        scene_binding = {
            "path": f"scene_results/{index:03d}.json",
            "file_sha256": absolute_scene_result["file_sha256"],
            "byte_count": absolute_scene_result["byte_count"],
        }
        scene_results.append(scene_result)
        scene_bindings.append(scene_binding)
        baseline = 1_000_000_000 + index
        workers.append(
            {
                "pid": 1000 + index,
                "parent_pid": 999,
                "sys_executable": str(Path(sys.executable).resolve()),
                "parent_process_group_id": 777,
                "child_process_group_id": 777,
                "prelaunch_baseline_used_bytes": baseline,
            }
        )
        barriers.append(
            _barrier(scene=scene, worker_pid=1000 + index, baseline=baseline)
        )
    evidence = collector._build_scene_process_evidence_v2(  # noqa: SLF001
        plan=plan,
        scene_results=scene_results,
        scene_result_bindings=scene_bindings,
        worker_receipts=workers,
        release_barriers=barriers,
    )
    result = {
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
        "state_receipt_bindings": [
            binding
            for scene_result in scene_results
            for binding in scene_result["state_receipt_bindings"]
        ],
        "render_receipt_bindings": [
            scene_result["render_receipt_binding"] for scene_result in scene_results
        ],
        "scene_metrics": [scene_result["scene_metric"] for scene_result in scene_results],
        "scene_result_bindings": scene_bindings,
        "scene_process_evidence": evidence,
    }
    return result, plan, authority_binding, root


def test_pre_dino_closure_rehashes_all_outputs_meshes_and_128_scene_inputs(
    tmp_path: Path,
) -> None:
    result, plan, authority_binding, root = _filesystem_closure_fixture(tmp_path)
    report = collector.validate_scene_process_closure_v2(
        result,
        collection_root=root,
        authority_binding=authority_binding,
        plan_binding=result["plan_binding"],
        plan=plan,
    )
    assert report == {
        "validated": True,
        "evidence_validated": True,
        "closure_rehashed": True,
        "scene_results_rehashed": True,
        "state_receipts_rehashed": True,
        "render_receipts_rehashed": True,
        "derived_meshes_rehashed": True,
        "plan_scene_input_bindings_rehashed": True,
    }
    first_mesh = Path(result["scene_metrics"][0]["derived_mesh_bindings"][0]["path"])
    first_mesh.write_text("tampered\n")
    with pytest.raises(collector.SceneProcessCollectionError):
        collector.validate_scene_process_closure_v2(
            result,
            collection_root=root,
            authority_binding=authority_binding,
            plan_binding=result["plan_binding"],
            plan=plan,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "worker_order",
        "worker_pid_reuse",
        "worker_seed",
        "pgid_unobserved",
        "barrier_baseline_not_worker_baseline",
        "barrier_failed",
        "final_barrier_not_before_join",
        "scene_result_reordered",
        "state_reordered",
    ),
)
def test_scene_process_evidence_rejects_adversarial_drift(mutation: str) -> None:
    result, plan, authority_binding = _valid_result()
    evidence = result["scene_process_evidence"]
    if mutation == "worker_order":
        evidence["workers"][0], evidence["workers"][1] = (
            evidence["workers"][1],
            evidence["workers"][0],
        )
    elif mutation == "worker_pid_reuse":
        evidence["workers"][1]["pid"] = evidence["workers"][0]["pid"]
        evidence["release_barriers"][1]["after_worker_pid"] = evidence["workers"][0]["pid"]
    elif mutation == "worker_seed":
        evidence["workers"][1]["full_genesis_seed"] += 1
    elif mutation == "pgid_unobserved":
        evidence["workers"][1]["process_group_equality_observed"] = False
    elif mutation == "barrier_baseline_not_worker_baseline":
        evidence["release_barriers"][1]["baseline_used_bytes"] += 1
        evidence["release_barriers"][1]["release_ceiling_bytes"] += 1
    elif mutation == "barrier_failed":
        evidence["release_barriers"][1]["status"] = "FAILED"
    elif mutation == "final_barrier_not_before_join":
        evidence["sequential_launch"]["final_release_barrier_passed_before_join"] = False
    elif mutation == "scene_result_reordered":
        result["scene_result_bindings"][0], result["scene_result_bindings"][1] = (
            result["scene_result_bindings"][1],
            result["scene_result_bindings"][0],
        )
    elif mutation == "state_reordered":
        result["state_receipt_bindings"][0], result["state_receipt_bindings"][1] = (
            result["state_receipt_bindings"][1],
            result["state_receipt_bindings"][0],
        )
        evidence["join"]["ordered_state_binding_identity_sha256"] = (
            collector._ordered_binding_identity_v2(  # noqa: SLF001
                result["state_receipt_bindings"]
            )
        )
    with pytest.raises(collector.SceneProcessCollectionError):
        collector.validate_scene_process_evidence_v2(
            result,
            authority_binding=authority_binding,
            plan_binding=result["plan_binding"],
            plan=plan,
        )


def _mock_orchestration_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, dict[str, object]]:
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    attempt = tmp_path / ".generated/dev/replacement/attempt_v1"
    attempt.mkdir(parents=True)
    collection = attempt / "collection"
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}\n")
    plan_binding = pilot.file_binding(plan_file)
    plan = {
        **_plan(),
        "attempt_id": collector.ATTEMPT_ID,
        "output_root": str(collection),
    }
    authority = {
        "attempt_root": str(attempt),
        "collection_root": str(collection),
        "caps": copy.deepcopy(collector.EXPECTED_CAPS),
    }
    authority_binding = {
        "path": str(tmp_path / "authority.json"),
        "sha256": "b" * 64,
        "byte_count": 1,
    }
    monkeypatch.setattr(
        collector,
        "load_and_validate_v2",
        lambda **kwargs: (authority, authority_binding, plan, plan_binding),
    )
    counter = tmp_path / "vram_used"
    counter.write_text("0")
    monkeypatch.setattr(
        collector.calibration_supervisor,
        "_selected_gpu_memory_files",
        lambda _plan: (counter, counter, "vendor", "device"),
    )
    return authority, plan_file, plan_binding


def test_orchestrator_runs_64_exit_release_validate_cycles_then_final_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan_file, plan_binding = _mock_orchestration_inputs(
        tmp_path, monkeypatch
    )
    events: list[str] = []

    def run(
        _argv: list[str], *, scene_index: int, role: str, scene_id: str, **kwargs: object
    ) -> dict[str, object]:
        events.append(f"run-{scene_index}")
        return {
            "scene_index": scene_index,
            "role": role,
            "scene_id": scene_id,
            "pid": 1000 + scene_index,
            "prelaunch_baseline_used_bytes": scene_index,
            "exit_code": 0,
            "process_group_observation_error": None,
            "process_group_equality_observed": True,
            "parent_process_group_id": 777,
            "child_process_group_id": 777,
        }

    def load(*, scene: dict[str, object], **kwargs: object):
        index = int(scene["scene_index"])
        events.append(f"validate-{index}")
        return {"scene_index": index}, _binding(f"scene_results/{index:03d}.json")

    monkeypatch.setattr(collector, "_run_worker_process_v2", run)
    monkeypatch.setattr(collector, "_load_scene_result_v2", load)

    def wait(*args: object, baseline_used_bytes: int, **kwargs: object):
        events.append(f"release-{baseline_used_bytes}")
        return {"status": "PASSED"}

    monkeypatch.setattr(collector, "_wait_for_vram_release_v2", wait)
    monkeypatch.setattr(
        collector,
        "_barrier_with_identity_v2",
        lambda barrier, *, scene, worker_pid: {"scene_index": scene["scene_index"]},
    )

    def join(**kwargs: object) -> dict[str, object]:
        events.append("join")
        assert len(kwargs["release_barriers"]) == 64
        return {"status": "PHYSICS_COMPLETE"}

    monkeypatch.setattr(collector, "_join_scene_results_v2", join)
    result, result_path = collector.collect_v2(
        plan_path=plan_file,
        expected_plan_byte_count=plan_binding["byte_count"],
        expected_plan_sha256=plan_binding["file_sha256"],
        authority_path=tmp_path / "authority.json",
        expected_authority_byte_count=1,
        expected_authority_sha256="b" * 64,
    )
    expected_events = []
    for index in range(64):
        expected_events.extend(
            [f"run-{index}", f"release-{index}", f"validate-{index}"]
        )
    expected_events.append("join")
    assert events == expected_events
    assert result["status"] == "PHYSICS_COMPLETE"
    assert result_path == Path(authority["collection_root"]) / "physics_result.json"
    assert result_path.is_file()


@pytest.mark.parametrize(
    "failure_phase", ("worker", "pgid", "release", "malformed", "join")
)
def test_any_orchestration_failure_never_writes_physics_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    authority, plan_file, plan_binding = _mock_orchestration_inputs(
        tmp_path, monkeypatch
    )
    release_baselines: list[int] = []

    def run(_argv: list[str], *, scene_index: int, role: str, scene_id: str, **kwargs: object):
        return {
            "pid": 1000 + scene_index,
            "prelaunch_baseline_used_bytes": scene_index,
            "scene_index": scene_index,
            "exit_code": (
                1 if failure_phase == "worker" and scene_index == 3 else 0
            ),
            "process_group_observation_error": None,
            "process_group_equality_observed": not (
                failure_phase == "pgid" and scene_index == 3
            ),
            "parent_process_group_id": 777,
            "child_process_group_id": (
                778 if failure_phase == "pgid" and scene_index == 3 else 777
            ),
        }

    monkeypatch.setattr(collector, "_run_worker_process_v2", run)
    monkeypatch.setattr(
        collector,
        "_load_scene_result_v2",
        lambda *, scene, **kwargs: (
            (_ for _ in ()).throw(
                collector.SceneProcessCollectionError("malformed scene result")
            )
            if failure_phase == "malformed" and scene["scene_index"] == 3
            else (
                {"scene_index": scene["scene_index"]},
                _binding(f"scene_results/{scene['scene_index']:03d}.json"),
            )
        ),
    )

    def wait(*args: object, baseline_used_bytes: int, **kwargs: object):
        release_baselines.append(baseline_used_bytes)
        if failure_phase == "release" and baseline_used_bytes == 3:
            raise collector.SceneProcessCollectionError("release failure")
        return {"status": "PASSED"}

    monkeypatch.setattr(collector, "_wait_for_vram_release_v2", wait)
    monkeypatch.setattr(
        collector,
        "_barrier_with_identity_v2",
        lambda barrier, *, scene, worker_pid: {"scene_index": scene["scene_index"]},
    )

    def join(**kwargs: object):
        if failure_phase == "join":
            raise collector.SceneProcessCollectionError("join failure")
        return {"status": "PHYSICS_COMPLETE"}

    monkeypatch.setattr(collector, "_join_scene_results_v2", join)
    with pytest.raises(collector.SceneProcessCollectionError):
        collector.collect_v2(
            plan_path=plan_file,
            expected_plan_byte_count=plan_binding["byte_count"],
            expected_plan_sha256=plan_binding["file_sha256"],
            authority_path=tmp_path / "authority.json",
            expected_authority_byte_count=1,
            expected_authority_sha256="b" * 64,
        )
    assert not (Path(authority["collection_root"]) / "physics_result.json").exists()
    if failure_phase in {"worker", "pgid", "release", "malformed"}:
        assert 3 in release_baselines
