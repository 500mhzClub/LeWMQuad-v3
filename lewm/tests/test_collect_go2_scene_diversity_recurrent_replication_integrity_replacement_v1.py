from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import (
    collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v1
    as collector,
)


def _binding(path: str) -> dict[str, object]:
    return {"path": path, "file_sha256": "a" * 64, "byte_count": 1}


def _plan() -> dict[str, object]:
    states: list[dict[str, object]] = []
    for role in collector.ROLE_ORDER:
        for scene_index in range(collector.ROLE_SCENE_COUNT):
            scene_id = f"{role}-scene-{scene_index:02d}"
            for state_index in range(4):
                states.append(
                    {
                        "role": role,
                        "scene_id": scene_id,
                        "state_id": f"{role}-state-{scene_index:02d}-{state_index}",
                    }
                )
    return {
        "states": states,
        "execution_contract": {
            "seed": 20260802,
            "python_invocation_path": str(Path(sys.executable).resolve()),
        },
    }


def _valid_result() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    plan = _plan()
    role_results: dict[str, dict[str, object]] = {}
    role_bindings: dict[str, dict[str, object]] = {}
    worker_receipts = []
    for index, role in enumerate(collector.ROLE_ORDER):
        role_states = [row for row in plan["states"] if row["role"] == role]
        scenes: list[str] = []
        for row in role_states:
            if row["scene_id"] not in scenes:
                scenes.append(row["scene_id"])
        state_bindings = [
            _binding(
                f"scenes/{role}/{row['scene_id']}/state_receipts/"
                f"{row['state_id']}.json"
            )
            for row in role_states
        ]
        render_bindings = [
            _binding(f"scenes/{role}/{scene}/live_render_receipt.json")
            for scene in scenes
        ]
        expected = collector._role_expected_counts_v1(role)  # noqa: SLF001
        role_results[role] = {
            "execution_seed": 20260802,
            "role_local_mesh_cache": {
                "path": f"scenes/{role}/derived_meshes",
                "cross_role_reuse_authorized": False,
                "mesh_count": 1,
                "bindings_identity_sha256": "d" * 64,
            },
            "expected_counts": expected,
            "observed_counts": copy.deepcopy(expected),
            "state_receipt_bindings": state_bindings,
            "render_receipt_bindings": render_bindings,
            "scene_metrics": [
                {"role": role, "scene_id": scene, "states": 4}
                for scene in scenes
            ],
        }
        role_bindings[role] = _binding(f"role_results/{role}.json")
        worker_receipts.append(
            {
                "role": role,
                "pid": 1000 + index,
                "parent_pid": 999,
                "sys_executable": str(Path(sys.executable).resolve()),
            }
        )
    barrier = {
        "status": "PASSED",
        "read_only": True,
        "counter_path": "/sys/mock/mem_info_vram_used",
        "baseline_used_bytes": 1_000_000_000,
        "release_margin_bytes": collector.VRAM_RELEASE_MARGIN_BYTES,
        "release_ceiling_bytes": 1_000_000_000
        + collector.VRAM_RELEASE_MARGIN_BYTES,
        "absolute_vram_ceiling_bytes": collector.EXPECTED_CAPS[
            "selected_device_vram_byte_ceiling"
        ],
        "required_consecutive_samples": collector.VRAM_RELEASE_CONSECUTIVE_SAMPLES,
        "sample_interval_seconds": collector.VRAM_RELEASE_POLL_SECONDS,
        "sample_count": 3,
        "minimum_used_bytes": 1_000_000_000,
        "maximum_used_bytes": 1_100_000_000,
        "final_used_bytes": 1_000_000_000,
        "final_consecutive_samples": 3,
        "elapsed_seconds": 0.1,
    }
    evidence = collector._build_split_evidence_v1(  # noqa: SLF001
        plan=plan,
        role_results=role_results,
        role_result_bindings=role_bindings,
        worker_receipts=worker_receipts,
        release_barrier=barrier,
    )
    result = {
        "authority_binding": {"path": "/authority", "sha256": "b" * 64, "byte_count": 1},
        "plan_binding": {"path": "/plan", "file_sha256": "c" * 64, "byte_count": 1},
        "state_receipt_bindings": [
            row
            for role in collector.ROLE_ORDER
            for row in role_results[role]["state_receipt_bindings"]
        ],
        "render_receipt_bindings": [
            row
            for role in collector.ROLE_ORDER
            for row in role_results[role]["render_receipt_bindings"]
        ],
        "scene_metrics": [
            row
            for role in collector.ROLE_ORDER
            for row in role_results[role]["scene_metrics"]
        ],
        "role_result_bindings": [role_bindings[role] for role in collector.ROLE_ORDER],
        "split_process_evidence": evidence,
    }
    return result, plan, result["authority_binding"]


def test_both_workers_initialize_from_exact_full_plan_first_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    binding = pilot.write_json_exclusive(
        manifest,
        {"physics_seed": collector.PLAN_FIRST_PHYSICS_SEED},
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
        "_initialize_genesis_v1",
        lambda *, backend, seed: calls.append((backend, seed)),
    )

    train = collector._initialize_from_plan_first_scene_v1(plan=plan)  # noqa: SLF001
    evaluation = collector._initialize_from_plan_first_scene_v1(plan=plan)  # noqa: SLF001

    assert calls == [
        ("vulkan", collector.PLAN_FIRST_PHYSICS_SEED),
        ("vulkan", collector.PLAN_FIRST_PHYSICS_SEED),
    ]
    assert train == evaluation
    assert train["effective_genesis_seed"] == 315_871_188


def test_worker_inherits_outer_process_group_and_uses_sys_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Process:
        pid = 1234

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(collector.subprocess, "Popen", Process)
    receipt = collector._run_worker_process_v1(  # noqa: SLF001
        [sys.executable, "/collector.py", "--worker-role", "train"],
        role="train",
    )

    assert captured["argv"][0] == sys.executable
    assert captured["kwargs"] == {"cwd": collector.REPO_ROOT}
    assert receipt["inherited_outer_process_group"] is True
    assert receipt["exit_code"] == 0


def test_mesh_cache_is_role_local_for_kernel_and_render_replay(
    tmp_path: Path,
) -> None:
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
    role_root = tmp_path / "collection/scenes/eval"
    role_root.mkdir(parents=True)
    cache_root = collector._install_role_local_mesh_cache_v1(  # noqa: SLF001
        runtime, role_root=role_root
    )

    assert runtime["cached_box_obj"]((1.0, 2.0, 3.0)) == str(
        cache_root / "mesh.obj"
    )
    render_cached = render_namespace["build_scene"].__globals__["cached_box_obj"]
    assert render_cached((4.0, 5.0, 6.0)) == str(
        cache_root / "mesh.obj"
    )
    assert calls == [
        ((1.0, 2.0, 3.0), 0.7, cache_root),
        ((4.0, 5.0, 6.0), 0.7, cache_root),
    ]
    assert cache_root.is_relative_to(role_root)


def test_release_barrier_requires_three_bounded_read_only_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = 1_000_000_000
    release = baseline + collector.VRAM_RELEASE_MARGIN_BYTES
    counter = tmp_path / "counter"
    counter.write_text(str(baseline))
    values = iter([release + 1, release, release - 1, release])
    monkeypatch.setattr(
        collector, "_read_vram_counter_v1", lambda _path: next(values)
    )
    monkeypatch.setattr(collector.time, "sleep", lambda _seconds: None)

    evidence = collector._wait_for_vram_release_v1(  # noqa: SLF001
        counter,
        baseline_used_bytes=baseline,
        ceiling_bytes=collector.EXPECTED_CAPS["selected_device_vram_byte_ceiling"],
    )

    assert evidence["status"] == "PASSED"
    assert evidence["read_only"] is True
    assert evidence["release_margin_bytes"] == 512 * 1024 * 1024
    assert evidence["sample_count"] == 4
    assert evidence["final_consecutive_samples"] == 3


def test_relative_receipt_binding_is_rehashed_and_missing_or_tampered_rejects(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "scenes/train/scene/state_receipts/state.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n")
    absolute = pilot.file_binding(receipt)
    relative = {
        "path": "scenes/train/scene/state_receipts/state.json",
        "file_sha256": absolute["file_sha256"],
        "byte_count": absolute["byte_count"],
    }
    assert collector._rehash_relative_binding_v1(  # noqa: SLF001
        relative, collection_root=tmp_path, label="state"
    ) == relative
    receipt.write_text('{"changed":true}\n')
    with pytest.raises(collector.SplitCollectionError, match="content binding"):
        collector._rehash_relative_binding_v1(  # noqa: SLF001
            relative, collection_root=tmp_path, label="state"
        )
    receipt.unlink()
    with pytest.raises(collector.SplitCollectionError, match="absent"):
        collector._rehash_relative_binding_v1(  # noqa: SLF001
            relative, collection_root=tmp_path, label="state"
        )


def test_split_evidence_accepts_only_exact_workers_seed_barrier_and_join() -> None:
    result, plan, authority_binding = _valid_result()
    report = collector.validate_split_collection_evidence_v1(
        result,
        authority_binding=authority_binding,
        plan_binding=result["plan_binding"],
        plan=plan,
    )
    assert report == {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barrier_exact": True,
        "join_exact": True,
    }


def _mock_orchestration_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object], Path, dict[str, object]]:
    monkeypatch.setattr(collector, "REPO_ROOT", tmp_path)
    development = tmp_path / ".generated/dev"
    attempt = development / "replacement/attempt_v1"
    attempt.mkdir(parents=True)
    collection = attempt / "collection"
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}\n")
    plan_binding = pilot.file_binding(plan_file)
    plan = {
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
        "load_and_validate_v1",
        lambda **_kwargs: (authority, authority_binding, plan, plan_binding),
    )
    counter = tmp_path / "vram_used"
    counter.write_text("0")
    monkeypatch.setattr(
        collector.calibration_supervisor,
        "_selected_gpu_memory_files",
        lambda _plan: (counter, counter, "vendor", "device"),
    )
    return authority, plan, plan_file, plan_binding


def test_orchestrator_orders_train_exit_release_eval_join_then_final_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _plan_document, plan_file, plan_binding = _mock_orchestration_inputs(
        tmp_path, monkeypatch
    )
    events: list[str] = []

    def run(_argv: list[str], *, role: str) -> dict[str, object]:
        events.append(f"run-{role}")
        return {"role": role, "pid": 100 if role == "train" else 101}

    def load(*, role: str, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        events.append(f"validate-{role}")
        return {"role": role}, _binding(f"role_results/{role}.json")

    monkeypatch.setattr(collector, "_run_worker_process_v1", run)
    monkeypatch.setattr(collector, "_load_role_result_v1", load)
    monkeypatch.setattr(
        collector,
        "_wait_for_vram_release_v1",
        lambda *_args, **_kwargs: events.append("release") or {"status": "PASSED"},
    )

    def join(**_kwargs: object) -> dict[str, object]:
        events.append("join")
        return {"status": "PHYSICS_COMPLETE"}

    monkeypatch.setattr(collector, "_join_role_results_v1", join)
    result, result_path = collector.collect_v1(
        plan_path=plan_file,
        expected_plan_byte_count=plan_binding["byte_count"],
        expected_plan_sha256=plan_binding["file_sha256"],
        authority_path=tmp_path / "authority.json",
        expected_authority_byte_count=1,
        expected_authority_sha256="b" * 64,
    )

    assert events == [
        "run-train",
        "validate-train",
        "release",
        "run-eval",
        "validate-eval",
        "join",
    ]
    assert result["status"] == "PHYSICS_COMPLETE"
    assert result_path == Path(authority["collection_root"]) / "physics_result.json"
    assert result_path.is_file()


@pytest.mark.parametrize("failure_phase", ("train", "release", "eval"))
def test_orchestrator_failure_never_writes_combined_physics_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    authority, _plan_document, plan_file, plan_binding = _mock_orchestration_inputs(
        tmp_path, monkeypatch
    )

    def run(_argv: list[str], *, role: str) -> dict[str, object]:
        if failure_phase == role:
            raise collector.SplitCollectionError(f"{role} failed")
        return {"role": role, "pid": 100 if role == "train" else 101}

    monkeypatch.setattr(collector, "_run_worker_process_v1", run)
    monkeypatch.setattr(
        collector,
        "_load_role_result_v1",
        lambda *, role, **_kwargs: (
            {"role": role},
            _binding(f"role_results/{role}.json"),
        ),
    )

    def release(*_args: object, **_kwargs: object) -> dict[str, object]:
        if failure_phase == "release":
            raise collector.SplitCollectionError("release failed")
        return {"status": "PASSED"}

    monkeypatch.setattr(collector, "_wait_for_vram_release_v1", release)
    monkeypatch.setattr(
        collector,
        "_join_role_results_v1",
        lambda **_kwargs: {"status": "PHYSICS_COMPLETE"},
    )
    with pytest.raises(collector.SplitCollectionError):
        collector.collect_v1(
            plan_path=plan_file,
            expected_plan_byte_count=plan_binding["byte_count"],
            expected_plan_sha256=plan_binding["file_sha256"],
            authority_path=tmp_path / "authority.json",
            expected_authority_byte_count=1,
            expected_authority_sha256="b" * 64,
        )
    assert not (Path(authority["collection_root"]) / "physics_result.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "worker_order",
        "worker_seed",
        "worker_pid_reuse",
        "release_failed",
        "release_too_slow",
        "joined_state_reordered",
        "role_result_reordered",
    ),
)
def test_split_evidence_rejects_adversarial_drift(mutation: str) -> None:
    result, plan, authority_binding = _valid_result()
    evidence = result["split_process_evidence"]
    if mutation == "worker_order":
        evidence["workers"].reverse()
    elif mutation == "worker_seed":
        evidence["workers"][1]["full_genesis_seed"] += 1
    elif mutation == "worker_pid_reuse":
        evidence["workers"][1]["pid"] = evidence["workers"][0]["pid"]
    elif mutation == "release_failed":
        evidence["release_barrier"]["status"] = "FAILED"
    elif mutation == "release_too_slow":
        evidence["release_barrier"]["elapsed_seconds"] = 61.0
    elif mutation == "joined_state_reordered":
        result["state_receipt_bindings"][0], result["state_receipt_bindings"][1] = (
            result["state_receipt_bindings"][1],
            result["state_receipt_bindings"][0],
        )
        evidence["join"]["ordered_state_binding_identity_sha256"] = (
            collector._ordered_binding_identity_v1(  # noqa: SLF001
                result["state_receipt_bindings"]
            )
        )
    elif mutation == "role_result_reordered":
        result["role_result_bindings"].reverse()
        evidence["workers"][0]["role_result_binding"], evidence["workers"][1][
            "role_result_binding"
        ] = (
            evidence["workers"][1]["role_result_binding"],
            evidence["workers"][0]["role_result_binding"],
        )
    with pytest.raises(collector.SplitCollectionError):
        collector.validate_split_collection_evidence_v1(
            result,
            authority_binding=authority_binding,
            plan_binding=result["plan_binding"],
            plan=plan,
        )
