from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as runner,
)


def test_cpu_runner_binds_v3_hard_stop_and_material_identity() -> None:
    evidence = runner.predecessor_failure_bindings_cpu()
    assert evidence["predecessor_v3_failure_terminal"] == {
        "path": str(runner.PREDECESSOR_V3_TERMINAL.resolve()),
        "sha256": runner.PREDECESSOR_V3_TERMINAL_SHA256,
        "byte_count": runner.PREDECESSOR_V3_TERMINAL_BYTE_COUNT,
    }
    assert evidence["predecessor_v3_terminal_review"] == {
        "path": str(runner.PREDECESSOR_V3_TERMINAL_REVIEW.resolve()),
        "sha256": runner.PREDECESSOR_V3_TERMINAL_REVIEW_SHA256,
        "byte_count": runner.PREDECESSOR_V3_TERMINAL_REVIEW_BYTE_COUNT,
    }
    assert runner.AUTHORITY_STATUS.endswith("CPU_BACKEND_V1")
    assert runner.SOURCE_PATHS["cpu_backend_runner"] == Path(runner.__file__).resolve()
    assert set(runner.CPU_BACKEND_DEPENDENCY_PATHS) == {
        "cpu_backend_dependency_genesis_constants",
        "cpu_backend_dependency_rigid_constraint_solver",
        "cpu_backend_dependency_rigid_collider",
    }
    assert all(path.is_file() for path in runner.CPU_BACKEND_DEPENDENCY_PATHS.values())


def _pass_result(collection_root: Path) -> dict:
    plan = json.loads(runner.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text())
    scenes = runner.qualifier.predecessor._scene_slices_v2(plan)  # noqa: SLF001
    scene_results = collection_root / "scene_results"
    scene_results.mkdir(parents=True)
    probes = []
    elapsed_by_index = {12: 80.0, 0: 70.0}
    for index in runner.qualifier.QUALIFICATION_PROBE_ORDER:
        scene = scenes[index]
        relative = Path("scene_results") / f"{index:03d}.json"
        payload = b"{}\n"
        (collection_root / relative).write_bytes(payload)
        baseline = 0
        margin = runner.qualifier.predecessor.VRAM_RELEASE_MARGIN_BYTES
        worker = {
            "scene_index": index,
            "role": scene["role"],
            "scene_id": scene["scene_id"],
            "pid": 1000 + index,
            "parent_pid": 900,
            "process_group_id": 1000 + index,
            "fresh_process_group": True,
            "sys_executable": str(
                Path(plan["execution_contract"]["python_invocation_path"]).resolve()
            ),
            "prelaunch_baseline_used_bytes": baseline,
            "peak_selected_device_vram_bytes": baseline,
            "selected_device_vram_cap_breached": False,
            "watchdog_timeout": False,
            "exit_code": 0,
            "elapsed_seconds": elapsed_by_index[index],
        }
        barrier = {
            "status": "PASSED",
            "read_only": True,
            "counter_path": "/sys/test-vram-counter",
            "baseline_used_bytes": baseline,
            "release_margin_bytes": margin,
            "release_ceiling_bytes": baseline + margin,
            "absolute_vram_ceiling_bytes": runner.collector.EXPECTED_CAPS[
                "selected_device_vram_byte_ceiling"
            ],
            "required_consecutive_samples": (
                runner.qualifier.predecessor.VRAM_RELEASE_CONSECUTIVE_SAMPLES
            ),
            "sample_interval_seconds": (
                runner.qualifier.predecessor.VRAM_RELEASE_POLL_SECONDS
            ),
            "sample_count": (
                runner.qualifier.predecessor.VRAM_RELEASE_CONSECUTIVE_SAMPLES
            ),
            "minimum_used_bytes": baseline,
            "maximum_used_bytes": baseline,
            "final_used_bytes": baseline,
            "final_consecutive_samples": (
                runner.qualifier.predecessor.VRAM_RELEASE_CONSECUTIVE_SAMPLES
            ),
            "elapsed_seconds": 0.01,
        }
        probes.append(
            {
                "scene_index": index,
                "role": scene["role"],
                "scene_id": scene["scene_id"],
                "worker": worker,
                "release_barrier": barrier,
                "scene_result_binding": {
                    "path": relative.as_posix(),
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                },
                "observed_counts": (
                    runner.qualifier.predecessor._scene_expected_counts_v2(  # noqa: SLF001
                        str(scene["role"])
                    )
                ),
                "existing_scene_validation_passed": True,
                "probe_output_scientific_reuse_authorized": False,
            }
        )
    expectation = plan["execution_contract"]["graphics_preflight"]
    return {
        "schema": runner.qualifier.QUALIFICATION_RESULT_SCHEMA,
        "status": runner.qualifier.QUALIFICATION_RESULT_STATUS,
        "attempt_id": runner.plan_builder.QUALIFICATION_ATTEMPT_ID,
        "backend": "cpu",
        "qualification_contract": runner.qualifier.QUALIFICATION_CONTRACT,
        "graphics_preflight": {
            "phase": "graphics_preflight",
            "status": "PASS",
            "environment": runner.plan_builder.CPU_EXECUTION_ENVIRONMENT,
            "expectation": expectation,
            "vulkan_stdout_sha256": "a" * 64,
            "egl_stdout_sha256": "b" * 64,
            "egl_stderr_sha256": "c" * 64,
            "egl_exit_code": expectation["eglinfo_expected_exit_code"],
        },
        "probe_order": list(runner.qualifier.QUALIFICATION_PROBE_ORDER),
        "probes": probes,
        "kernel_reset_audit": {
            "query_succeeded": True,
            "new_amdgpu_ring_timeout_or_reset_count": 0,
            "matching_lines_sha256": "d" * 64,
        },
        "timing_gate": {
            "maximum_worker_elapsed_seconds": 80.0,
            "passed": True,
            "projected_scientific_wall_seconds": 6020.0,
            "wall_ceiling_seconds": runner.collector.EXPECTED_CAPS["wall_seconds"],
        },
        "all_existing_scene_gates_passed": True,
        "scientific_attempt_root_absent": True,
        "probe_output_scientific_reuse_authorized": False,
        "authorizes_cpu_scientific_authority_consideration": True,
        "authorizes_retry_or_resume": False,
    }


def test_scientific_authority_requires_exact_pass_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qualification_result.json"
    qualification_authority = tmp_path / "qualification_authority.json"
    qualification_authority.write_text("{}\n")
    monkeypatch.setattr(
        runner.qualifier, "QUALIFICATION_AUTHORITY", qualification_authority
    )
    monkeypatch.setattr(runner.qualifier, "QUALIFICATION_RESULT_PATH", path)
    collection_root = tmp_path / "collection"
    monkeypatch.setattr(
        runner.plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection_root
    )
    monkeypatch.setattr(
        runner.plan_builder, "validate_cpu_plan", lambda plan, **_kwargs: plan
    )
    passing = _pass_result(collection_root)
    passing["authority_binding"] = runner.file_binding_v1(qualification_authority)
    passing["plan_binding"] = runner.file_binding_v1(
        runner.plan_builder.QUALIFICATION_PLAN_OUTPUT
    )
    monkeypatch.setattr(
        runner,
        "_load_qualification_reservation",
        lambda **_kwargs: (
            {"orchestrator_pid": 900},
            {
                "path": "reservation.json",
                "file_sha256": "e" * 64,
                "byte_count": 1,
            },
        ),
    )

    def revalidate_scene(*, scene, **_kwargs):
        relative = Path("scene_results") / f"{int(scene['scene_index']):03d}.json"
        raw = (collection_root / relative).read_bytes()
        return (
            {
                "observed_counts": (
                    runner.qualifier.predecessor._scene_expected_counts_v2(  # noqa: SLF001
                        str(scene["role"])
                    )
                )
            },
            {
                "path": relative.as_posix(),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            },
        )

    monkeypatch.setattr(
        runner, "_revalidate_qualification_scene_result", revalidate_scene
    )
    path.write_text(json.dumps(passing, indent=2, sort_keys=True) + "\n")
    binding = runner.file_binding_v1(path)
    result, observed = runner.validate_qualification_result_binding(binding)
    assert observed == binding
    assert result["status"] == runner.qualifier.QUALIFICATION_RESULT_STATUS

    mutations = []
    changed = copy.deepcopy(passing)
    del changed["attempt_id"]
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    del changed["probes"][0]["worker"]["exit_code"]
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["release_barrier"]["status"] = "FAILED"
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["observed_counts"]["actions"] = 8
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["scene_result_binding"]["file_sha256"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["worker"]["elapsed_seconds"] = float("nan")
    mutations.append(changed)
    for changed in mutations:
        path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
        changed_binding = runner.file_binding_v1(path)
        with pytest.raises(runner.SceneDiversityRunnerError, match="did not pass"):
            runner.validate_qualification_result_binding(changed_binding)


def test_semantic_revalidator_rejects_empty_scene_result_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_root = tmp_path / "collection"
    result = _pass_result(collection_root)
    plan = json.loads(runner.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text())
    scene = runner.qualifier.predecessor._scene_slices_v2(plan)[12]  # noqa: SLF001
    monkeypatch.setattr(
        runner.plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection_root
    )
    with pytest.raises(
        runner.SceneDiversityRunnerError, match="scene evidence did not validate"
    ):
        runner._revalidate_qualification_scene_result(  # noqa: SLF001
            authority={},
            authority_binding={},
            plan=plan,
            plan_binding={},
            reservation_binding={},
            scene=scene,
            worker=result["probes"][0]["worker"],
        )


def test_cpu_plan_validator_accepts_only_scientific_cpu_plan() -> None:
    plan = json.loads(runner.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    runner._validate_plan_cpu(  # noqa: SLF001
        plan, {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
    )
    plan["execution_contract"]["backend"] = "vulkan"
    with pytest.raises(runner.SceneDiversityRunnerError, match="backend"):
        runner._validate_plan_cpu(  # noqa: SLF001
            plan, {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
        )


def test_runner_overlay_scopes_all_inherited_plan_validators() -> None:
    v3_original = runner.predecessor_runner._validate_plan_v3  # noqa: SLF001
    v2_original = runner.v2_runner._validate_plan_v2  # noqa: SLF001
    v1_original = runner.v1_replacement_runner._validate_plan_v1  # noqa: SLF001
    with runner._configured_predecessor_runner_cpu():  # noqa: SLF001
        assert runner.predecessor_runner._validate_plan_v3 is runner._validate_plan_cpu  # noqa: SLF001
        assert runner.v2_runner._validate_plan_v2 is runner._validate_plan_cpu  # noqa: SLF001
        assert runner.v1_replacement_runner._validate_plan_v1 is runner._validate_plan_cpu  # noqa: SLF001
    assert runner.predecessor_runner._validate_plan_v3 is v3_original  # noqa: SLF001
    assert runner.v2_runner._validate_plan_v2 is v2_original  # noqa: SLF001
    assert runner.v1_replacement_runner._validate_plan_v1 is v1_original  # noqa: SLF001


def test_execute_rechecks_qualification_before_delegation(monkeypatch) -> None:
    observed: list[str] = []
    monkeypatch.setattr(runner, "_validate_plan_cpu", lambda *_args: observed.append("plan"))
    monkeypatch.setattr(
        runner, "predecessor_failure_bindings_cpu", lambda: observed.append("v3") or {}
    )
    monkeypatch.setattr(
        runner,
        "validate_qualification_result_binding",
        lambda _value: observed.append("qualification") or ({}, {}),
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "execute_v3",
        lambda *_args, **_kwargs: observed.append("execute") or {"status": "TEST"},
    )
    result = runner.execute_cpu(
        {"qualification_result_binding": {}},
        authority_binding={},
        plan={},
    )
    assert result == {"status": "TEST"}
    assert observed == ["plan", "v3", "qualification", "execute"]
