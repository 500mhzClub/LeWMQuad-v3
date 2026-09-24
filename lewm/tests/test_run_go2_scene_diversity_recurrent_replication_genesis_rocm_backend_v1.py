from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1
    as runner,
)


def _plan(*, role: str) -> dict:
    runtime = runner.plan_builder.build_rocm_runtime_bindings()
    if role == "scientific":
        attempt_id = runner.plan_builder.DEFAULT_ATTEMPT_ID
        output_root = runner.plan_builder.DEFAULT_OUTPUT_ROOT
    elif role == "qualification":
        attempt_id = runner.plan_builder.QUALIFICATION_ATTEMPT_ID
        output_root = runner.plan_builder.QUALIFICATION_OUTPUT_ROOT
    else:  # pragma: no cover - test helper guard
        raise AssertionError(role)
    return runner.plan_builder._expected_rocm_plan(  # noqa: SLF001
        attempt_id=attempt_id,
        output_root=output_root,
        plan_role=role,
        runtime_bindings=runtime,
    )


def test_runner_binds_cpu_hard_stop_and_new_runtime_dependency_closure() -> None:
    evidence = runner.predecessor_failure_bindings_rocm()
    assert evidence["predecessor_cpu_qualification_terminal_review"] == {
        "path": str(runner.CPU_TERMINAL_REVIEW.resolve()),
        "sha256": runner.CPU_TERMINAL_REVIEW_SHA256,
        "byte_count": runner.CPU_TERMINAL_REVIEW_BYTE_COUNT,
    }
    assert runner.AUTHORITY_STATUS.endswith("GENESIS_ROCM_BACKEND_V1")
    assert runner.SOURCE_PATHS["rocm_backend_runner"] == Path(
        runner.__file__
    ).resolve()
    assert len(runner.ROCM_BACKEND_DEPENDENCY_PATHS) == 16
    assert all(
        path.is_file() for path in runner.ROCM_BACKEND_DEPENDENCY_PATHS.values()
    )
    assert all(
        runner.plan_builder.ROCM_SITE_PACKAGES in path.parents
        for path in runner.ROCM_BACKEND_DEPENDENCY_PATHS.values()
    )


def _pass_result(collection_root: Path, plan: dict) -> dict:
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
                Path(
                    plan["execution_contract"]["python_invocation_path"]
                ).resolve()
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
    environment = runner.plan_builder.rocm_execution_environment(
        "qualification"
    )
    lld = str(
        Path(
            plan["runtime_bindings"]["rocm_lld_executable"]["path"]
        ).resolve()
    )
    return {
        "schema": runner.qualifier.QUALIFICATION_RESULT_SCHEMA,
        "status": runner.qualifier.QUALIFICATION_RESULT_STATUS,
        "attempt_id": runner.plan_builder.QUALIFICATION_ATTEMPT_ID,
        "backend": "amdgpu",
        "backend_api": "gs.amdgpu",
        "qualification_contract": runner.qualifier.QUALIFICATION_CONTRACT,
        "rocm_egl_preflight": {
            "status": "PASS_EXACT_ROCM_HIP_AND_EGL_R9700",
            "environment": environment,
            "expectation": expectation,
            "identity": {
                "torch_version": "2.12.0+rocm7.2",
                "torch_hip_version": "7.2.53211",
                "visible_device_count": 1,
                "device_name": "AMD Radeon AI PRO R9700",
                "arch_name": "gfx1201:sramecc+:xnack-",
                "genesis_version": "0.4.6",
                "genesis_backend_symbol": "gs.amdgpu",
                "hsa_override_present": False,
                "genesis_file": str(
                    Path(
                        plan["runtime_bindings"]["genesis_init_source"]["path"]
                    ).resolve()
                ),
                "torch_file": str(
                    (
                        runner.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                        / "torch/__init__.py"
                    ).resolve()
                ),
                "numpy_file": str(
                    (
                        runner.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                        / "numpy/__init__.py"
                    ).resolve()
                ),
                "pillow_file": str(
                    (
                        runner.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                        / "PIL/__init__.py"
                    ).resolve()
                ),
            },
            "egl_device_index": expectation["egl_device_index"],
            "path_ld_lld": lld,
            "rocm_path_ld_lld": lld,
            "lld_stdout_sha256": "a" * 64,
            "rocminfo_stdout_sha256": "b" * 64,
            "egl_stdout_sha256": "c" * 64,
            "egl_stderr_sha256": "d" * 64,
            "egl_exit_code": expectation["eglinfo_expected_exit_code"],
        },
        "probe_order": list(runner.qualifier.QUALIFICATION_PROBE_ORDER),
        "probes": probes,
        "kernel_reset_audit": {
            "query_succeeded": True,
            "new_amdgpu_ring_timeout_or_reset_count": 0,
            "matching_lines_sha256": "e" * 64,
        },
        "timing_gate": {
            "cold_scene_12_worker_elapsed_seconds": 80.0,
            "warm_scene_0_worker_elapsed_seconds": 70.0,
            "maximum_worker_elapsed_seconds": 80.0,
            "projected_scientific_wall_seconds": 6020.0,
            "wall_ceiling_seconds": runner.collector.EXPECTED_CAPS[
                "wall_seconds"
            ],
            "passed": True,
        },
        "contact_force_route_audit": runner.collector.CONTACT_FORCE_ROUTE_AUDIT,
        "all_existing_scene_gates_passed": True,
        "exact_v03_renderer_compatibility_passed": True,
        "scientific_attempt_root_absent": True,
        "probe_output_scientific_reuse_authorized": False,
        "authorizes_scientific_authority_consideration": True,
        "authorizes_retry_or_resume": False,
    }


def test_scientific_authority_requires_semantic_rocm_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_attempt = tmp_path / "qualification_attempt"
    collection_root = qualification_attempt / "collection"
    scientific_attempt = tmp_path / "scientific_attempt"
    plan_path = tmp_path / "qualification_plan.json"
    authority_path = tmp_path / "qualification_authority.json"
    result_path = qualification_attempt / "qualification_result.json"
    monkeypatch.setattr(
        runner.plan_builder, "QUALIFICATION_ATTEMPT_ROOT", qualification_attempt
    )
    monkeypatch.setattr(
        runner.plan_builder, "QUALIFICATION_OUTPUT_ROOT", collection_root
    )
    monkeypatch.setattr(
        runner.plan_builder, "QUALIFICATION_PLAN_OUTPUT", plan_path
    )
    monkeypatch.setattr(
        runner.plan_builder, "DEFAULT_ATTEMPT_ROOT", scientific_attempt
    )
    monkeypatch.setattr(
        runner.qualifier, "QUALIFICATION_AUTHORITY", authority_path
    )
    monkeypatch.setattr(
        runner.qualifier, "QUALIFICATION_RESULT_PATH", result_path
    )
    plan = _plan(role="qualification")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan_binding = runner.file_binding_v1(plan_path)
    authority_document = {
        name: None for name in runner.qualifier.QUALIFICATION_AUTHORITY_FIELDS
    }
    authority_document.update(
        {
            "schema": runner.qualifier.QUALIFICATION_AUTHORITY_SCHEMA,
            "status": runner.qualifier.QUALIFICATION_AUTHORITY_STATUS,
            "attempt_id": runner.plan_builder.QUALIFICATION_ATTEMPT_ID,
            "plan_binding": plan_binding,
            "qualification_contract": runner.qualifier.QUALIFICATION_CONTRACT,
            "predecessor_cpu_terminal_review_binding": runner._standard_binding(  # noqa: SLF001
                runner.plan_builder.CPU_TERMINAL_REVIEW_BINDING
            ),
        }
    )
    authority_path.write_text(
        json.dumps(authority_document, indent=2, sort_keys=True) + "\n"
    )
    authority_binding = runner.file_binding_v1(authority_path)
    passing = _pass_result(collection_root, plan)
    passing["authority_binding"] = authority_binding
    passing["plan_binding"] = plan_binding
    monkeypatch.setattr(
        runner,
        "_load_qualification_reservation",
        lambda **_kwargs: (
            {"orchestrator_pid": 900},
            {
                "path": "reservation.json",
                "file_sha256": "f" * 64,
                "byte_count": 1,
            },
        ),
    )

    replay_plan_bindings = []

    def revalidate_scene(*, scene, plan_binding, **_kwargs):
        replay_plan_bindings.append(dict(plan_binding))
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
    result_path.write_text(json.dumps(passing, indent=2, sort_keys=True) + "\n")
    binding = runner.file_binding_v1(result_path)
    result, observed = runner.validate_qualification_result_binding(binding)
    assert observed == binding
    assert result["status"] == runner.qualifier.QUALIFICATION_RESULT_STATUS
    assert len(replay_plan_bindings) == 2
    assert all(
        set(value) == {"path", "file_sha256", "byte_count"}
        for value in replay_plan_bindings
    )
    assert all(
        runner._standard_binding(value) == plan_binding  # noqa: SLF001
        for value in replay_plan_bindings
    )

    mutations = []
    changed = copy.deepcopy(passing)
    changed["backend"] = "cpu"
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["rocm_egl_preflight"]["identity"]["device_name"] = "wrong"
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probe_output_scientific_reuse_authorized"] = True
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["worker"]["elapsed_seconds"] = float("nan")
    mutations.append(changed)
    changed = copy.deepcopy(passing)
    changed["probes"][0]["scene_result_binding"]["file_sha256"] = "0" * 64
    mutations.append(changed)
    for changed in mutations:
        result_path.write_text(
            json.dumps(changed, indent=2, sort_keys=True) + "\n"
        )
        changed_binding = runner.file_binding_v1(result_path)
        with pytest.raises(
            runner.SceneDiversityRunnerError, match="did not pass"
        ):
            runner.validate_qualification_result_binding(changed_binding)


def test_rocm_plan_validator_accepts_only_scientific_plan() -> None:
    plan = _plan(role="scientific")
    runner._validate_plan_rocm(  # noqa: SLF001
        plan, {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
    )
    changed = copy.deepcopy(plan)
    changed["execution_contract"]["backend"] = "vulkan"
    with pytest.raises(runner.SceneDiversityRunnerError, match="changed"):
        runner._validate_plan_rocm(  # noqa: SLF001
            changed, {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
        )


def test_runner_overlay_scopes_all_inherited_plan_validators() -> None:
    v3_original = runner.predecessor_runner._validate_plan_v3  # noqa: SLF001
    v2_original = runner.v2_runner._validate_plan_v2  # noqa: SLF001
    v1_original = runner.v1_replacement_runner._validate_plan_v1  # noqa: SLF001
    with runner._configured_predecessor_runner_rocm():  # noqa: SLF001
        assert runner.predecessor_runner._validate_plan_v3 is runner._validate_plan_rocm  # noqa: SLF001
        assert runner.v2_runner._validate_plan_v2 is runner._validate_plan_rocm  # noqa: SLF001
        assert runner.v1_replacement_runner._validate_plan_v1 is runner._validate_plan_rocm  # noqa: SLF001
        assert runner.collector.pilot.EXECUTION_ENVIRONMENT == (
            runner.plan_builder.rocm_execution_environment("scientific")
        )
    assert runner.predecessor_runner._validate_plan_v3 is v3_original  # noqa: SLF001
    assert runner.v2_runner._validate_plan_v2 is v2_original  # noqa: SLF001
    assert runner.v1_replacement_runner._validate_plan_v1 is v1_original  # noqa: SLF001


def test_execute_rechecks_qualification_before_delegation(monkeypatch) -> None:
    observed: list[str] = []
    monkeypatch.setattr(
        runner, "_validate_plan_rocm", lambda *_args: observed.append("plan")
    )
    monkeypatch.setattr(
        runner,
        "predecessor_failure_bindings_rocm",
        lambda: observed.append("cpu_terminal") or {},
    )
    monkeypatch.setattr(
        runner,
        "validate_qualification_result_binding",
        lambda _value: observed.append("qualification") or ({}, {}),
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "execute_v3",
        lambda *_args, **_kwargs: observed.append("execute")
        or {"status": "TEST"},
    )
    authority = {
        "qualification_result_binding": {},
        "predecessor_cpu_terminal_review_binding": runner._standard_binding(  # noqa: SLF001
            runner.plan_builder.CPU_TERMINAL_REVIEW_BINDING
        ),
    }
    result = runner.execute_rocm(authority, authority_binding={}, plan={})
    assert result == {"status": "TEST"}
    assert observed == ["plan", "cpu_terminal", "qualification", "execute"]
