from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import (
    qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1
    as qualifier,
)


def _frozen() -> dict:
    return json.loads(qualifier.plan_builder.FROZEN_V1_EXACT_PLAN.read_text())


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return qualifier.plan_builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def scientific_plan(runtime_bindings: dict) -> dict:
    return qualifier.plan_builder.build_scientific_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


@pytest.fixture(scope="module")
def qualification_plan(runtime_bindings: dict) -> dict:
    return qualifier.plan_builder.build_qualification_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


def test_qualification_contract_is_exact_bounded_and_nonreusable() -> None:
    contract = qualifier.QUALIFICATION_CONTRACT
    assert contract["probe_scene_indices_in_order"] == [12, 0]
    assert contract["fresh_worker_processes"] == 2
    assert contract["states_per_worker"] == 4
    assert contract["candidate_actions_per_state"] == 9
    assert contract["branches_per_worker"] == 36
    assert contract["context_frames_per_worker"] == 12
    assert contract["target_frames_per_worker"] == 36
    assert contract["stored_rgb_frames_per_worker"] == 48
    assert contract["auxiliary_depth_validation_renders_per_worker"] == 48
    assert contract["worker_process_group_watchdog_seconds"] == 300.0
    assert contract["timing_gate_formula"] == (
        "64 * max(worker_elapsed_seconds) + 900 <= 7200"
    )
    assert contract["contact_force_reads_forbidden"] is True
    assert contract["probe_output_scientific_reuse_authorized"] is False
    assert qualifier.plan_builder.QUALIFICATION_ATTEMPT_ROOT != (
        qualifier.plan_builder.DEFAULT_ATTEMPT_ROOT
    )


def test_qualification_overlay_uses_separate_identity_and_restores() -> None:
    names = (
        "AUTHORITY_FIELDS",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "ATTEMPT_ID",
        "RESERVATION_SCHEMA",
        "SCENE_RESULT_SCHEMA",
        "_read_collection_reservation_rocm",
        "_worker_argv_rocm",
    )
    originals = {name: getattr(qualifier.collector, name) for name in names}
    with qualifier._configured_qualification_collector():  # noqa: SLF001
        assert qualifier.collector.ATTEMPT_ID == (
            qualifier.plan_builder.QUALIFICATION_ATTEMPT_ID
        )
        assert qualifier.collector.AUTHORITY_FIELDS == (
            qualifier.QUALIFICATION_AUTHORITY_FIELDS
        )
        assert qualifier.collector._worker_argv_rocm is (  # noqa: SLF001
            qualifier._worker_argv_qualification  # noqa: SLF001
        )
    assert all(
        getattr(qualifier.collector, name) is value
        for name, value in originals.items()
    )


def test_nested_plan_validation_does_not_recurse(
    scientific_plan: dict, qualification_plan: dict
) -> None:
    with (
        qualifier._configured_qualification_collector(),  # noqa: SLF001
        qualifier.collector._configured_predecessor_collector_rocm(),  # noqa: SLF001
    ):
        assert qualifier.collector.pilot.validate_plan(qualification_plan) == (
            qualification_plan
        )
    with qualifier.collector._configured_predecessor_collector_rocm():  # noqa: SLF001
        assert qualifier.collector.pilot.validate_plan(scientific_plan) == (
            scientific_plan
        )


def test_child_environment_removes_ambient_runtime_selectors(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    poison = {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        "LD_LIBRARY_PATH": "/poison",
        "VK_ICD_FILENAMES": "/poison.json",
        "QD_ARCH": "cpu",
        "QD_ENABLE_AMDGPU": "0",
        "GS_TORCH_FORCE_CPU_DEVICE": "1",
        "ROCM_PATH": "/wrong",
        "GS_CACHE_FILE_PATH": "/shared/cache",
    }
    for key, value in poison.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("QUALIFIER_TEST_SECRET", "do-not-persist")

    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    expected = qualification_plan["execution_contract"]["environment"]
    assert all(child[key] == value for key, value in expected.items())
    assert child["QUALIFIER_TEST_SECRET"] == "do-not-persist"
    assert "HSA_OVERRIDE_GFX_VERSION" not in child
    assert "LD_LIBRARY_PATH" not in child
    assert "VK_ICD_FILENAMES" not in child
    assert "QD_ARCH" not in child
    assert "QD_ENABLE_AMDGPU" not in child
    assert "GS_TORCH_FORCE_CPU_DEVICE" not in child


def _mock_preflight_run(*, bad_selected_egl_device: bool = False):
    identity = {
        "arch_name": "gfx1201:sramecc-:xnack-",
        "device_name": "AMD Radeon AI PRO R9700",
        "genesis_file": str(
            qualifier.plan_builder.ROCM_RUNTIME_PATHS[
                "genesis_init_source"
            ].resolve()
        ),
        "genesis_backend_symbol": "gs.amdgpu",
        "genesis_version": "0.4.6",
        "hsa_override_present": False,
        "numpy_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "numpy/__init__.py"
            ).resolve()
        ),
        "pillow_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "PIL/__init__.py"
            ).resolve()
        ),
        "torch_file": str(
            (
                qualifier.plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "torch/__init__.py"
            ).resolve()
        ),
        "torch_hip_version": "7.2.0",
        "torch_version": "2.12.0+rocm7.2",
        "visible_device_count": 1,
    }

    def run(argv, **_kwargs):
        executable = Path(argv[0]).name
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, "AMD LLD 20.0.0\n", "")
        if executable == "rocminfo":
            return subprocess.CompletedProcess(argv, 0, "Name: gfx1201\n", "")
        if "python" in executable:
            stdout = "Genesis logging banner\n" + json.dumps(identity) + "\n"
            return subprocess.CompletedProcess(argv, 0, stdout, "")
        selected = "llvmpipe" if bad_selected_egl_device else "AMD Radeon AI PRO R9700"
        stdout = (
            "Device #0:\n"
            "EGL vendor string: Mesa Project\n"
            "OpenGL core profile renderer: AMD Radeon AI PRO R9700\n"
            "Device #1:\n"
            "EGL vendor string: Mesa Project\n"
            f"OpenGL core profile renderer: {selected}\n"
        )
        return subprocess.CompletedProcess(argv, 2, stdout, "expected eglinfo stderr")

    return run


def test_preflight_binds_selected_egl_section_and_does_not_persist_host_env(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    child["QUALIFIER_TEST_SECRET"] = "do-not-persist"
    monkeypatch.setattr(qualifier.subprocess, "run", _mock_preflight_run())

    result = qualifier._run_rocm_egl_preflight(  # noqa: SLF001
        qualification_plan, child_env=child
    )
    assert result["status"] == "PASS_EXACT_ROCM_HIP_AND_EGL_R9700"
    assert result["egl_device_index"] == 1
    assert result["environment"] == (
        qualification_plan["execution_contract"]["environment"]
    )
    assert "QUALIFIER_TEST_SECRET" not in result["environment"]


def test_preflight_rejects_r9700_on_unselected_egl_device(
    qualification_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = qualifier._child_environment(qualification_plan)  # noqa: SLF001
    monkeypatch.setattr(
        qualifier.subprocess,
        "run",
        _mock_preflight_run(bad_selected_egl_device=True),
    )
    with pytest.raises(
        qualifier.GenesisRocmBackendQualificationError,
        match="EGL R9700",
    ):
        qualifier._run_rocm_egl_preflight(  # noqa: SLF001
            qualification_plan, child_env=child
        )


def test_execute_qualification_joins_first_scene_with_pilot_plan_binding(
    tmp_path: Path,
    qualification_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "qualification_plan.json"
    plan_path.write_text(json.dumps(qualification_plan, sort_keys=True))
    pilot_plan_binding = qualifier.pilot.file_binding(plan_path)
    plan_binding = qualifier._standard_binding(pilot_plan_binding)  # noqa: SLF001

    collection = tmp_path / "collection"
    collection.mkdir()
    reservation_path = collection / "reservation.json"
    reservation_path.write_text("{}\n")
    reservation = qualifier.pilot.file_binding(reservation_path)
    monkeypatch.setattr(
        qualifier,
        "_reserve_qualification",
        lambda **_kwargs: (collection, reservation),
    )
    monkeypatch.setattr(
        qualifier,
        "_child_environment",
        lambda _plan: dict(
            qualifier.plan_builder.rocm_execution_environment("qualification")
        ),
    )
    monkeypatch.setattr(
        qualifier,
        "_run_rocm_egl_preflight",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    counter = tmp_path / "vram"
    counter.write_text("0")
    monkeypatch.setattr(
        qualifier.collector,
        "_selected_gpu_memory_files_rocm",
        lambda _plan: (counter, counter, "0x1002", "0x7551"),
    )
    observed_order: list[int] = []

    def worker(_argv, *, scene, **_kwargs):
        index = int(scene["scene_index"])
        observed_order.append(index)
        return {
            "scene_index": index,
            "role": str(scene["role"]),
            "scene_id": str(scene["scene_id"]),
            "pid": 1000 + index,
            "parent_pid": 999,
            "process_group_id": 1000 + index,
            "fresh_process_group": True,
            "sys_executable": "/usr/bin/python3.12",
            "prelaunch_baseline_used_bytes": 0,
            "peak_selected_device_vram_bytes": 0,
            "selected_device_vram_cap_breached": False,
            "watchdog_timeout": False,
            "exit_code": 0,
            "elapsed_seconds": 80.0 if index == 12 else 70.0,
        }

    monkeypatch.setattr(qualifier, "_run_worker_with_watchdog", worker)
    monkeypatch.setattr(
        qualifier.predecessor,
        "_wait_for_vram_release_v2",
        lambda *_args, **_kwargs: {"status": "PASSED"},
    )
    observed_join_bindings: list[dict] = []

    def load(*, scene, plan_binding, **_kwargs):
        observed_join_bindings.append(dict(plan_binding))
        assert dict(plan_binding) == pilot_plan_binding
        role = str(scene["role"])
        counts = qualifier.predecessor._scene_expected_counts_v2(role)  # noqa: SLF001
        return (
            {"observed_counts": counts, "scene_metric": {"native_render_calls": 48}},
            {
                "path": f"scene_results/{int(scene['scene_index']):03d}.json",
                "file_sha256": "b" * 64,
                "byte_count": 1,
            },
        )

    monkeypatch.setattr(qualifier.predecessor, "_load_scene_result_v2", load)
    monkeypatch.setattr(
        qualifier,
        "_kernel_events_since",
        lambda _epoch: {
            "query_succeeded": True,
            "new_amdgpu_ring_timeout_or_reset_count": 0,
            "matching_lines_sha256": "c" * 64,
        },
    )
    output = tmp_path / "qualification_result.json"
    monkeypatch.setattr(qualifier, "QUALIFICATION_RESULT_PATH", output)
    monkeypatch.setattr(
        qualifier.pilot,
        "write_json_exclusive",
        lambda path, value: path.write_text(json.dumps(value)) or {},
    )

    result = qualifier.execute_qualification(
        {
            "collection_root": str(collection),
            "caps": qualifier.collector.EXPECTED_CAPS,
            "source_bindings": {},
        },
        authority_binding={"path": "/authority", "sha256": "d" * 64, "byte_count": 1},
        plan=qualification_plan,
        plan_binding=plan_binding,
    )

    assert observed_order == [12, 0]
    assert observed_join_bindings == [pilot_plan_binding, pilot_plan_binding]
    assert result["status"] == qualifier.QUALIFICATION_RESULT_STATUS
    assert result["timing_gate"]["projected_scientific_wall_seconds"] == 6020.0
    assert result["probe_output_scientific_reuse_authorized"] is False
    assert result["authorizes_retry_or_resume"] is False


def test_worker_argv_changes_only_entry_point() -> None:
    kwargs = {
        "scene_index": 12,
        "plan_path": Path("/tmp/plan"),
        "expected_plan_byte_count": 1,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority"),
        "expected_authority_byte_count": 2,
        "expected_authority_sha256": "b" * 64,
        "reservation_binding": {"byte_count": 3, "file_sha256": "c" * 64},
        "orchestrator_nonce": "d" * 64,
    }
    expected = qualifier._ROCM_WORKER_ARGV(**kwargs)  # noqa: SLF001
    actual = qualifier._worker_argv_qualification(**kwargs)  # noqa: SLF001
    assert actual[0] == expected[0]
    assert actual[2:] == expected[2:]
    assert Path(actual[1]).resolve() == Path(qualifier.__file__).resolve()
