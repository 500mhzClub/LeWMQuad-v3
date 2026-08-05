from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from scripts import (
    collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1
    as collector,
)


def _frozen() -> dict:
    return json.loads(collector.plan_builder.FROZEN_V1_EXACT_PLAN.read_text())


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return collector.plan_builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def scientific_plan(runtime_bindings: dict) -> dict:
    return collector.plan_builder.build_scientific_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


@pytest.fixture(scope="module")
def qualification_plan(runtime_bindings: dict) -> dict:
    return collector.plan_builder.build_qualification_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


def test_rocm_collector_preserves_science_and_selects_safe_contact_route() -> None:
    assert collector.EXPECTED_CAPS is collector.predecessor.EXPECTED_CAPS
    assert collector.EXPECTED_COUNTS is collector.predecessor.EXPECTED_COUNTS
    assert collector.EXPECTED_PERMISSIONS is collector.predecessor.EXPECTED_PERMISSIONS
    assert collector.SCENE_COUNT == 64
    assert set(collector.AUTHORITY_FIELDS) == set(
        collector.predecessor.AUTHORITY_FIELDS
    ) | {
        "qualification_result_binding",
        "predecessor_cpu_terminal_review_binding",
    }
    assert collector.CONTACT_FORCE_ROUTE_AUDIT == {
        "schema": (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
            "backend_v1_contact_force_route_source_audit_v1"
        ),
        "known_bad_api": "robot.get_links_net_contact_force",
        "selected_entrypoint": "RolloutRunner.execute_requested_block",
        "selected_entrypoint_emits_per_tick_records": False,
        "selected_entrypoint_calls_extract_foot_contacts": False,
        "rollout_config_foot_contact_source": "zero",
        "known_bad_api_reachable": False,
    }


def test_runtime_validator_accepts_both_exact_rocm_plan_roles(
    scientific_plan: dict, qualification_plan: dict
) -> None:
    assert collector._validate_rocm_plan_runtime(scientific_plan) == scientific_plan  # noqa: SLF001

    original = collector.ATTEMPT_ID
    collector.ATTEMPT_ID = collector.plan_builder.QUALIFICATION_ATTEMPT_ID
    try:
        assert collector._validate_rocm_plan_runtime(qualification_plan) == (  # noqa: SLF001
            qualification_plan
        )
    finally:
        collector.ATTEMPT_ID = original


def test_scoped_overlay_is_role_local_sanitized_and_fully_restored(
    scientific_plan: dict,
) -> None:
    del scientific_plan
    overrides = collector._configuration_overrides_rocm()  # noqa: SLF001
    originals = {name: getattr(collector.predecessor, name) for name in overrides}
    validate = collector.pilot.validate_plan
    environment = collector.pilot.EXECUTION_ENVIRONMENT
    graphics = collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION
    build_runner = collector.kernel._build_rollout_runner  # noqa: SLF001
    memory_selector = (
        collector.predecessor.calibration_supervisor._selected_gpu_memory_files  # noqa: SLF001
    )
    selectors = collector.kernel._SANITIZED_SELECTOR_KEYS  # noqa: SLF001

    with collector._configured_predecessor_collector_rocm():  # noqa: SLF001
        assert collector.predecessor.ATTEMPT_ID == collector.ATTEMPT_ID
        assert collector.pilot.validate_plan is collector._validate_rocm_plan_runtime  # noqa: SLF001
        assert collector.pilot.EXECUTION_ENVIRONMENT == (
            collector.plan_builder.rocm_execution_environment("scientific")
        )
        assert collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION == (
            collector.plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
        )
        assert collector.kernel._build_rollout_runner is (  # noqa: SLF001
            collector._build_rollout_runner_rocm  # noqa: SLF001
        )
        assert collector.ROCM_ADDITIONAL_SANITIZED_KEYS <= set(
            collector.kernel._SANITIZED_SELECTOR_KEYS  # noqa: SLF001
        )
        assert {
            "GS_CACHE_FILE_PATH",
            "GS_TORCH_FORCE_CPU_DEVICE",
            "QD_ARCH",
            "QD_ENABLE_AMDGPU",
            "QD_PERFDISPATCH_FORCE",
            "ROCM_PATH",
        } <= collector.ROCM_ADDITIONAL_SANITIZED_KEYS

    assert collector.pilot.validate_plan is validate
    assert collector.pilot.EXECUTION_ENVIRONMENT is environment
    assert collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION is graphics
    assert collector.kernel._build_rollout_runner is build_runner  # noqa: SLF001
    assert (
        collector.predecessor.calibration_supervisor._selected_gpu_memory_files  # noqa: SLF001
        is memory_selector
    )
    assert collector.kernel._SANITIZED_SELECTOR_KEYS is selectors  # noqa: SLF001
    assert all(
        getattr(collector.predecessor, name) is value
        for name, value in originals.items()
    )


def test_nested_qualification_overlay_uses_immutable_validator(
    qualification_plan: dict,
) -> None:
    original = collector.ATTEMPT_ID
    collector.ATTEMPT_ID = collector.plan_builder.QUALIFICATION_ATTEMPT_ID
    try:
        with collector._configured_predecessor_collector_rocm():  # noqa: SLF001
            assert collector.pilot.validate_plan(qualification_plan) == (
                qualification_plan
            )
            assert collector.pilot.EXECUTION_ENVIRONMENT == (
                collector.plan_builder.rocm_execution_environment("qualification")
            )
    finally:
        collector.ATTEMPT_ID = original


def test_plan_first_initializer_uses_amdgpu_and_rejects_hsa_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {"path": "/manifest", "file_sha256": "a" * 64, "byte_count": 1}
    plan = {
        "states": [
            {
                "scene_id": "large_enclosed_maze_8a6599d5327d",
                "state_id": "state-0",
                "scene_manifest_binding": binding,
            }
        ],
        "execution_contract": {"backend": "amdgpu"},
    }
    monkeypatch.setattr(
        collector.pilot,
        "read_bound_json",
        lambda *_args, **_kwargs: (
            {"physics_seed": collector.PLAN_FIRST_PHYSICS_SEED},
            binding,
        ),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        collector.predecessor,
        "_initialize_genesis_v2",
        lambda **kwargs: observed.update(kwargs),
    )
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)

    receipt = collector._initialize_from_plan_first_scene_rocm(plan=plan)  # noqa: SLF001
    assert observed == {
        "backend": "amdgpu",
        "seed": collector.PLAN_FIRST_PHYSICS_SEED,
    }
    assert receipt["backend_api"] == "gs.amdgpu"
    assert receipt["hsa_override_gfx_version_present"] is False

    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    with pytest.raises(
        collector.SceneProcessCollectionError,
        match="must remain absent",
    ):
        collector._initialize_from_plan_first_scene_rocm(plan=plan)  # noqa: SLF001


def test_rollout_runner_forces_zero_contact_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []

    def rollout_config(**kwargs):
        observed.append(dict(kwargs))
        return kwargs

    def original_builder(**kwargs):
        return kwargs["runtime"]["RolloutConfig"](horizon_steps=3)

    monkeypatch.setattr(
        collector, "_ORIGINAL_BUILD_ROLLOUT_RUNNER", original_builder
    )
    result = collector._build_rollout_runner_rocm(  # noqa: SLF001
        plan={},
        runtime={"RolloutConfig": rollout_config},
        platform=object(),
        build=object(),
        registry=object(),
    )
    assert result == {"foot_contact_source": "zero", "horizon_steps": 3}
    assert observed == [{"foot_contact_source": "zero", "horizon_steps": 3}]


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
    expected = collector._ORIGINAL_WORKER_ARGV(**kwargs)  # noqa: SLF001
    actual = collector._worker_argv_rocm(**kwargs)  # noqa: SLF001
    assert actual[0] == expected[0]
    assert actual[2:] == expected[2:]
    assert Path(actual[1]).resolve() == Path(collector.__file__).resolve()


def test_selected_gpu_memory_files_bind_exact_r9700(
    tmp_path: Path, scientific_plan: dict
) -> None:
    card = tmp_path / "card7/device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002\n")
    (card / "device").write_text("0x7551\n")
    used = card / "mem_info_vram_used"
    total = card / "mem_info_vram_total"
    used.write_text("0\n")
    total.write_text("1\n")

    assert collector._selected_gpu_memory_files_rocm(  # noqa: SLF001
        scientific_plan, drm_root=tmp_path
    ) == (used, total, "0x1002", "0x7551")


def test_contact_audit_matches_selected_source_route() -> None:
    source = (
        collector.plan_builder.REPO_ROOT
        / "lewm_genesis/lewm_genesis/rollout.py"
    ).read_text()
    execute = source[source.index("    def execute_requested_block(") :]
    execute = execute[: execute.index("\n    def ", 1)]
    assert "_extract_foot_contacts" not in execute
    assert "get_links_net_contact_force" not in execute
    assert os.path.basename(collector.__file__).endswith("backend_v1.py")
