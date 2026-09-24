from __future__ import annotations

import json
from pathlib import Path

from scripts import (
    collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as collector,
)


def test_cpu_collector_preserves_policy_caps_and_expands_only_science_authority() -> None:
    assert collector.EXPECTED_CAPS is collector.predecessor.EXPECTED_CAPS
    assert collector.EXPECTED_COUNTS is collector.predecessor.EXPECTED_COUNTS
    assert collector.EXPECTED_PERMISSIONS is collector.predecessor.EXPECTED_PERMISSIONS
    assert collector.SCENE_COUNT == 64
    assert set(collector.AUTHORITY_FIELDS) == set(collector.predecessor.AUTHORITY_FIELDS) | {
        "qualification_result_binding"
    }
    assert collector.PROCESS_RESET_EQUIVALENCE_AUDIT_CPU["execution_backend"] == "cpu"
    assert (
        collector.PROCESS_RESET_EQUIVALENCE_AUDIT_CPU[
            "physics_numerics_may_differ_from_vulkan"
        ]
        is True
    )


def test_runtime_validator_accepts_both_exact_cpu_plan_identities() -> None:
    science = json.loads(collector.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())
    assert collector._validate_cpu_plan_runtime(science) == science  # noqa: SLF001

    original = collector.ATTEMPT_ID
    collector.ATTEMPT_ID = collector.plan_builder.QUALIFICATION_ATTEMPT_ID
    try:
        qualification = json.loads(
            collector.plan_builder.QUALIFICATION_PLAN_OUTPUT.read_text()
        )
        assert collector._validate_cpu_plan_runtime(qualification) == qualification  # noqa: SLF001
    finally:
        collector.ATTEMPT_ID = original


def test_plan_first_initializer_changes_only_backend(monkeypatch) -> None:
    binding = {"path": "/manifest", "file_sha256": "a" * 64, "byte_count": 1}
    plan = {
        "states": [
            {
                "scene_id": "large_enclosed_maze_8a6599d5327d",
                "state_id": "state-0",
                "scene_manifest_binding": binding,
            }
        ],
        "execution_contract": {"backend": "cpu"},
    }
    monkeypatch.setattr(
        collector.pilot,
        "read_bound_json",
        lambda *_args, **_kwargs: (
            {"physics_seed": collector.PLAN_FIRST_PHYSICS_SEED},
            binding,
        ),
    )
    observed = {}
    monkeypatch.setattr(
        collector.predecessor,
        "_initialize_genesis_v2",
        lambda **kwargs: observed.update(kwargs),
    )

    receipt = collector._initialize_from_plan_first_scene_cpu(plan=plan)  # noqa: SLF001

    assert observed == {
        "backend": "cpu",
        "seed": collector.PLAN_FIRST_PHYSICS_SEED,
    }
    assert receipt["backend"] == "cpu"
    assert receipt["effective_genesis_seed"] == (
        collector.PLAN_FIRST_EFFECTIVE_GENESIS_SEED
    )


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
    actual = collector._worker_argv_cpu(**kwargs)  # noqa: SLF001
    assert actual[0] == expected[0]
    assert actual[2:] == expected[2:]
    assert Path(actual[1]).resolve() == Path(collector.__file__).resolve()


def test_scoped_cpu_overlay_restores_shared_validator_and_environment() -> None:
    overrides = collector._configuration_overrides_cpu()  # noqa: SLF001
    originals = {name: getattr(collector.predecessor, name) for name in overrides}
    validate = collector.pilot.validate_plan
    environment = collector.pilot.EXECUTION_ENVIRONMENT

    with collector._configured_predecessor_collector_cpu():  # noqa: SLF001
        assert collector.predecessor.ATTEMPT_ID == collector.ATTEMPT_ID
        assert collector.predecessor._initialize_from_plan_first_scene_v2 is (  # noqa: SLF001
            collector._initialize_from_plan_first_scene_cpu  # noqa: SLF001
        )
        assert collector.pilot.EXECUTION_ENVIRONMENT["GS_BACKEND"] == "cpu"
        assert collector.pilot.validate_plan is collector._validate_cpu_plan_runtime  # noqa: SLF001

    assert collector.pilot.validate_plan is validate
    assert collector.pilot.EXECUTION_ENVIRONMENT is environment
    assert all(
        getattr(collector.predecessor, name) is value
        for name, value in originals.items()
    )
