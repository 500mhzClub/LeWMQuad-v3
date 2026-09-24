from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_plan
    as builder,
)


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def plans(runtime_bindings: dict) -> tuple[dict, dict]:
    frozen = copy.deepcopy(
        builder.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return (
        builder.build_scientific_plan(
            frozen_plan=frozen, runtime_bindings=runtime_bindings
        ),
        builder.build_qualification_plan(
            frozen_plan=frozen, runtime_bindings=runtime_bindings
        ),
    )


def _v2_plan(role: str, runtime_bindings: dict) -> dict:
    predecessor = builder.predecessor
    if role == "scientific":
        attempt_id = predecessor.DEFAULT_ATTEMPT_ID
        output_root = predecessor.DEFAULT_OUTPUT_ROOT
    else:
        attempt_id = predecessor.QUALIFICATION_ATTEMPT_ID
        output_root = predecessor.QUALIFICATION_OUTPUT_ROOT
    return predecessor._expected_rocm_plan(  # noqa: SLF001
        attempt_id=attempt_id,
        output_root=output_root,
        plan_role=role,
        runtime_bindings=runtime_bindings,
    )


def test_exact_v2_terminal_review_is_the_only_v2_failure_witness() -> None:
    binding = builder.v2_qualification_terminal_review_binding()
    assert binding == {
        "path": str(builder.V2_QUALIFICATION_TERMINAL_REVIEW.resolve()),
        "file_sha256": (
            "166aec87b6e61d62116069a12472b768c3ff462c09cf1e6088af62ab7397dd0e"
        ),
        "byte_count": 16_198,
    }


@pytest.mark.parametrize("index,role", [(0, "scientific"), (1, "qualification")])
def test_v2_to_v3_plan_delta_is_exact_home_identity_only(
    plans: tuple[dict, dict],
    runtime_bindings: dict,
    index: int,
    role: str,
) -> None:
    v3 = plans[index]
    v2 = _v2_plan(role, runtime_bindings)
    assert set(v3) == set(v2)
    for key in set(v2) - {
        "attempt_id",
        "output_root",
        "execution_contract",
        "successor_contract",
    }:
        assert v3[key] == v2[key]

    v2_execution = copy.deepcopy(v2["execution_contract"])
    v3_execution = copy.deepcopy(v3["execution_contract"])
    v3_environment = v3_execution.pop("environment")
    v2_environment = v2_execution.pop("environment")
    assert v3_execution == v2_execution
    assert set(v3_environment) == set(v2_environment) | {"HOME"}
    assert v3_environment["HOME"] == builder.REQUIRED_HOST_HOME
    for key in set(v2_environment) - {"GS_CACHE_FILE_PATH"}:
        assert v3_environment[key] == v2_environment[key]
    assert "backend_v3" in v3_environment["GS_CACHE_FILE_PATH"]
    assert "backend_v2" in v2_environment["GS_CACHE_FILE_PATH"]

    v2_successor = v2["successor_contract"]
    v3_successor = v3["successor_contract"]
    invariant_fields = {
        "frozen_vulkan_scientific_plan_binding",
        "frozen_cpu_scientific_plan_binding",
        "cpu_qualification_terminal_review_binding",
        "rocm_lld_driver_entrypoint",
        "rocm_lld_direct_target_invocation_forbidden",
        "genesis_world_version",
        "quadrants_version",
        "torch_version",
        "torchvision_version",
        "tensordict_version",
        "rsl_rl_version",
        "genesis_backend_symbol",
        "qualification_scene_indices_in_order",
        "qualification_worker_watchdog_seconds",
        "qualification_timing_gate",
        "qualification_execution_authorized",
        "scientific_execution_authorized",
        "probe_output_reuse_authorized",
        "plan_role",
    }
    assert {key: v3_successor[key] for key in invariant_fields} == {
        key: v2_successor[key] for key in invariant_fields
    }
    assert v3_successor["required_host_home"] == builder.REQUIRED_HOST_HOME
    assert v3_successor["v2_runtime_payload_reuse_authorized"] is False
    assert (
        v3_successor["v2_runtime_metadata_as_identity_evidence_authorized"]
        is False
    )
    assert v3_successor["v2_qualification_terminal_review_binding"] == (
        builder.v2_qualification_terminal_review_binding()
    )


def test_home_is_literal_not_ambient_and_other_identity_keys_stay_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp/ambient-home-must-not-win")
    monkeypatch.setenv("USER", "ambient-user")
    environment = builder.rocm_execution_environment("scientific")
    assert environment["HOME"] == "/home/andrewknowles"
    assert all(key not in environment for key in ("USER", "LOGNAME", "LANG"))


@pytest.mark.parametrize("index,role", [(0, "scientific"), (1, "qualification")])
def test_plan_validator_rejects_missing_or_mutated_home(
    plans: tuple[dict, dict], index: int, role: str
) -> None:
    attempt_id = (
        builder.DEFAULT_ATTEMPT_ID
        if role == "scientific"
        else builder.QUALIFICATION_ATTEMPT_ID
    )
    output_root = (
        builder.DEFAULT_OUTPUT_ROOT
        if role == "scientific"
        else builder.QUALIFICATION_OUTPUT_ROOT
    )
    for mutation in (None, "/tmp/wrong-home"):
        changed = copy.deepcopy(plans[index])
        if mutation is None:
            changed["execution_contract"]["environment"].pop("HOME")
        else:
            changed["execution_contract"]["environment"]["HOME"] = mutation
        with pytest.raises(
            builder.SceneDiversityGenesisRocmV3PlanError,
            match="exact driver successor overlay",
        ):
            builder.validate_rocm_plan(
                changed,
                expected_attempt_id=attempt_id,
                expected_output_root=output_root,
                plan_role=role,
            )


def test_fresh_v3_roots_and_role_local_caches(plans: tuple[dict, dict]) -> None:
    science, qualification = plans
    assert not builder.DEFAULT_ATTEMPT_ROOT.exists()
    assert not builder.QUALIFICATION_ATTEMPT_ROOT.exists()
    assert science["output_root"] != qualification["output_root"]
    assert (
        science["execution_contract"]["environment"]["GS_CACHE_FILE_PATH"]
        != qualification["execution_contract"]["environment"][
            "GS_CACHE_FILE_PATH"
        ]
    )
    serialized = json.dumps(plans)
    assert ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2" not in serialized


def test_plan_outputs_are_fresh_metadata_targets() -> None:
    for path in (builder.DEFAULT_PLAN_OUTPUT, builder.QUALIFICATION_PLAN_OUTPUT):
        assert Path(path).parent == builder.REPO_ROOT / "docs"
