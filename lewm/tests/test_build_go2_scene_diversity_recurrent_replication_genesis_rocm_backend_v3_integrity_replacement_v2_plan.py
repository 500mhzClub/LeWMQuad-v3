from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan
    as builder,
)


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def plans(runtime_bindings: dict) -> tuple[dict, dict]:
    frozen = copy.deepcopy(
        builder.predecessor.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return (
        builder.build_scientific_plan(
            frozen_plan=frozen, runtime_bindings=runtime_bindings
        ),
        builder.build_qualification_plan(
            frozen_plan=frozen, runtime_bindings=runtime_bindings
        ),
    )


def _v3_plan(role: str, runtime_bindings: dict) -> dict:
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


def test_exact_replacement_v1_terminal_review_is_the_only_new_failure_witness() -> None:
    binding = builder.replacement_v1_qualification_terminal_review_binding()
    assert binding == {
        "path": str(
            builder.REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW.resolve()
        ),
        "file_sha256": (
            "45bcd329d778dc9cba71882b398aa25943600dc160436bdba6b919868418f6fa"
        ),
        "byte_count": 10_267,
    }


@pytest.mark.parametrize("index,role", [(0, "scientific"), (1, "qualification")])
def test_v3_to_replacement_plan_delta_is_exact_identity_cache_and_witness_only(
    plans: tuple[dict, dict],
    runtime_bindings: dict,
    index: int,
    role: str,
) -> None:
    replacement = plans[index]
    v3 = _v3_plan(role, runtime_bindings)
    assert set(replacement) == set(v3)
    for key in set(v3) - {
        "attempt_id",
        "output_root",
        "execution_contract",
        "successor_contract",
    }:
        assert replacement[key] == v3[key]

    v3_execution = copy.deepcopy(v3["execution_contract"])
    replacement_execution = copy.deepcopy(replacement["execution_contract"])
    v3_environment = v3_execution.pop("environment")
    replacement_environment = replacement_execution.pop("environment")
    assert replacement_execution == v3_execution
    assert set(replacement_environment) == set(v3_environment)
    assert len(replacement_environment) == 17
    for key in set(v3_environment) - {"GS_CACHE_FILE_PATH"}:
        assert replacement_environment[key] == v3_environment[key]
    assert replacement_environment["HOME"] == builder.REQUIRED_HOST_HOME
    assert "integrity_replacement_v2" in replacement_environment[
        "GS_CACHE_FILE_PATH"
    ]

    v3_successor = v3["successor_contract"]
    replacement_successor = replacement["successor_contract"]
    added = {
        "replacement_v1_qualification_terminal_review_binding",
        "replacement_v1_authority_or_command_reuse_authorized",
        "replacement_v1_runtime_payload_reuse_authorized",
        "replacement_v1_runtime_metadata_as_identity_evidence_authorized",
    }
    assert set(replacement_successor) == set(v3_successor) | added
    for key in set(v3_successor) - {"schema", "material_infrastructure_hypothesis"}:
        assert replacement_successor[key] == v3_successor[key]
    assert replacement_successor["required_host_home"] == builder.REQUIRED_HOST_HOME
    assert (
        replacement_successor["replacement_v1_runtime_payload_reuse_authorized"]
        is False
    )
    assert (
        replacement_successor[
            "replacement_v1_runtime_metadata_as_identity_evidence_authorized"
        ]
        is False
    )
    assert replacement_successor[
        "replacement_v1_qualification_terminal_review_binding"
    ] == (
        builder.replacement_v1_qualification_terminal_review_binding()
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
            builder.SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError,
            match="exact driver successor overlay",
        ):
            builder.validate_rocm_plan(
                changed,
                expected_attempt_id=attempt_id,
                expected_output_root=output_root,
                plan_role=role,
            )


def test_fresh_replacement_roots_and_role_local_caches(
    plans: tuple[dict, dict]
) -> None:
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
    assert (
        ".generated/dev/go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v3/attempt_v1"
    ) not in serialized


def test_plan_outputs_are_fresh_metadata_targets() -> None:
    for path in (builder.DEFAULT_PLAN_OUTPUT, builder.QUALIFICATION_PLAN_OUTPUT):
        assert Path(path).parent == builder.REPO_ROOT / "docs"
