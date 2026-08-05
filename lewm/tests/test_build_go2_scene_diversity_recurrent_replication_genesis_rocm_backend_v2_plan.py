from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_plan
    as builder,
)


def _frozen() -> dict:
    return copy.deepcopy(builder.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN)  # noqa: SLF001


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def scientific_plan(runtime_bindings: dict) -> dict:
    return builder.build_scientific_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


@pytest.fixture(scope="module")
def qualification_plan(runtime_bindings: dict) -> dict:
    return builder.build_qualification_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


def test_v2_is_fresh_and_changes_only_runtime_identity_contract(
    scientific_plan: dict,
) -> None:
    frozen = _frozen()
    assert scientific_plan["attempt_id"] == builder.DEFAULT_ATTEMPT_ID
    assert scientific_plan["output_root"] == str(
        builder.DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    )
    assert builder.DEFAULT_ATTEMPT_ROOT != (
        builder.predecessor.DEFAULT_ATTEMPT_ROOT
    )
    assert set(scientific_plan) == {*set(frozen), "successor_contract"}
    changed = {
        "attempt_id",
        "output_root",
        "runtime_bindings",
        "execution_contract",
    }
    assert all(
        scientific_plan[name] == frozen[name]
        for name in set(frozen) - changed
    )
    successor = scientific_plan["successor_contract"]
    assert successor["v1_runtime_payload_reuse_authorized"] is False
    assert (
        successor["v1_runtime_metadata_as_identity_evidence_authorized"]
        is False
    )
    assert successor["v1_qualification_terminal_review_binding"] == (
        builder.V1_QUALIFICATION_TERMINAL_REVIEW_BINDING
    )
    serialized = json.dumps(scientific_plan)
    assert "genesis_rocm_backend_v1_qualification/attempt_v1" not in serialized


def test_successor_runtime_version_declarations_match_frozen_v1() -> None:
    v1 = builder.predecessor._successor_contract(  # noqa: SLF001
        plan_role="scientific"
    )
    v2 = builder._successor_contract(plan_role="scientific")  # noqa: SLF001
    fields = {
        "genesis_world_version",
        "quadrants_version",
        "torch_version",
        "torchvision_version",
        "tensordict_version",
        "rsl_rl_version",
        "genesis_backend_symbol",
    }
    assert {name: v2[name] for name in fields} == {
        name: v1[name] for name in fields
    }


def test_graphics_preflight_explicitly_binds_unresolved_driver(
    qualification_plan: dict,
) -> None:
    graphics = qualification_plan["execution_contract"]["graphics_preflight"]
    target = qualification_plan["runtime_bindings"]["rocm_lld_executable"]
    driver = builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT
    assert graphics["rocm_lld_driver_entrypoint"] == str(driver)
    assert graphics["rocm_lld_driver_link_text"] == "lld"
    assert graphics["rocm_lld_resolved_target_path"] == target["path"]
    assert graphics["rocm_lld_version_stdout_prefix"] == "AMD LLD 20.0.0"
    assert graphics["rocm_lld_direct_target_invocation_forbidden"] is True
    assert driver.is_symlink()
    assert os.readlink(driver) == "lld"
    assert driver.resolve(strict=True) == Path(target["path"])


def test_actual_plan_interpreter_is_the_v2_bound_venv(
    scientific_plan: dict,
) -> None:
    interpreter = scientific_plan["execution_contract"][
        "python_invocation_path"
    ]
    completed = subprocess.run(
        [
            interpreter,
            "-I",
            "-B",
            "-c",
            (
                "import importlib.util,json,sys; "
                "print(json.dumps({'executable':sys.executable,"
                "'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
                "'torch_spec':importlib.util.find_spec('torch') is not None,"
                "'genesis_spec':importlib.util.find_spec('genesis') is not None}))"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0
    identity = json.loads(completed.stdout)
    assert identity["executable"] == str(builder.ROCM_PYTHON.absolute())
    assert identity["prefix"] == str(builder.ROCM_VENV.absolute())
    assert identity["base_prefix"] != identity["prefix"]
    assert identity["torch_spec"] is True
    assert identity["genesis_spec"] is True


def test_actual_ld_lld_driver_and_generic_target_are_distinguished(
    qualification_plan: dict,
) -> None:
    graphics = qualification_plan["execution_contract"]["graphics_preflight"]
    driver = graphics["rocm_lld_driver_entrypoint"]
    target = qualification_plan["runtime_bindings"]["rocm_lld_executable"][
        "path"
    ]
    driver_result = subprocess.run(
        [driver, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    target_result = subprocess.run(
        [target, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    assert driver_result.returncode == 0
    assert driver_result.stdout.startswith(
        graphics["rocm_lld_version_stdout_prefix"]
    )
    assert target_result.returncode != 0
    assert "generic driver" in target_result.stderr


def test_qualification_is_two_fresh_nonreusable_probes(
    qualification_plan: dict,
) -> None:
    successor = qualification_plan["successor_contract"]
    assert qualification_plan["attempt_id"] == (
        builder.QUALIFICATION_ATTEMPT_ID
    )
    assert successor["plan_role"] == "qualification"
    assert successor["qualification_scene_indices_in_order"] == [12, 0]
    assert successor["qualification_worker_watchdog_seconds"] == 300
    assert successor["probe_output_reuse_authorized"] is False
    assert successor["qualification_execution_authorized"] is False
    assert successor["scientific_execution_authorized"] is False
    assert qualification_plan["execution_contract"]["environment"] == (
        builder.rocm_execution_environment("qualification")
    )


def test_emitted_exact_plans_match_validated_builds(
    scientific_plan: dict, qualification_plan: dict
) -> None:
    assert json.loads(builder.DEFAULT_PLAN_OUTPUT.read_text()) == scientific_plan
    assert json.loads(builder.QUALIFICATION_PLAN_OUTPUT.read_text()) == (
        qualification_plan
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda plan: plan["execution_contract"]["graphics_preflight"].__setitem__(
            "rocm_lld_driver_entrypoint", "/opt/rocm-7.1.1/lib/llvm/bin/lld"
        ),
        lambda plan: plan["successor_contract"].__setitem__(
            "v1_runtime_payload_reuse_authorized", True
        ),
        lambda plan: plan["execution_contract"].__setitem__("seed", 7),
    ),
)
def test_unregistered_changes_are_rejected(
    scientific_plan: dict, mutation
) -> None:
    changed = copy.deepcopy(scientific_plan)
    mutation(changed)
    with pytest.raises(builder.SceneDiversityGenesisRocmV2PlanError):
        builder.validate_rocm_plan(
            changed,
            expected_attempt_id=builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=builder.DEFAULT_OUTPUT_ROOT,
            plan_role="scientific",
        )
