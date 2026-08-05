from __future__ import annotations

import copy
import json

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan as builder,
)


def _frozen() -> dict:
    return json.loads(builder.FROZEN_V1_EXACT_PLAN.read_text())


@pytest.mark.parametrize(
    ("path", "attempt_id", "output_root"),
    (
        (builder.DEFAULT_PLAN_OUTPUT, builder.DEFAULT_ATTEMPT_ID, builder.DEFAULT_OUTPUT_ROOT),
        (
            builder.QUALIFICATION_PLAN_OUTPUT,
            builder.QUALIFICATION_ATTEMPT_ID,
            builder.QUALIFICATION_OUTPUT_ROOT,
        ),
    ),
)
def test_cpu_plans_have_exactly_four_allowed_differences(
    path, attempt_id, output_root
) -> None:
    frozen = _frozen()
    plan = json.loads(path.read_text())
    expected = copy.deepcopy(frozen)
    expected["attempt_id"] = attempt_id
    expected["output_root"] = str(output_root.resolve())
    expected["execution_contract"]["backend"] = "cpu"
    expected["execution_contract"]["environment"]["GS_BACKEND"] = "cpu"

    assert plan == expected
    assert builder.validate_cpu_plan(
        plan, expected_attempt_id=attempt_id, expected_output_root=output_root
    ) == plan
    assert plan["execution_contract"]["python_invocation_path"] == frozen[
        "execution_contract"
    ]["python_invocation_path"]
    assert plan["execution_contract"]["graphics_preflight"] == frozen[
        "execution_contract"
    ]["graphics_preflight"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda plan: plan["execution_contract"].__setitem__("seed", 7),
        lambda plan: plan["execution_contract"]["environment"].__setitem__(
            "EGL_DEVICE_ID", "0"
        ),
        lambda plan: plan.__setitem__("states_per_scene", 5),
    ),
)
def test_any_additional_plan_change_is_rejected(mutation) -> None:
    plan = json.loads(builder.DEFAULT_PLAN_OUTPUT.read_text())
    mutation(plan)
    with pytest.raises(
        builder.SceneDiversityCpuBackendPlanError,
        match="changed|binding|exact state",
    ):
        builder.validate_cpu_plan(
            plan,
            expected_attempt_id=builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=builder.DEFAULT_OUTPUT_ROOT,
        )


def test_cpu_environment_changes_only_gs_backend() -> None:
    expected = dict(builder.pilot.EXECUTION_ENVIRONMENT)
    expected["GS_BACKEND"] = "cpu"
    assert builder.CPU_EXECUTION_ENVIRONMENT == expected
