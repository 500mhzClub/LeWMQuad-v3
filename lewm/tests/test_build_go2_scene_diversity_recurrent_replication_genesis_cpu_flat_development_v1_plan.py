from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v1_plan as builder


def test_reviewed_cpu_pair_is_one_frozen_scientific_payload() -> None:
    qualification = builder.validate_reviewed_cpu_delta()
    science = json.loads(builder.FROZEN_CPU_SCIENTIFIC_PLAN.read_bytes())
    qualification["attempt_id"] = builder.FROZEN_CPU_SCIENTIFIC_ATTEMPT_ID
    qualification["output_root"] = str(
        builder.FROZEN_CPU_SCIENTIFIC_OUTPUT_ROOT.resolve(strict=False)
    )
    assert qualification == science
    assert science["branch_mechanism"] == "parallel_lockstep_envs_no_restore"


def test_qualification_plan_freezes_cpu_flat_contract() -> None:
    plan = builder.build_qualification_plan()
    contract = plan["successor_contract"]
    assert plan["attempt_id"] == builder.QUALIFICATION_ATTEMPT_ID
    assert plan["output_root"] == str(
        builder.QUALIFICATION_OUTPUT_ROOT.resolve(strict=False)
    )
    assert plan["execution_contract"]["backend"] == "cpu"
    assert (
        plan["execution_contract"]["environment"]
        == builder.CPU_EXECUTION_ENVIRONMENT
    )
    assert plan["branch_mechanism"] == "parallel_lockstep_envs_no_restore"
    assert contract["qualification_scene_indices_in_order"] == [12, 0]
    assert contract["qualification_fresh_process_groups"] == 2
    assert contract["branches_per_worker"] == 36
    assert contract["qualification_worker_watchdog_seconds"] == 300
    assert contract["selected_device_vram_ceiling_bytes"] == 16_977_405_952
    assert contract["exact_nine_lane_state_group_equality_required"] is True
    assert contract["numerical_tolerance_relaxation_authorized"] is False
    assert contract["probe_output_reuse_authorized"] is False
    assert contract["scientific_plan_created"] is False
    assert contract["scientific_execution_authorized"] is False


def test_qualification_validator_rejects_science_or_gate_mutation() -> None:
    plan = builder.build_qualification_plan()
    changed = copy.deepcopy(plan)
    changed["expected_counts"]["states"] += 1
    with pytest.raises(
        builder.CpuFlatDevelopmentPlanError,
        match="changed beyond identity/root/flat metadata",
    ):
        builder.validate_qualification_plan(changed)


def test_cli_emits_only_qualification_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "qualification.json"
    assert builder.main(["--qualification-plan-output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert output.is_file()
    assert summary["scientific_plan"] is None
    assert summary["scientific_plan_release_deferred_until_qualification_pass"]
    assert not hasattr(builder, "DEFAULT_PLAN_OUTPUT")


def test_builder_has_no_project_module_overlay_writes() -> None:
    tree = ast.parse(Path(builder.__file__).read_text(encoding="utf-8"))
    forbidden_calls = {"setattr", "delattr"}
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden_calls
    ]
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
