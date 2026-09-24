from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_plan as builder


def _changed_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changed: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                changed.add(path)
            else:
                changed.update(_changed_paths(left[key], right[key], path))
        return changed
    return {prefix} if left != right else set()


def test_scientific_delta_is_only_identity_root_and_contract() -> None:
    qualification = json.loads(builder.QUALIFICATION_PLAN.read_bytes())
    science = builder.build_scientific_plan()
    assert _changed_paths(qualification, science) >= {
        "attempt_id",
        "output_root",
    }
    assert {
        path
        for path in _changed_paths(qualification, science)
        if not path.startswith("successor_contract")
    } == {"attempt_id", "output_root"}

    normalized = copy.deepcopy(science)
    normalized["attempt_id"] = qualification["attempt_id"]
    normalized["output_root"] = qualification["output_root"]
    normalized["successor_contract"] = qualification["successor_contract"]
    assert normalized == qualification


def test_decision_and_review_release_exact_science_without_payload_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    forbidden_names = {
        "qualification_result.json",
        "terminal.json",
        "reservation.json",
    }

    def guarded(path: Path) -> bytes:
        if path.name in forbidden_names or path.suffix.lower() == ".png":
            raise AssertionError(f"qualification payload opened: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert builder.qualification_pass_decision_binding()["sha256"] == (
        "6eac9ff6458092d6284011934c865b45519cbd2ff14c0b2da3a515bbf4a6a299"
    )
    assert builder.independent_qualification_terminal_review_binding()[
        "sha256"
    ] == "bc13e880fc348515384b906ea7dce32bc7587df089f7aa7f6b630d71e87ce31d"
    science = builder.build_scientific_plan()
    assert science["successor_contract"][
        "qualification_runtime_payload_opened_by_builder"
    ] is False


def test_counts_environment_protocol_and_gates_are_frozen() -> None:
    science = builder.build_scientific_plan()
    assert science["expected_counts"] == {
        "actions": 9,
        "candidate_branches": 2304,
        "context_frames": 768,
        "roles": {"eval": 128, "train": 128},
        "scenes": 64,
        "sentinel_branches": 0,
        "states": 256,
        "target_frames": 2304,
        "total_branches": 2304,
    }
    assert science["branch_mechanism"] == "parallel_lockstep_envs_no_restore"
    environment = science["execution_contract"]["environment"]
    assert len(environment) == 11
    assert environment["HOME"] == "/home/andrewknowles"
    assert environment["PATH"] == "/usr/bin:/bin"
    protocol = science["successor_contract"]["scientific_protocol"]
    assert protocol["learned_arms_in_order"] == [
        "no_vision_recurrent_direct",
        "visual_recurrent_direct",
    ]
    assert protocol["live_control_arm"] == "task_action_only"
    assert protocol["model_seeds"] == [2026080411, 2026080412, 2026080413]
    assert protocol["shared_sampler_seed"] == 2026080414
    assert protocol["updates"] == 800
    assert protocol["fixed_gates"] == builder.SCIENTIFIC_PROTOCOL["fixed_gates"]


def test_validator_rejects_data_or_protocol_mutation() -> None:
    science = builder.build_scientific_plan()
    changed = copy.deepcopy(science)
    changed["expected_counts"]["states"] += 1
    with pytest.raises(
        builder.CpuFlatDevelopmentV3ScientificPlanError,
        match="changed beyond identity/root/release metadata",
    ):
        builder.validate_scientific_plan(changed)

    changed = copy.deepcopy(science)
    changed["successor_contract"]["scientific_protocol"]["updates"] = 801
    with pytest.raises(builder.CpuFlatDevelopmentV3ScientificPlanError):
        builder.validate_scientific_plan(changed)


def test_existing_scientific_root_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "already_reserved"
    existing.mkdir()
    monkeypatch.setattr(builder, "SCIENTIFIC_ATTEMPT_ROOT", existing)
    with pytest.raises(
        builder.CpuFlatDevelopmentV3ScientificPlanError,
        match="fresh scientific root changed",
    ):
        builder.build_scientific_plan()


def test_cli_emits_scientific_plan_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "science.json"
    assert builder.main(["--scientific-plan-output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert output.is_file()
    assert summary["scientific_plan"] is not None
    assert summary["qualification_payload_reused"] is False
    assert summary["scientific_execution_authorized_by_plan_builder"] is False
