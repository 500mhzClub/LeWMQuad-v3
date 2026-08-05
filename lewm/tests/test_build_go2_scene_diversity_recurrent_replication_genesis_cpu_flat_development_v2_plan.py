from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v2_plan as builder


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


def test_recursive_delta_is_only_identity_root_contract_and_home() -> None:
    predecessor = json.loads(builder.V1_QUALIFICATION_PLAN.read_bytes())
    candidate = builder.build_qualification_plan()
    changed = _changed_paths(predecessor, candidate)
    outside_contract = {
        path for path in changed if not path.startswith("successor_contract")
    }
    assert outside_contract == {
        "attempt_id",
        "output_root",
        "execution_contract.environment.HOME",
    }

    normalized = copy.deepcopy(candidate)
    normalized["attempt_id"] = predecessor["attempt_id"]
    normalized["output_root"] = predecessor["output_root"]
    normalized["execution_contract"]["environment"].pop("HOME")
    normalized["successor_contract"] = predecessor["successor_contract"]
    assert normalized == predecessor


def test_exact_ten_key_environment_and_frozen_gates() -> None:
    predecessor = json.loads(builder.V1_QUALIFICATION_PLAN.read_bytes())
    candidate = builder.build_qualification_plan()
    environment = candidate["execution_contract"]["environment"]
    assert len(environment) == 10
    assert environment == {
        **predecessor["execution_contract"]["environment"],
        "HOME": "/home/andrewknowles",
    }
    assert candidate["branch_mechanism"] == "parallel_lockstep_envs_no_restore"
    assert candidate["states"] == predecessor["states"]
    assert candidate["history_blocks"] == predecessor["history_blocks"]
    assert candidate["action_catalog"] == predecessor["action_catalog"]
    assert candidate["expected_counts"] == predecessor["expected_counts"]
    contract = candidate["successor_contract"]
    assert contract["qualification_scene_indices_in_order"] == [12, 0]
    assert contract["qualification_worker_watchdog_seconds"] == 300
    assert contract["selected_device_vram_ceiling_bytes"] == 16_977_405_952
    assert contract["probe_output_reuse_authorized"] is False
    assert contract["scientific_plan_created"] is False


def test_validator_rejects_any_additional_environment_or_science_delta() -> None:
    candidate = builder.build_qualification_plan()
    changed = copy.deepcopy(candidate)
    changed["execution_contract"]["environment"]["LANG"] = "C.UTF-8"
    with pytest.raises(
        builder.CpuFlatDevelopmentV2PlanError,
        match="changed beyond fresh identity/root/contracts and fixed HOME",
    ):
        builder.validate_qualification_plan(changed)

    changed = copy.deepcopy(candidate)
    changed["expected_counts"]["states"] += 1
    with pytest.raises(builder.CpuFlatDevelopmentV2PlanError):
        builder.validate_qualification_plan(changed)


def test_existing_attempt_root_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "already_reserved"
    existing.mkdir()
    monkeypatch.setattr(builder, "QUALIFICATION_ATTEMPT_ROOT", existing)
    with pytest.raises(
        builder.CpuFlatDevelopmentV2PlanError,
        match="fresh V2 qualification/scientific roots changed",
    ):
        builder.build_qualification_plan()


def test_cli_emits_qualification_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "qualification.json"
    assert builder.main(["--qualification-plan-output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert output.is_file()
    assert summary["scientific_plan"] is None
    assert summary["scientific_plan_release_deferred_until_qualification_pass"]
    assert not hasattr(builder, "DEFAULT_PLAN_OUTPUT")
