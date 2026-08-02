from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from lewm.tests.test_go2_world_model_counterfactual_pilot_v1 import _smoke_plan


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = (
    ROOT / "scripts/build_go2_world_model_counterfactual_calibration_plan_v1.py"
)
V2_PLAN_PATH = (
    ROOT
    / "docs/lewm_go2_world_model_counterfactual_calibration_exact_plan_v2_2026-08-02.json"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_calibration_plan_builder_v1", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_emits_exact_160_branch_rotating_repeat_plan(tmp_path: Path) -> None:
    builder = _load_builder()
    builder.REPO_ROOT = tmp_path
    smoke = _smoke_plan(tmp_path / "runtime")
    runtime_contract = {
        "schema": builder.RUNTIME_CONTRACT_SCHEMA,
        "runtime_bindings": smoke["runtime_bindings"],
        "execution_contract": smoke["execution_contract"],
    }
    scenes = []
    for family_index, family in enumerate(pilot.FAMILIES):
        scene_root = tmp_path / "scenes" / family
        scene_root.mkdir(parents=True)
        manifest = scene_root / "manifest.json"
        genesis = scene_root / "genesis_scene.json"
        canonical_target = [1.0, float(family_index)]
        manifest.write_text(json.dumps({
            "scene_id": f"calibration-{family_index}",
            "family": family,
            "landmarks": [
                {
                    "object_id": "target-000",
                    "center_xyz_m": [*canonical_target, 0.5],
                }
            ],
        }))
        genesis.write_text("{}")
        scenes.append({
            "family": family,
            "scene_id": f"calibration-{family_index}",
            "scene_manifest_binding": pilot.file_binding(manifest),
            "scene_genesis_binding": pilot.file_binding(genesis),
            "states": [
                {
                    "state_id": f"calibration-{family_index}:{state_index}",
                    "history_action_ids": [state_index, state_index + 1],
                    "target_xy_m": canonical_target,
                }
                for state_index in range(2)
            ],
        })
    output_root = (tmp_path / ".generated/dev/calibration-attempt").resolve()
    plan = builder.build_calibration_plan_v1(
        attempt_id="calibration-attempt-v1",
        output_root=output_root,
        scene_panel={"schema": builder.SCENE_PANEL_SCHEMA, "scenes": scenes},
        runtime_contract=runtime_contract,
    )
    assert plan["expected_counts"] == {
        "scenes": 8,
        "states": 16,
        "roles": {"calibration": 16},
        "actions": 9,
        "candidate_branches": 144,
        "sentinel_branches": 16,
        "total_branches": 160,
        "context_frames": 48,
        "target_frames": 160,
    }
    repeated = [state["sentinel_duplicate_action_id"] for state in plan["states"]]
    assert repeated == [index % 9 for index in range(16)]
    assert set(repeated) == set(range(9))


def test_written_v2_plan_has_fresh_identity_and_frozen_branch_design() -> None:
    plan = pilot.validate_plan(json.loads(V2_PLAN_PATH.read_bytes()))
    assert plan["attempt_id"] == "lewm-go2-wm-counterfactual-calibration-v2"
    assert plan["output_root"] == str(
        (
            ROOT
            / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v2"
        ).resolve()
    )
    assert plan["attempt_id"] != "lewm-go2-wm-counterfactual-calibration-v1"
    assert plan["expected_counts"]["candidate_branches"] == 144
    assert plan["expected_counts"]["sentinel_branches"] == 16
    assert [
        state["sentinel_duplicate_action_id"] for state in plan["states"]
    ] == [index % 9 for index in range(16)]


def test_builder_rejects_target_not_bound_to_manifest(tmp_path: Path) -> None:
    builder = _load_builder()
    builder.REPO_ROOT = tmp_path
    smoke = _smoke_plan(tmp_path / "runtime")
    runtime_contract = {
        "schema": builder.RUNTIME_CONTRACT_SCHEMA,
        "runtime_bindings": smoke["runtime_bindings"],
        "execution_contract": smoke["execution_contract"],
    }
    scenes = []
    for family_index, family in enumerate(pilot.FAMILIES):
        scene_root = tmp_path / "scenes" / family
        scene_root.mkdir(parents=True)
        manifest = scene_root / "manifest.json"
        genesis = scene_root / "genesis_scene.json"
        canonical_target = [1.0, float(family_index)]
        manifest.write_text(json.dumps({
            "scene_id": f"calibration-{family_index}",
            "family": family,
            "landmarks": [
                {
                    "object_id": "target-000",
                    "center_xyz_m": [*canonical_target, 0.5],
                }
            ],
        }))
        genesis.write_text("{}")
        scenes.append({
            "family": family,
            "scene_id": f"calibration-{family_index}",
            "scene_manifest_binding": pilot.file_binding(manifest),
            "scene_genesis_binding": pilot.file_binding(genesis),
            "states": [
                {
                    "state_id": f"calibration-{family_index}:{state_index}",
                    "history_action_ids": [state_index, state_index + 1],
                    "target_xy_m": canonical_target,
                }
                for state_index in range(2)
            ],
        })
    tampered = copy.deepcopy(scenes)
    tampered[0]["states"][0]["target_xy_m"] = [9.0, 9.0]
    with pytest.raises(builder.CalibrationPlanBuildError, match="canonical landmark"):
        builder.build_calibration_plan_v1(
            attempt_id="calibration-attempt-v1",
            output_root=(tmp_path / ".generated/dev/calibration-attempt").resolve(),
            scene_panel={"schema": builder.SCENE_PANEL_SCHEMA, "scenes": tampered},
            runtime_contract=runtime_contract,
        )
