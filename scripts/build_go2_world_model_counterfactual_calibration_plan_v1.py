#!/usr/bin/env python3
"""Build the exact metadata-only 160-branch counterfactual calibration plan.

The builder opens the two caller-bound metadata inputs plus each exact ordinary
scene manifest named by the panel.  It does not open RGB, checkpoints, Genesis
packs, or other simulator payloads.  Each target must equal that manifest's
canonical landmark center, preventing an arbitrary finite coordinate from
becoming experiment identity.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402


SCENE_PANEL_SCHEMA = "lewm_go2_world_model_counterfactual_calibration_scene_panel_v1"
RUNTIME_CONTRACT_SCHEMA = "lewm_go2_world_model_counterfactual_runtime_contract_v1"


class CalibrationPlanBuildError(RuntimeError):
    """Raised before malformed metadata can become an executable plan."""


def _canonical_manifest_target(
    binding: Mapping[str, Any], *, scene_id: str, family: str
) -> list[float]:
    """Return the exact canonical landmark center from one bound manifest."""

    document, actual = pilot.read_bound_json(
        Path(str(binding["path"])),
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label=f"calibration scene {scene_id} manifest",
    )
    if actual != dict(binding) or not isinstance(document, Mapping):
        raise CalibrationPlanBuildError("calibration manifest binding changed")
    if document.get("scene_id") != scene_id or document.get("family") != family:
        raise CalibrationPlanBuildError(
            "calibration panel identity disagrees with its bound manifest"
        )
    landmarks = document.get("landmarks")
    if not isinstance(landmarks, list) or not landmarks:
        raise CalibrationPlanBuildError(
            f"calibration scene {scene_id} has no target landmark"
        )
    normalized: list[tuple[str, list[float]]] = []
    seen_ids: set[str] = set()
    for landmark in landmarks:
        if not isinstance(landmark, Mapping):
            raise CalibrationPlanBuildError("calibration landmark is malformed")
        object_id = landmark.get("object_id")
        center = landmark.get("center_xyz_m")
        if (
            not isinstance(object_id, str)
            or not object_id
            or object_id in seen_ids
            or not isinstance(center, list)
            or len(center) != 3
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                for coordinate in center
            )
        ):
            raise CalibrationPlanBuildError("calibration landmark is malformed")
        seen_ids.add(object_id)
        normalized.append((object_id, [float(center[0]), float(center[1])]))
    return min(normalized, key=lambda item: item[0])[1]


def _validate_scene_panel(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"schema", "scenes"}:
        raise CalibrationPlanBuildError("calibration scene panel fields changed")
    if value["schema"] != SCENE_PANEL_SCHEMA or not isinstance(value["scenes"], list):
        raise CalibrationPlanBuildError("calibration scene panel schema changed")
    scenes = value["scenes"]
    if len(scenes) != len(pilot.FAMILIES):
        raise CalibrationPlanBuildError("calibration panel must contain eight scenes")
    normalized: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    for family, scene in zip(pilot.FAMILIES, scenes, strict=True):
        if not isinstance(scene, Mapping) or set(scene) != {
            "family",
            "scene_id",
            "scene_manifest_binding",
            "scene_genesis_binding",
            "states",
        }:
            raise CalibrationPlanBuildError("calibration scene entry changed")
        scene_id = scene["scene_id"]
        if (
            scene["family"] != family
            or not isinstance(scene_id, str)
            or not scene_id
            or scene_id in seen_scene_ids
        ):
            raise CalibrationPlanBuildError("calibration scene order/identity changed")
        seen_scene_ids.add(scene_id)
        manifest = pilot.require_binding(
            scene["scene_manifest_binding"],
            label=f"calibration scene {scene_id} manifest",
        )
        genesis = pilot.require_binding(
            scene["scene_genesis_binding"],
            label=f"calibration scene {scene_id} Genesis pack",
        )
        if (
            Path(manifest["path"]).name != "manifest.json"
            or Path(genesis["path"]).name != "genesis_scene.json"
            or Path(manifest["path"]).parent != Path(genesis["path"]).parent
        ):
            raise CalibrationPlanBuildError("calibration scene binding pair changed")
        canonical_target = _canonical_manifest_target(
            manifest, scene_id=scene_id, family=family
        )
        states = scene["states"]
        if not isinstance(states, list) or len(states) != 2:
            raise CalibrationPlanBuildError("each calibration scene needs two states")
        normalized_states = []
        for state_index, state in enumerate(states):
            if not isinstance(state, Mapping) or set(state) != {
                "state_id",
                "history_action_ids",
                "target_xy_m",
            }:
                raise CalibrationPlanBuildError("calibration state fields changed")
            state_id = state["state_id"]
            history = state["history_action_ids"]
            target = state["target_xy_m"]
            if (
                not isinstance(state_id, str)
                or not state_id
                or not isinstance(history, list)
                or len(history) != 2
                or any(type(action) is not int or not 0 <= action < pilot.ACTION_COUNT for action in history)
                or not isinstance(target, list)
                or len(target) != 2
                or any(
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not math.isfinite(float(coordinate))
                    for coordinate in target
                )
            ):
                raise CalibrationPlanBuildError(
                    f"calibration state {scene_id}/{state_index} is invalid"
                )
            normalized_target = [float(coordinate) for coordinate in target]
            if normalized_target != canonical_target:
                raise CalibrationPlanBuildError(
                    f"calibration state {scene_id}/{state_index} target is not the "
                    "bound manifest's canonical landmark center"
                )
            normalized_states.append({
                "state_id": state_id,
                "history_action_ids": list(history),
                "target_xy_m": normalized_target,
            })
        normalized.append({
            "family": family,
            "scene_id": scene_id,
            "scene_manifest_binding": manifest,
            "scene_genesis_binding": genesis,
            "states": normalized_states,
        })
    return normalized


def _validate_runtime_contract(value: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "runtime_bindings",
        "execution_contract",
    }:
        raise CalibrationPlanBuildError("runtime contract fields changed")
    if value["schema"] != RUNTIME_CONTRACT_SCHEMA:
        raise CalibrationPlanBuildError("runtime contract schema changed")
    runtime = value["runtime_bindings"]
    execution = value["execution_contract"]
    if not isinstance(runtime, Mapping) or not isinstance(execution, Mapping):
        raise CalibrationPlanBuildError("runtime/execution contract is absent")
    normalized_runtime = {
        str(name): pilot.require_binding(binding, label=f"runtime {name}")
        for name, binding in runtime.items()
    }
    return normalized_runtime, copy.deepcopy(dict(execution))


def build_calibration_plan_v1(
    *,
    attempt_id: str,
    output_root: Path,
    scene_panel: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}", attempt_id) is None:
        raise CalibrationPlanBuildError("attempt_id is invalid")
    selected_output_root = Path(output_root)
    if not selected_output_root.is_absolute():
        raise CalibrationPlanBuildError("output_root must be absolute")
    development_root = (REPO_ROOT / ".generated/dev").resolve()
    if not selected_output_root.resolve().is_relative_to(development_root):
        raise CalibrationPlanBuildError(
            f"calibration output_root must remain under {development_root}"
        )
    if selected_output_root.exists() or selected_output_root.is_symlink():
        raise CalibrationPlanBuildError("calibration output_root must be fresh")
    scenes = _validate_scene_panel(scene_panel)
    runtime_bindings, execution_contract = _validate_runtime_contract(runtime_contract)
    states: list[dict[str, Any]] = []
    for scene in scenes:
        for state_index, state in enumerate(scene["states"]):
            group_index = len(states)
            states.append({
                "state_id": state["state_id"],
                "role": "calibration",
                "family": scene["family"],
                "scene_id": scene["scene_id"],
                "scene_manifest_binding": scene["scene_manifest_binding"],
                "scene_genesis_binding": scene["scene_genesis_binding"],
                "scene_generation": None,
                "group_index": group_index,
                "state_index_in_scene": state_index,
                "history_action_ids": state["history_action_ids"],
                "candidate_action_ids": list(range(pilot.ACTION_COUNT)),
                "sentinel_duplicate_action_id": group_index % pilot.ACTION_COUNT,
                "target_xy_m": state["target_xy_m"],
            })
    repeated = [state["sentinel_duplicate_action_id"] for state in states]
    if set(repeated) != set(range(pilot.ACTION_COUNT)):
        raise CalibrationPlanBuildError("repeat allocation misses a primitive")
    action_catalog = [
        {
            "action_id": action_id,
            "name": name,
            "requested_block": [list(command) for command in pilot.CANONICAL_ACTION_BLOCKS[action_id]],
        }
        for action_id, name in enumerate(pilot.CANONICAL_ACTIONS)
    ]
    plan = {
        "schema": pilot.PLAN_SCHEMA,
        "attempt_id": attempt_id,
        "purpose": "sizing_calibration_only",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "states_per_scene": 2,
        "history_blocks": pilot.HISTORY_BLOCK_COUNT,
        "output_root": str(selected_output_root),
        "runtime_bindings": runtime_bindings,
        "execution_contract": execution_contract,
        "render_contract": dict(pilot.RENDER_CONTRACT),
        "action_catalog": action_catalog,
        "states": states,
        "expected_counts": pilot.expected_counts_from_states(states),
    }
    return pilot.validate_plan(plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene-panel", required=True, type=Path)
    parser.add_argument("--expected-scene-panel-sha256", required=True)
    parser.add_argument("--expected-scene-panel-byte-count", required=True, type=int)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--expected-runtime-contract-byte-count", required=True, type=int)
    parser.add_argument("--plan-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_panel, _ = pilot.read_bound_json(
        args.scene_panel,
        expected_sha256=args.expected_scene_panel_sha256,
        expected_byte_count=args.expected_scene_panel_byte_count,
        label="calibration scene panel",
    )
    runtime_contract, _ = pilot.read_bound_json(
        args.runtime_contract,
        expected_sha256=args.expected_runtime_contract_sha256,
        expected_byte_count=args.expected_runtime_contract_byte_count,
        label="counterfactual runtime contract",
    )
    plan = build_calibration_plan_v1(
        attempt_id=args.attempt_id,
        output_root=args.output_root,
        scene_panel=scene_panel,
        runtime_contract=runtime_contract,
    )
    binding = pilot.write_json_exclusive(args.plan_output, plan)
    print(json.dumps({"plan": binding, "expected_counts": plan["expected_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationPlanBuildError",
    "RUNTIME_CONTRACT_SCHEMA",
    "SCENE_PANEL_SCHEMA",
    "build_calibration_plan_v1",
]
