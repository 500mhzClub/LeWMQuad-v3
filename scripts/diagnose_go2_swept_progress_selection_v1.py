#!/usr/bin/env python3
"""Census a post-action swept-progress target on checkpoint selection only."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from scripts import (  # noqa: E402
    diagnose_go2_post_action_projective_support_selection_admissibility_v1
    as admissibility,
)


labels = admissibility.labels
SCHEMA = "lewm_go2_swept_progress_selection_census_v1"
SELECTION_ROLE = admissibility.SELECTION_ROLE
NON_HOLD_ACTIONS = admissibility.NON_HOLD_ACTIONS
SEGMENT_COUNT = 15
SEGMENT_LENGTH_M = 0.1
MINIMUM_INFORMATIVE_STATES = 128
MINIMUM_INFORMATIVE_STATES_PER_FAMILY = 8


def _new_action_counts() -> dict[str, Any]:
    return {
        "state_count": 0,
        "immediate_feasible_count": 0,
        "positive_prefix_count": 0,
        "variation_participation_count": 0,
        "prefix_histogram_0_through_15": [0] * (SEGMENT_COUNT + 1),
    }


def _new_scope() -> dict[str, Any]:
    return {
        "state_count": 0,
        "informative_state_count": 0,
        "rejection_counts": {
            "zero_best_prefix": 0,
            "positive_but_no_action_difference": 0,
        },
        "prefix_histogram_0_through_15": [0] * (SEGMENT_COUNT + 1),
        "actions": {action: _new_action_counts() for action in NON_HOLD_ACTIONS},
    }


def swept_progress_prefix_v1(
    checker: Any,
    post_action_pose_world: Any,
    *,
    immediate_feasible: bool,
) -> int:
    """Count consecutive collision-free 0.1 m segments after an action."""

    if not immediate_feasible:
        return 0
    cos_yaw = math.cos(post_action_pose_world.yaw_rad)
    sin_yaw = math.sin(post_action_pose_world.yaw_rad)
    prefix = 0
    for segment_index in range(SEGMENT_COUNT):
        start_distance = segment_index * SEGMENT_LENGTH_M
        end_distance = (segment_index + 1) * SEGMENT_LENGTH_M
        start = labels.Pose2D(
            post_action_pose_world.x_m + start_distance * cos_yaw,
            post_action_pose_world.y_m + start_distance * sin_yaw,
            post_action_pose_world.yaw_rad,
        )
        end = labels.Pose2D(
            post_action_pose_world.x_m + end_distance * cos_yaw,
            post_action_pose_world.y_m + end_distance * sin_yaw,
            post_action_pose_world.yaw_rad,
        )
        samples = checker.interpolated_sweep(
            start,
            end,
            maximum_corner_step_m=labels.MAXIMUM_CORNER_STEP_M,
            maximum_yaw_step_rad=labels.MAXIMUM_YAW_STEP_RAD,
        )
        if any(not checker.pose_feasibility(pose).feasible for _, pose in samples):
            break
        prefix += 1
    return prefix


def state_rows_v1(
    *,
    source_pose_world: Any,
    scene_manifest: Any,
    footprint: Any,
    commands_by_action: Mapping[str, Sequence[Sequence[float]]],
    role_state_index: int,
    family: str,
) -> tuple[dict[str, Any], ...]:
    """Measure the eight non-HOLD actions with the reviewed geometry checker."""

    checker = labels.ManifestDirectionalFootprintFeasibility(scene_manifest, footprint)
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(labels.ACTION_ORDER):
        if action == "hold":
            continue
        commands = commands_by_action.get(action)
        if commands is None:
            raise labels.LabelContractError(f"missing command block for {action}")
        immediate = labels._feasibility_summary_v1(
            checker,
            labels._sampled_immediate_poses_v1(
                checker, source_pose_world, commands
            ),
        )
        post_local = labels.integrate_action_v1(commands)[-1]
        post_world = labels.transform_pose_v1(source_pose_world, post_local)
        rows.append(
            {
                "dataset_role": SELECTION_ROLE,
                "role_state_index": role_state_index,
                "family": family,
                "action_index": action_index,
                "action": action,
                "immediate_feasible": bool(immediate["feasible"]),
                "swept_progress_prefix_length": swept_progress_prefix_v1(
                    checker,
                    post_world,
                    immediate_feasible=bool(immediate["feasible"]),
                ),
            }
        )
    return tuple(rows)


def _increment_scope(
    scope: MutableMapping[str, Any],
    *,
    by_action: Mapping[str, Mapping[str, Any]],
) -> None:
    prefixes = {
        action: int(row["swept_progress_prefix_length"])
        for action, row in by_action.items()
    }
    if any(prefix < 0 or prefix > SEGMENT_COUNT for prefix in prefixes.values()):
        raise labels.LabelContractError("swept-progress prefix escaped 0 through 15")
    positive = max(prefixes.values()) > 0
    varied = len(set(prefixes.values())) >= 2
    informative = positive and varied

    scope["state_count"] += 1
    scope["informative_state_count"] += int(informative)
    if not positive:
        scope["rejection_counts"]["zero_best_prefix"] += 1
    elif not varied:
        scope["rejection_counts"]["positive_but_no_action_difference"] += 1
    for action, row in by_action.items():
        prefix = prefixes[action]
        participates = any(prefix != other for other in prefixes.values())
        scope["prefix_histogram_0_through_15"][prefix] += 1
        counts = scope["actions"][action]
        counts["state_count"] += 1
        counts["immediate_feasible_count"] += int(row["immediate_feasible"])
        counts["positive_prefix_count"] += int(prefix > 0)
        counts["variation_participation_count"] += int(participates)
        counts["prefix_histogram_0_through_15"][prefix] += 1


def aggregate_selection_rows_v1(
    rows: Sequence[Mapping[str, Any]], *, families: Iterable[str]
) -> dict[str, Any]:
    family_order = tuple(families)
    if not family_order or len(set(family_order)) != len(family_order):
        raise labels.LabelContractError("census family registry is empty or repeated")
    family_scopes = {family: _new_scope() for family in family_order}
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_role") != SELECTION_ROLE:
            raise labels.LabelContractError("census row escaped checkpoint selection")
        grouped[int(row["role_state_index"])].append(row)

    aggregate = _new_scope()
    for state_index in sorted(grouped):
        state_rows = sorted(grouped[state_index], key=lambda row: int(row["action_index"]))
        if [row.get("action") for row in state_rows] != list(NON_HOLD_ACTIONS):
            raise labels.LabelContractError("census state action set/order changed")
        family = str(state_rows[0].get("family"))
        if family not in family_scopes or any(row.get("family") != family for row in state_rows):
            raise labels.LabelContractError("census state family escaped its registry")
        by_action = {str(row["action"]): row for row in state_rows}
        _increment_scope(aggregate, by_action=by_action)
        _increment_scope(family_scopes[family], by_action=by_action)
    aggregate["family_count"] = len(family_order)
    return {"aggregate": aggregate, "families": family_scopes}


def run_census_v1() -> Mapping[str, Any]:
    ledger = labels.new_access_ledger_v1()
    binding_path = ROOT / labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH
    raw_binding = binding_path.read_bytes()
    ledger["execution_binding_opens"] += 1
    if (
        len(raw_binding) != admissibility.V4_BINDING_BYTE_COUNT
        or hashlib.sha256(raw_binding).hexdigest()
        != admissibility.V4_BINDING_FILE_SHA256
    ):
        raise labels.LabelContractError("exact V4 execution binding bytes changed")
    binding = json.loads(raw_binding)
    if (
        not isinstance(binding, dict)
        or labels.canonical_json_bytes(binding) + b"\n" != raw_binding
        or binding.get("content_sha256")
        != admissibility.V4_BINDING_CONTENT_SHA256
    ):
        raise labels.LabelContractError("exact V4 execution binding content changed")
    labels.validate_execution_binding_envelope_v1(binding)
    inputs = {
        name: admissibility._path(record)
        for name, record in binding["inputs"].items()
    }

    raw_indexes = labels.load_and_validate_raw_indexes(
        inputs["raw_manifest"],
        inputs["raw_pairs"],
        inputs["raw_endpoints"],
        access_ledger=ledger,
    )
    for name in ("raw_manifest", "raw_pairs", "raw_endpoints"):
        admissibility._assert_bound_size(binding, name, inputs[name])
    labels.validate_raw_audit_v1(inputs["raw_audit"], access_ledger=ledger)
    admissibility._assert_bound_size(binding, "raw_audit", inputs["raw_audit"])
    source_records = labels.validate_execution_binding_v1(
        binding, raw_indexes=raw_indexes
    )
    geometry = labels.load_geometry_inputs_v1(
        repository_root=ROOT,
        geometry_path=inputs["geometry_contract"],
        directional_policy_path=inputs["directional_policy"],
        primitive_registry_path=inputs["primitive_registry"],
        access_ledger=ledger,
    )
    for name in ("geometry_contract", "directional_policy", "primitive_registry"):
        admissibility._assert_bound_size(binding, name, inputs[name])

    selection_pairs = tuple(
        pair for pair in raw_indexes.pairs if pair["dataset_role"] == SELECTION_ROLE
    )
    role_index = {
        str(pair["content_sha256"]): index
        for index, pair in enumerate(selection_pairs)
    }
    selection_scenes = sorted(
        scene
        for scene, shard in raw_indexes.shard_by_scene.items()
        if shard["dataset_role"] == SELECTION_ROLE
    )
    if len(selection_scenes) != 8:
        raise labels.LabelContractError("checkpoint-selection scene count changed")
    rows: list[dict[str, Any]] = []
    for scene_id in selection_scenes:
        scene_manifest, states = labels.load_joined_scene_v1(
            raw_indexes=raw_indexes,
            scene_id=scene_id,
            source_records=source_records[scene_id],
            repository_root=ROOT,
            access_ledger=ledger,
        )
        for state in states:
            pair_sha256 = str(state.pair["content_sha256"])
            rows.extend(
                state_rows_v1(
                    source_pose_world=state.source_pose_world,
                    scene_manifest=scene_manifest,
                    footprint=geometry.footprint,
                    commands_by_action=geometry.commands_by_action,
                    role_state_index=role_index[pair_sha256],
                    family=str(state.pair["family"]),
                )
            )
    rows.sort(key=lambda row: (int(row["role_state_index"]), int(row["action_index"])))
    census = aggregate_selection_rows_v1(
        rows, families=labels.REGISTERED_SELECTION_FAMILIES
    )

    expected_ledger = labels.new_access_ledger_v1()
    expected_ledger.update(
        {
            "execution_binding_opens": 1,
            "raw_manifest_opens": 1,
            "raw_pairs_opens": 1,
            "raw_endpoints_opens": 1,
            "raw_audit_opens": 1,
            "geometry_contract_opens": 1,
            "geometry_contract_validation_calls": 1,
            "directional_policy_opens": 1,
            "primitive_registry_opens": 1,
            "scene_join_calls_started": 8,
            "render_summary_opens": 8,
            "source_frames_jsonl_opens": 8,
            "scene_manifest_opens": 8,
        }
    )
    if ledger != expected_ledger:
        raise labels.LabelContractError("selection census access ledger changed")
    aggregate = census["aggregate"]
    family_counts = {
        family: scope["informative_state_count"]
        for family, scope in census["families"].items()
    }
    passes = (
        aggregate["informative_state_count"] >= MINIMUM_INFORMATIVE_STATES
        and all(
            count >= MINIMUM_INFORMATIVE_STATES_PER_FAMILY
            for count in family_counts.values()
        )
    )
    return labels.with_content_sha256(
        {
            "schema": SCHEMA,
            "status": "PASS_SELECTION_SCREEN" if passes else "STOP_SELECTION_SCREEN",
            "decision_rule": {
                "minimum_informative_states": MINIMUM_INFORMATIVE_STATES,
                "minimum_informative_states_per_registered_family": (
                    MINIMUM_INFORMATIVE_STATES_PER_FAMILY
                ),
            },
            "target": {
                "non_hold_action_count": len(NON_HOLD_ACTIONS),
                "immediate_primitive_must_be_feasible": True,
                "post_action_straight_segment_count": SEGMENT_COUNT,
                "straight_segment_length_m": SEGMENT_LENGTH_M,
                "maximum_distance_m": SEGMENT_COUNT * SEGMENT_LENGTH_M,
                "maximum_corner_step_m": labels.MAXIMUM_CORNER_STEP_M,
                "maximum_yaw_step_rad": labels.MAXIMUM_YAW_STEP_RAD,
                "informative_requires_positive_best_and_two_distinct_prefixes": True,
            },
            "binding": {
                "path": labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH,
                "byte_count": admissibility.V4_BINDING_BYTE_COUNT,
                "file_sha256": admissibility.V4_BINDING_FILE_SHA256,
                "content_sha256": admissibility.V4_BINDING_CONTENT_SHA256,
            },
            "selection_state_count": len(selection_pairs),
            "selection_action_row_count": len(rows),
            "selection_scene_count": len(selection_scenes),
            "census": census,
            "access_ledger": ledger,
            "authority": {
                "rgb_opened": False,
                "model_or_checkpoint_opened": False,
                "gpu_or_training_used": False,
                "schedule_opened": False,
                "labels_or_v4_output_opened": False,
                "g2_heldout_or_sealed_opened": False,
                "filesystem_outputs_written": False,
            },
        }
    )


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot diagnostic accepts no arguments")
    print(json.dumps(run_census_v1(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
