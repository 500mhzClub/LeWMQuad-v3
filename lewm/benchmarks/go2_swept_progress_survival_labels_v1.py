"""Model-free swept-progress labels over the exact V4 development sources."""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from lewm.benchmarks import go2_post_action_projective_support_labels_v1 as v4
from scripts import diagnose_go2_swept_progress_selection_v1 as census


ROW_SCHEMA = "lewm_go2_swept_progress_survival_label_row_v1"
MANIFEST_SCHEMA = "lewm_go2_swept_progress_survival_labels_v1"
FAILURE_SCHEMA = "lewm_go2_swept_progress_survival_labels_v1_failure_v1"
OUTPUT_RELATIVE_PATH = ".generated/go2_swept_progress_survival_labels_v1"
V4_BINDING_RELATIVE_PATH = v4.LABEL_EXECUTION_BINDING_RELATIVE_PATH
V4_BINDING_BYTE_COUNT = census.admissibility.V4_BINDING_BYTE_COUNT
V4_BINDING_FILE_SHA256 = census.admissibility.V4_BINDING_FILE_SHA256
V4_BINDING_CONTENT_SHA256 = census.admissibility.V4_BINDING_CONTENT_SHA256
ACTION_ORDER = v4.ACTION_ORDER
NON_HOLD_ACTIONS = tuple(action for action in ACTION_ORDER if action != "hold")
ROLE_ORDER = v4.OUTPUT_ROLE_ORDER
ROLE_FILENAMES = {
    "train": "train.jsonl",
    "probability_calibration": "calibration.jsonl",
    "checkpoint_selection": "selection.jsonl",
}
INFORMATIVE_FLOORS = {
    "train": 512,
    "probability_calibration": 128,
    "checkpoint_selection": 128,
}
SELECTION_FAMILY_FLOOR = 8
SCHEDULE_PRESENTATION_COUNT = 16_000
SCHEDULE_INFORMATIVE_FLOOR = 512
SCHEDULE_ACTION_PARTICIPATION_FLOOR = 32

# This is the exact committed target that passed the 495-state selection census.
swept_progress_prefix_v1 = census.swept_progress_prefix_v1


class SweptProgressLabelError(v4.LabelContractError):
    """Raised when labels or their model-free gates change."""


def _hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    return v4.with_content_sha256(core)


def _valid_hash(value: Mapping[str, Any]) -> bool:
    return value.get("content_sha256") == v4.canonical_json_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def _pose(pose: Any) -> list[float]:
    return [float(pose.x_m), float(pose.y_m), float(pose.yaw_rad)]


def _require_zero_hold(commands: Mapping[str, Sequence[Sequence[float]]]) -> None:
    hold = commands.get("hold")
    if (
        hold is None
        or len(hold) != v4.COMMAND_COUNT
        or any(len(command) != 3 for command in hold)
        or any(float(value) != 0.0 for command in hold for value in command)
    ):
        raise SweptProgressLabelError("HOLD is not the exact zero primitive")


def label_state_v1(
    *,
    pair: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    source_pose_world: Any,
    source_line_number: int,
    scene_manifest: Any,
    footprint: Any,
    commands_by_action: Mapping[str, Sequence[Sequence[float]]],
    source_bindings: Mapping[str, Mapping[str, Any]],
    role_state_index: int,
) -> tuple[dict[str, Any], ...]:
    """Return nine rows; HOLD uses zero motion and the same continuation target."""

    role = str(pair.get("dataset_role"))
    scene_id = str(pair.get("scene_id"))
    family = str(pair.get("family"))
    if role not in ROLE_ORDER:
        raise SweptProgressLabelError("state escaped development roles")
    if (scene_manifest.scene_id, scene_manifest.family) != (scene_id, family):
        raise SweptProgressLabelError("state crossed its scene manifest")
    if endpoint.get("endpoint_identity_sha256") != pair.get("current_endpoint_sha256"):
        raise SweptProgressLabelError("state crossed its endpoint")
    _require_zero_hold(commands_by_action)

    checker = v4.ManifestDirectionalFootprintFeasibility(scene_manifest, footprint)
    candidates: list[tuple[str, int, Any, bool, int]] = []
    for action_index, action in enumerate(ACTION_ORDER):
        commands = commands_by_action[action]
        post_local = v4.integrate_action_v1(commands)[-1]
        if action == "hold" and _pose(post_local) != [0.0, 0.0, 0.0]:
            raise SweptProgressLabelError("HOLD integration moved the robot")
        immediate = v4._feasibility_summary_v1(
            checker,
            v4._sampled_immediate_poses_v1(checker, source_pose_world, commands),
        )
        feasible = bool(immediate["feasible"])
        prefix = swept_progress_prefix_v1(
            checker,
            v4.transform_pose_v1(source_pose_world, post_local),
            immediate_feasible=feasible,
        )
        if type(prefix) is not int or prefix < 0 or prefix > census.SEGMENT_COUNT:
            raise SweptProgressLabelError("prefix escaped 0 through 15")
        candidates.append((action, action_index, post_local, feasible, prefix))

    non_hold_prefixes = tuple(
        prefix for action, _, _, _, prefix in candidates if action != "hold"
    )
    informative = max(non_hold_prefixes) > 0 and len(set(non_hold_prefixes)) >= 2
    provenance = {
        "endpoint_index_content_sha256": endpoint.get("content_sha256"),
        "source_frame_line_number": int(source_line_number),
        "source_pose_world_xy_yaw": _pose(source_pose_world),
        "executed_pair_primitive": pair.get("primitive"),
        "source_bindings_sha256": v4.canonical_json_sha256(
            {key: dict(value) for key, value in sorted(source_bindings.items())}
        ),
        "source_frames_jsonl_sha256": pair.get("frames_jsonl_sha256"),
        "scene_manifest_content_sha256": pair.get("scene_manifest_sha256"),
    }
    rows = []
    for action, action_index, post_local, feasible, prefix in candidates:
        participates = action != "hold" and any(
            prefix != other for other in non_hold_prefixes
        )
        rows.append(
            _hashed(
                {
                    "schema": ROW_SCHEMA,
                    "dataset_role": role,
                    "role_state_index": int(role_state_index),
                    "global_row": int(pair["global_row"]),
                    "pair_content_sha256": pair.get("content_sha256"),
                    "current_endpoint_sha256": pair.get("current_endpoint_sha256"),
                    "scene_id": scene_id,
                    "family": family,
                    "action_index": action_index,
                    "action": action,
                    "nominal_post_action_se2_current_frame": _pose(post_local),
                    "immediate_primitive_feasible": feasible,
                    "swept_progress_prefix_length": prefix,
                    "informative_state": informative,
                    "action_participates_in_unequal_prefix": participates,
                    "provenance": provenance,
                }
            )
        )
    return tuple(rows)


def _state_groups(
    rows: Sequence[Mapping[str, Any]], *, role: str, frozen: bool
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    rows = tuple(rows)
    if len(rows) % len(ACTION_ORDER):
        raise SweptProgressLabelError("role has a partial action group")
    if frozen and len(rows) != v4.ROLE_STATE_COUNTS[role] * len(ACTION_ORDER):
        raise SweptProgressLabelError(f"{role} population changed")
    groups = tuple(
        tuple(rows[offset : offset + len(ACTION_ORDER)])
        for offset in range(0, len(rows), len(ACTION_ORDER))
    )
    for state_index, group in enumerate(groups):
        if [row.get("action") for row in group] != list(ACTION_ORDER):
            raise SweptProgressLabelError("action order changed")
        if any(
            row.get("schema") != ROW_SCHEMA
            or row.get("dataset_role") != role
            or row.get("role_state_index") != state_index
            or row.get("action_index") != action_index
            or not _valid_hash(row)
            for action_index, row in enumerate(group)
        ):
            raise SweptProgressLabelError("row identity or content hash changed")
        hold = group[ACTION_ORDER.index("hold")]
        if hold.get("nominal_post_action_se2_current_frame") != [0.0, 0.0, 0.0]:
            raise SweptProgressLabelError("HOLD output moved the robot")
        prefixes = tuple(
            int(row["swept_progress_prefix_length"])
            for row in group
            if row["action"] != "hold"
        )
        informative = max(prefixes) > 0 and len(set(prefixes)) >= 2
        for row in group:
            prefix = row.get("swept_progress_prefix_length")
            action = str(row["action"])
            expected_participation = action != "hold" and any(
                prefix != other for other in prefixes
            )
            if (
                type(prefix) is not int
                or prefix < 0
                or prefix > census.SEGMENT_COUNT
                or type(row.get("immediate_primitive_feasible")) is not bool
                or (not row["immediate_primitive_feasible"] and prefix != 0)
                or row.get("informative_state") is not informative
                or row.get("action_participates_in_unequal_prefix")
                is not expected_participation
            ):
                raise SweptProgressLabelError("row target fields changed")
    return groups


def summarize_preflight_v1(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule_indices: Sequence[int],
    *,
    enforce_frozen_gates: bool = True,
) -> dict[str, Any]:
    if set(rows_by_role) != set(ROLE_ORDER):
        raise SweptProgressLabelError("preflight requires exactly three roles")
    groups = {
        role: _state_groups(rows_by_role[role], role=role, frozen=enforce_frozen_gates)
        for role in ROLE_ORDER
    }
    informative = {
        role: sum(bool(group[0]["informative_state"]) for group in role_groups)
        for role, role_groups in groups.items()
    }
    observed_families = {str(group[0]["family"]) for group in groups["checkpoint_selection"]}
    if not observed_families.issubset(v4.REGISTERED_SELECTION_FAMILIES):
        raise SweptProgressLabelError("selection family registry changed")
    family_counter = Counter(
        str(group[0]["family"])
        for group in groups["checkpoint_selection"]
        if group[0]["informative_state"]
    )
    family_counts = {
        family: family_counter[family] for family in v4.REGISTERED_SELECTION_FAMILIES
    }

    indices = tuple(schedule_indices)
    if enforce_frozen_gates and (
        len(indices) != SCHEDULE_PRESENTATION_COUNT
        or v4.canonical_json_sha256(list(indices)) != v4.SCHEDULE_PREFIX_SHA256
    ):
        raise SweptProgressLabelError("frozen schedule prefix changed")
    scheduled_informative = 0
    participation = {action: 0 for action in NON_HOLD_ACTIONS}
    for index in indices:
        if type(index) is not int or index < 0 or index >= len(groups["train"]):
            raise SweptProgressLabelError("schedule escaped train states")
        group = groups["train"][index]
        scheduled_informative += int(bool(group[0]["informative_state"]))
        for row in group:
            action = str(row["action"])
            if action in participation:
                participation[action] += int(
                    bool(row["action_participates_in_unequal_prefix"])
                )
    checks = {
        "state_counts": {role: len(role_groups) for role, role_groups in groups.items()},
        "informative_state_counts": informative,
        "selection_family_informative_counts": family_counts,
        "frozen_schedule": {
            "presentation_count": len(indices),
            "presentation_indices_sha256": v4.canonical_json_sha256(list(indices)),
            "informative_presentation_count": scheduled_informative,
            "unequal_prefix_participation_presentations_by_action": participation,
        },
    }
    if enforce_frozen_gates:
        try:
            enforce_preflight_gates_v1(checks)
        except SweptProgressLabelError as error:
            error.checks = checks
            raise
    return checks


def enforce_preflight_gates_v1(checks: Mapping[str, Any]) -> None:
    informative = checks.get("informative_state_counts", {})
    families = checks.get("selection_family_informative_counts", {})
    schedule = checks.get("frozen_schedule", {})
    participation = schedule.get(
        "unequal_prefix_participation_presentations_by_action", {}
    )
    if (
        checks.get("state_counts")
        != {role: v4.ROLE_STATE_COUNTS[role] for role in ROLE_ORDER}
        or any(informative.get(role, -1) < floor for role, floor in INFORMATIVE_FLOORS.items())
        or set(families) != set(v4.REGISTERED_SELECTION_FAMILIES)
        or any(value < SELECTION_FAMILY_FLOOR for value in families.values())
        or schedule.get("presentation_count") != SCHEDULE_PRESENTATION_COUNT
        or schedule.get("presentation_indices_sha256") != v4.SCHEDULE_PREFIX_SHA256
        or schedule.get("informative_presentation_count", -1) < SCHEDULE_INFORMATIVE_FLOOR
        or set(participation) != set(NON_HOLD_ACTIONS)
        or any(value < SCHEDULE_ACTION_PARTICIPATION_FLOOR for value in participation.values())
    ):
        raise SweptProgressLabelError("development swept-progress preflight failed")


def expected_access_ledger_v1() -> dict[str, int]:
    ledger = v4.new_access_ledger_v1()
    ledger.update(
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
            "schedule_opens": 1,
            "scene_join_calls_started": 88,
            "render_summary_opens": 88,
            "source_frames_jsonl_opens": 88,
            "scene_manifest_opens": 88,
        }
    )
    return ledger


def _input_paths(binding: Mapping[str, Any], root: Path) -> dict[str, Path]:
    return {
        str(name): (
            Path(str(record["path"]))
            if Path(str(record["path"])).is_absolute()
            else root / str(record["path"])
        ).absolute()
        for name, record in binding["inputs"].items()
    }


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_publish(output: Path, payloads: Mapping[str, bytes]) -> None:
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise PermissionError("output or staging path already exists")
    staging.mkdir(mode=0o700)
    for filename in sorted(payloads):
        _write(staging / filename, payloads[filename])
    fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if output.exists() or output.is_symlink():
        raise PermissionError("output appeared during publication")
    os.rename(staging, output)


def _success_payloads(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    binding: Mapping[str, Any],
    preflight: Mapping[str, Any],
    ledger: Mapping[str, int],
) -> tuple[Mapping[str, Any], dict[str, bytes]]:
    payloads = {
        ROLE_FILENAMES[role]: b"".join(
            v4.canonical_json_bytes(row) + b"\n" for row in rows_by_role[role]
        )
        for role in ROLE_ORDER
    }
    files = [
        {
            "path": ROLE_FILENAMES[role],
            "dataset_role": role,
            "state_count": len(rows_by_role[role]) // len(ACTION_ORDER),
            "action_row_count": len(rows_by_role[role]),
            "byte_count": len(payloads[ROLE_FILENAMES[role]]),
            "file_sha256": hashlib.sha256(payloads[ROLE_FILENAMES[role]]).hexdigest(),
            "ordered_row_content_sha256": v4.canonical_json_sha256(
                [row["content_sha256"] for row in rows_by_role[role]]
            ),
        }
        for role in ROLE_ORDER
    ]
    manifest = _hashed(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "complete_model_free_development_labels",
            "roles": list(ROLE_ORDER),
            "role_files": dict(ROLE_FILENAMES),
            "action_order": list(ACTION_ORDER),
            "state_count": sum(v4.ROLE_STATE_COUNTS.values()),
            "action_row_count": sum(v4.ROLE_STATE_COUNTS.values()) * len(ACTION_ORDER),
            "target": {
                "immediate_primitive_feasibility": True,
                "straight_segment_count": census.SEGMENT_COUNT,
                "straight_segment_length_m": census.SEGMENT_LENGTH_M,
                "hold_zero_then_same_continuation": True,
            },
            "preflight": dict(preflight),
            "files": sorted(files, key=lambda item: item["path"]),
            "input_bindings": {
                "v4_execution_binding": {
                    "path": V4_BINDING_RELATIVE_PATH,
                    "byte_count": V4_BINDING_BYTE_COUNT,
                    "file_sha256": V4_BINDING_FILE_SHA256,
                    "content_sha256": V4_BINDING_CONTENT_SHA256,
                },
                "inputs": {name: dict(record) for name, record in binding["inputs"].items()},
                "source_records_sha256": v4.canonical_json_sha256(binding["source_records"]),
                "schedule_prefix_sha256": v4.SCHEDULE_PREFIX_SHA256,
            },
            "access_ledger": dict(ledger),
            "authority": {
                "model_free_only": True,
                "rgb_model_gpu_training_opened_or_used": False,
                "g2_navigation_heldout_sealed_opened": False,
                "training_or_promotion_authorized": False,
            },
        }
    )
    payloads["manifest.json"] = v4.canonical_json_bytes(manifest) + b"\n"
    return manifest, payloads


def _failure_receipt(
    *, phase: str, error: BaseException, ledger: Mapping[str, int]
) -> Mapping[str, Any]:
    core = {
        "schema": FAILURE_SCHEMA,
        "status": "failed_without_label_publication",
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "access_ledger": dict(ledger),
        "authority": {
            "rgb_model_gpu_training_opened_or_used": False,
            "g2_navigation_heldout_sealed_opened": False,
        },
    }
    checks = getattr(error, "checks", None)
    if isinstance(checks, Mapping):
        core["preflight"] = dict(checks)
    return _hashed(core)


def build_from_v4_binding_v1(
    binding_path: Path, *, repository_root: Path
) -> Mapping[str, Any]:
    """Compute all roles in memory, then publish success or one failure receipt."""

    root = Path(repository_root).absolute()
    binding_path = Path(binding_path).absolute()
    expected_binding = (root / V4_BINDING_RELATIVE_PATH).absolute()
    output = (root / OUTPUT_RELATIVE_PATH).absolute()
    if binding_path != expected_binding:
        raise PermissionError("only the exact V4 execution binding path is accepted")
    if output.exists() or output.is_symlink():
        raise PermissionError("fresh swept-progress output already exists")

    ledger = v4.new_access_ledger_v1()
    phase = "load_v4_binding"
    try:
        binding = v4.load_execution_binding_file_v1(
            binding_path, repository_root=root, access_ledger=ledger
        )
        v4.validate_execution_binding_envelope_v1(binding)
        if (
            binding_path.stat().st_size != V4_BINDING_BYTE_COUNT
            or binding.get("content_sha256") != V4_BINDING_CONTENT_SHA256
        ):
            raise SweptProgressLabelError("exact V4 binding changed")
        inputs = _input_paths(binding, root)

        phase = "load_reviewed_metadata_geometry_schedule"
        raw = v4.load_and_validate_raw_indexes(
            inputs["raw_manifest"], inputs["raw_pairs"], inputs["raw_endpoints"],
            access_ledger=ledger,
        )
        v4.validate_raw_audit_v1(inputs["raw_audit"], access_ledger=ledger)
        source_records = v4.validate_execution_binding_v1(binding, raw_indexes=raw)
        geometry = v4.load_geometry_inputs_v1(
            repository_root=root,
            geometry_path=inputs["geometry_contract"],
            directional_policy_path=inputs["directional_policy"],
            primitive_registry_path=inputs["primitive_registry"],
            access_ledger=ledger,
        )
        _require_zero_hold(geometry.commands_by_action)
        schedule = v4.load_schedule_indices_v1(
            inputs["schedule"], raw_indexes=raw, access_ledger=ledger
        )

        phase = "materialize_all_development_roles_in_memory"
        role_index = {
            str(pair["content_sha256"]): index
            for role in ROLE_ORDER
            for index, pair in enumerate(
                pair for pair in raw.pairs if pair["dataset_role"] == role
            )
        }
        rows_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
        for scene_id in sorted(
            raw.shard_by_scene,
            key=lambda scene: (
                ROLE_ORDER.index(str(raw.shard_by_scene[scene]["dataset_role"])), scene
            ),
        ):
            scene_manifest, states = v4.load_joined_scene_v1(
                raw_indexes=raw,
                scene_id=scene_id,
                source_records=source_records[scene_id],
                repository_root=root,
                access_ledger=ledger,
            )
            for state in states:
                role = str(state.pair["dataset_role"])
                rows_by_role[role].extend(
                    label_state_v1(
                        pair=state.pair,
                        endpoint=state.endpoint,
                        source_pose_world=state.source_pose_world,
                        source_line_number=state.source_line_number,
                        scene_manifest=scene_manifest,
                        footprint=geometry.footprint,
                        commands_by_action=geometry.commands_by_action,
                        source_bindings=state.source_bindings,
                        role_state_index=role_index[str(state.pair["content_sha256"])],
                    )
                )
        for role in ROLE_ORDER:
            rows_by_role[role].sort(
                key=lambda row: (row["role_state_index"], row["action_index"])
            )

        phase = "enforce_scientific_gates"
        preflight = summarize_preflight_v1(rows_by_role, schedule)
        if ledger != expected_access_ledger_v1():
            raise SweptProgressLabelError("exact access ledger changed")

        phase = "atomic_publish"
        manifest, payloads = _success_payloads(
            rows_by_role, binding=binding, preflight=preflight, ledger=ledger
        )
        _atomic_publish(output, payloads)
        return manifest
    except Exception as error:
        if phase != "atomic_publish" and not output.exists() and not output.is_symlink():
            receipt = _failure_receipt(phase=phase, error=error, ledger=ledger)
            try:
                _atomic_publish(
                    output,
                    {"failure.json": v4.canonical_json_bytes(receipt) + b"\n"},
                )
            except Exception:
                pass
        raise


__all__ = [
    "ACTION_ORDER",
    "INFORMATIVE_FLOORS",
    "NON_HOLD_ACTIONS",
    "OUTPUT_RELATIVE_PATH",
    "ROLE_ORDER",
    "ROW_SCHEMA",
    "SCHEDULE_ACTION_PARTICIPATION_FLOOR",
    "SCHEDULE_INFORMATIVE_FLOOR",
    "SCHEDULE_PRESENTATION_COUNT",
    "SELECTION_FAMILY_FLOOR",
    "SweptProgressLabelError",
    "build_from_v4_binding_v1",
    "enforce_preflight_gates_v1",
    "expected_access_ledger_v1",
    "label_state_v1",
    "summarize_preflight_v1",
    "swept_progress_prefix_v1",
]
