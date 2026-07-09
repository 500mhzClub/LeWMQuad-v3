"""Strict, standalone scoring for Go2 closed-loop navigation results.

The scorer does not import the closed-loop benchmark. It reconstructs poses
from the result log, resolves claimed targets against the exact scene
manifest, and recomputes geometry-dependent metrics under a versioned
geometry contract.

Legacy result limitations are returned in ``StrictResultScore.limitations``.
In particular, old claim rows omit their pose and old normal rows round
``post_xy``. A legacy claim can therefore be checked at the prior logged pose,
but the score records that this is not full-precision event telemetry. Missing
logs or unresolved targets are never replaced with stored proxy distances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lewm.benchmarks.generalization_protocol import (
    StrictClaimObservation,
    fixed_spawn_audit_config_from_geometry_contract,
    reachable_area_normalized_coverage,
    strict_ground_truth_claim,
    supercover_segment_cells,
)
from lewm.planning.geometry_contract import GeometryContract, load_geometry_contract
from lewm_worlds.fixed_spawn_audit import FixedSpawnAuditReport, audit_fixed_spawn
from lewm_worlds.manifest import (
    BoxObject,
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid
from lewm_worlds.scene_graph import SceneGraph


_SCORE_SCHEMA = "lewm_navigation_strict_result_score_v0"
_SEALED_SCHEMA = "lewm_navigation_sealed_test_manifest_v0"
_COLLISION_LOG_KEYS = (
    "collision",
    "collided",
    "contact",
    "body_clearance_contact",
    "body_clearance_violation",
)


class SealedEvaluationAuthorizationError(PermissionError):
    """Raised before scoring when a sealed test was not explicitly authorized."""


@dataclass(frozen=True)
class TickTrajectoryPoint:
    """Pre/post xy reconstructed for one logged policy tick."""

    tick: int
    state: str
    pre_xy_m: tuple[float, float] | None
    post_xy_m: tuple[float, float] | None
    pre_source: str | None
    post_source: str | None


@dataclass(frozen=True)
class ClaimEventVerification:
    """Independent verification of one proxy-accepted claim event."""

    event_index: int
    tick: int | None
    source: str
    target_reference: str | None
    target_object_id: str | None
    pose_xy_m: tuple[float, float] | None
    pose_source: str | None
    proxy_distance_m: float | None
    true_distance_m: float | None
    within_claim_radius: bool | None
    line_of_sight: bool | None
    strict_accepted: bool | None
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProxyStrictDiscrepancy:
    """One disagreement between reported proxy state and strict reconstruction."""

    code: str
    detail: str
    tick: int | None = None
    target_object_id: str | None = None


@dataclass(frozen=True)
class StrictResultScore:
    """Serializable result of strict offline scoring."""

    schema: str
    source_schema: str
    source_payload_sha256: str
    scene_id: str
    scene_manifest_sha256: str
    geometry_contract_sha256: str
    sealed_final_evaluation_authorized: bool
    result_ticks_used: int | None
    log_row_count: int
    trajectory_complete: bool
    trajectory: tuple[TickTrajectoryPoint, ...]
    proxy_claim_event_count: int
    strict_claim_evaluation_complete: bool
    claim_verifications: tuple[ClaimEventVerification, ...]
    strict_accepted_claim_event_count: int
    strict_claimed_object_ids: tuple[str, ...]
    target_count: int
    strict_all_targets_complete: bool
    strict_four_of_four_complete: bool | None
    strict_completion_tick: int | None
    coverage_final_fraction: float | None
    coverage_normalized_auc: float | None
    coverage_visited_reachable_cell_count: int | None
    coverage_reachable_cell_count: int | None
    coverage_unique_pose_cell_count: int | None
    coverage_unique_swept_cell_count: int | None
    canonical_geometry_collision_ticks: tuple[int, ...] | None
    canonical_minimum_clearance_m: float | None
    logged_collision_ticks: tuple[int, ...] | None
    logged_stall_ticks: tuple[int, ...] | None
    logged_hard_stall_ticks: tuple[int, ...] | None
    proxy_collision_count: int | None
    proxy_stall_count: int | None
    proxy_hard_stall_count: int | None
    proxy_claimed: bool | None
    proxy_success: bool | None
    discrepancies: tuple[ProxyStrictDiscrepancy, ...]
    limitations: tuple[str, ...]
    score_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _NormalizedPayload:
    result: Mapping[str, Any]
    log: tuple[Mapping[str, Any], ...]
    source_schema: str
    limitations: tuple[str, ...]


def score_result_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    scene_manifest: SceneManifest,
    geometry_contract: GeometryContract,
    benchmark_manifest: Mapping[str, Any] | None = None,
    authorize_sealed_final_evaluation: bool = False,
) -> StrictResultScore:
    """Score one result payload against exact scene and geometry inputs.

    ``authorize_sealed_final_evaluation`` is deliberately a per-call flag. No
    environment variable, config file, or persisted state can authorize later
    calls accidentally.
    """

    sealed_context = _is_sealed_context(scene_manifest, benchmark_manifest)
    if sealed_context and not authorize_sealed_final_evaluation:
        raise SealedEvaluationAuthorizationError(
            "sealed-test scoring requires explicit one-shot final-evaluation "
            "authorization"
        )

    normalized = _normalize_payload(payload)
    result = normalized.result
    limitations = list(normalized.limitations)
    discrepancies: list[ProxyStrictDiscrepancy] = []

    reported_scene = result.get("scene", result.get("scene_id"))
    if reported_scene is not None and str(reported_scene) != scene_manifest.scene_id:
        raise ValueError(
            "result/manifest scene mismatch: "
            f"{reported_scene!r} != {scene_manifest.scene_id!r}"
        )

    audit_config = fixed_spawn_audit_config_from_geometry_contract(
        geometry_contract
    )
    audit = audit_fixed_spawn(scene_manifest, config=audit_config)
    trajectory, trajectory_complete, trajectory_limitations = _reconstruct_trajectory(
        normalized.log,
        result=result,
        scene_manifest=scene_manifest,
    )
    limitations.extend(trajectory_limitations)
    by_tick = {point.tick: point for point in trajectory}

    result_ticks_used = _optional_int(result.get("ticks_used"))
    if result_ticks_used is not None and result_ticks_used != len(normalized.log):
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="ticks_used_log_length_mismatch",
                detail=(
                    f"result ticks_used={result_ticks_used}, "
                    f"log rows={len(normalized.log)}"
                ),
            )
        )

    claim_rows, claim_source, claim_discrepancies = _accepted_claim_events(
        result,
        normalized.log,
    )
    discrepancies.extend(claim_discrepancies)
    scene_graph = SceneGraph(scene_manifest)
    alias_map = _landmark_alias_map(scene_manifest)
    verifications: list[ClaimEventVerification] = []
    strict_first_tick: dict[str, int | None] = {}
    for index, event in enumerate(claim_rows):
        verification, event_discrepancies, event_limitations = _verify_claim_event(
            event,
            event_index=index,
            source=claim_source,
            result=result,
            trajectory_by_tick=by_tick,
            scene_manifest=scene_manifest,
            scene_graph=scene_graph,
            alias_map=alias_map,
            claim_radius_m=geometry_contract.visibility_and_claim.claim_radius_m,
            distractors_occlude=bool(
                geometry_contract.configuration_space.distractors_are_obstacles
            ),
        )
        verifications.append(verification)
        discrepancies.extend(event_discrepancies)
        limitations.extend(event_limitations)
        if verification.strict_accepted and verification.target_object_id is not None:
            if verification.target_object_id in strict_first_tick:
                discrepancies.append(
                    ProxyStrictDiscrepancy(
                        code="duplicate_proxy_claim",
                        detail="target was proxy-claimed more than once",
                        tick=verification.tick,
                        target_object_id=verification.target_object_id,
                    )
                )
            else:
                strict_first_tick[verification.target_object_id] = verification.tick

    strict_ids = tuple(sorted(strict_first_tick))
    target_ids = tuple(
        sorted(landmark.object_id for landmark in scene_manifest.landmarks)
    )
    strict_all_complete = bool(target_ids) and set(target_ids).issubset(
        strict_first_tick
    )
    strict_four_complete = (
        strict_all_complete if len(target_ids) == 4 else None
    )
    if len(target_ids) != 4:
        limitations.append(
            f"four_of_four_not_applicable:manifest_has_{len(target_ids)}_landmarks"
        )
    completion_ticks = (
        [strict_first_tick[target_id] for target_id in target_ids]
        if strict_all_complete
        else []
    )
    if strict_all_complete and any(tick is None for tick in completion_ticks):
        limitations.append("completion_tick_unavailable_for_untimed_claim_event")
        completion_tick = None
    else:
        completion_tick = (
            max(tick for tick in completion_ticks if tick is not None)
            if completion_ticks
            else None
        )

    proxy_claimed = _optional_bool(result.get("claimed"))
    proxy_success = _optional_bool(result.get("success"))
    proxy_claim_ids, proxy_id_limitations = _proxy_claimed_object_ids(
        result,
        alias_map=alias_map,
    )
    limitations.extend(proxy_id_limitations)
    if proxy_claim_ids is not None and set(proxy_claim_ids) != set(strict_ids):
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="proxy_strict_claimed_targets_mismatch",
                detail=(
                    f"proxy={sorted(proxy_claim_ids)}, strict={list(strict_ids)}"
                ),
            )
        )
    requested_ids, requested_limitations = _requested_target_ids(
        result,
        alias_map=alias_map,
        all_target_ids=target_ids,
    )
    limitations.extend(requested_limitations)
    strict_requested_complete = bool(requested_ids) and set(requested_ids).issubset(
        strict_first_tick
    )
    if proxy_claimed is True and not strict_requested_complete:
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="proxy_claimed_true_without_strict_completion",
                detail=(
                    "result claimed=true but requested targets were not "
                    "strictly verified"
                ),
            )
        )
    if proxy_success is True and not strict_requested_complete:
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="proxy_success_true_without_strict_completion",
                detail=(
                    "result success=true but requested targets were not "
                    "strictly verified"
                ),
            )
        )

    coverage_values = _score_coverage(
        trajectory,
        trajectory_complete=trajectory_complete,
        audit=audit,
        limitations=limitations,
    )
    (
        coverage_final,
        coverage_auc,
        coverage_visited,
        coverage_reachable,
        coverage_pose_cells,
        coverage_swept_cells,
    ) = coverage_values

    (
        geometry_collision_ticks,
        minimum_clearance,
    ) = _canonical_geometry_collisions(
        trajectory,
        trajectory_complete=trajectory_complete,
        scene_manifest=scene_manifest,
        geometry_contract=geometry_contract,
        limitations=limitations,
    )
    logged_collisions = _logged_boolean_ticks(
        normalized.log,
        keys=_COLLISION_LOG_KEYS,
        non_action_claim_rows_are_false=True,
    )
    if logged_collisions is None:
        limitations.append("collision_tick_fields_missing")
    logged_stalls = _logged_boolean_ticks(
        normalized.log,
        keys=("stalled",),
        non_action_claim_rows_are_false=True,
    )
    if logged_stalls is None:
        limitations.append("stall_tick_fields_missing")
    logged_hard_stalls = _logged_boolean_ticks(
        normalized.log,
        keys=("hard_stalled",),
        non_action_claim_rows_are_false=True,
    )
    if logged_hard_stalls is None:
        limitations.append("hard_stall_tick_fields_missing")

    wall_metrics = result.get("wall_metrics")
    metrics = wall_metrics if isinstance(wall_metrics, Mapping) else {}
    proxy_collision_count = _first_int(
        metrics,
        ("collision_count", "body_clearance_contact_events", "collisions"),
    )
    proxy_stall_count = _first_int(metrics, ("contact_like_stalls", "stall_count"))
    proxy_hard_stall_count = _first_int(
        metrics,
        ("hard_contact_like_stalls", "hard_stall_count"),
    )
    _compare_logged_summary_count(
        discrepancies,
        code="proxy_collision_count_mismatch",
        label="collision",
        proxy_count=proxy_collision_count,
        logged_ticks=logged_collisions,
    )
    _compare_logged_summary_count(
        discrepancies,
        code="proxy_stall_count_mismatch",
        label="stall",
        proxy_count=proxy_stall_count,
        logged_ticks=logged_stalls,
    )
    _compare_logged_summary_count(
        discrepancies,
        code="proxy_hard_stall_count_mismatch",
        label="hard stall",
        proxy_count=proxy_hard_stall_count,
        logged_ticks=logged_hard_stalls,
    )
    if (
        geometry_collision_ticks is not None
        and logged_collisions is not None
        and set(geometry_collision_ticks) != set(logged_collisions)
    ):
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="logged_canonical_collision_ticks_differ",
                detail=(
                    f"logged={list(logged_collisions)}, "
                    f"canonical={list(geometry_collision_ticks)}"
                ),
            )
        )

    claim_evaluation_complete = all(
        verification.strict_accepted is not None for verification in verifications
    )
    limitations = _unique_strings(limitations)
    score_complete = bool(
        trajectory_complete
        and claim_evaluation_complete
        and (not strict_all_complete or completion_tick is not None)
        and coverage_final is not None
        and geometry_collision_ticks is not None
        and logged_stalls is not None
    )
    return StrictResultScore(
        schema=_SCORE_SCHEMA,
        source_schema=normalized.source_schema,
        source_payload_sha256=_json_sha256(payload),
        scene_id=scene_manifest.scene_id,
        scene_manifest_sha256=manifest_sha256(scene_manifest),
        geometry_contract_sha256=geometry_contract.sha256,
        sealed_final_evaluation_authorized=bool(
            sealed_context and authorize_sealed_final_evaluation
        ),
        result_ticks_used=result_ticks_used,
        log_row_count=len(normalized.log),
        trajectory_complete=trajectory_complete,
        trajectory=trajectory,
        proxy_claim_event_count=len(claim_rows),
        strict_claim_evaluation_complete=claim_evaluation_complete,
        claim_verifications=tuple(verifications),
        strict_accepted_claim_event_count=sum(
            verification.strict_accepted is True for verification in verifications
        ),
        strict_claimed_object_ids=strict_ids,
        target_count=len(target_ids),
        strict_all_targets_complete=strict_all_complete,
        strict_four_of_four_complete=strict_four_complete,
        strict_completion_tick=completion_tick,
        coverage_final_fraction=coverage_final,
        coverage_normalized_auc=coverage_auc,
        coverage_visited_reachable_cell_count=coverage_visited,
        coverage_reachable_cell_count=coverage_reachable,
        coverage_unique_pose_cell_count=coverage_pose_cells,
        coverage_unique_swept_cell_count=coverage_swept_cells,
        canonical_geometry_collision_ticks=geometry_collision_ticks,
        canonical_minimum_clearance_m=minimum_clearance,
        logged_collision_ticks=logged_collisions,
        logged_stall_ticks=logged_stalls,
        logged_hard_stall_ticks=logged_hard_stalls,
        proxy_collision_count=proxy_collision_count,
        proxy_stall_count=proxy_stall_count,
        proxy_hard_stall_count=proxy_hard_stall_count,
        proxy_claimed=proxy_claimed,
        proxy_success=proxy_success,
        discrepancies=tuple(discrepancies),
        limitations=tuple(limitations),
        score_complete=score_complete,
    )


def _normalize_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> _NormalizedPayload:
    limitations: list[str] = []
    if isinstance(payload, Mapping):
        nested_result = payload.get("result")
        result = nested_result if isinstance(nested_result, Mapping) else payload
        raw_log = payload.get("log")
        if raw_log is None and isinstance(result.get("log"), list):
            raw_log = result.get("log")
        source_schema = str(
            payload.get("schema")
            or result.get("schema")
            or "legacy_go2_closed_loop_result"
        )
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        result = {}
        raw_log = payload
        source_schema = "legacy_log_only"
        limitations.append("result_summary_missing_log_only_payload")
    else:
        raise TypeError("result payload must be a JSON object or log-row sequence")

    if raw_log is None:
        limitations.append("per_tick_log_missing")
        log: tuple[Mapping[str, Any], ...] = ()
    elif not isinstance(raw_log, Sequence) or isinstance(raw_log, (str, bytes)):
        limitations.append("per_tick_log_is_not_a_sequence")
        log = ()
    else:
        rows: list[Mapping[str, Any]] = []
        for index, row in enumerate(raw_log):
            if not isinstance(row, Mapping):
                limitations.append(f"log_row_{index}_is_not_an_object")
                continue
            rows.append(row)
        log = tuple(rows)
    return _NormalizedPayload(
        result=result,
        log=log,
        source_schema=source_schema,
        limitations=tuple(limitations),
    )


def _reconstruct_trajectory(
    log: Sequence[Mapping[str, Any]],
    *,
    result: Mapping[str, Any],
    scene_manifest: SceneManifest,
) -> tuple[tuple[TickTrajectoryPoint, ...], bool, tuple[str, ...]]:
    limitations: list[str] = []
    parsed: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, row in enumerate(log):
        tick = _optional_int(row.get("tick"))
        if tick is None:
            limitations.append(f"log_row_{index}_missing_integer_tick")
            continue
        parsed.append((tick, index, row))
    if not parsed:
        if log:
            limitations.append("trajectory_has_no_valid_tick_rows")
        return (), False, tuple(limitations)

    original_ticks = [tick for tick, _, _ in parsed]
    if original_ticks != sorted(original_ticks):
        limitations.append("log_ticks_out_of_order_sorted_for_scoring")
        parsed.sort(key=lambda item: (item[0], item[1]))
    if len(set(original_ticks)) != len(original_ticks):
        limitations.append("duplicate_log_ticks_make_trajectory_ambiguous")

    first_tick, _, first_row = parsed[0]
    first_pre, first_pre_source = _row_xy(
        first_row,
        ("pre_xy", "pose_xy", "pose_before_xy", "robot_xy_m"),
    )
    if first_pre is None:
        slice_start = result.get("wall_metrics", {})
        slice_start = (
            slice_start.get("slice_start")
            if isinstance(slice_start, Mapping)
            else None
        )
        if isinstance(slice_start, Mapping):
            first_pre = _xy(slice_start.get("start_xy"))
            if first_pre is not None:
                first_pre_source = "result.wall_metrics.slice_start.start_xy"
        if first_pre is None and first_tick == 0:
            first_pre = (
                float(scene_manifest.spawn.xyz_m[0]),
                float(scene_manifest.spawn.xyz_m[1]),
            )
            first_pre_source = "scene_manifest.spawn"
    if first_pre is None:
        limitations.append("initial_pose_unavailable_for_nonzero_tick_log")

    points: list[TickTrajectoryPoint] = []
    previous_tick: int | None = None
    previous_post = first_pre
    previous_post_source = first_pre_source
    complete = first_pre is not None
    seen_ticks: set[int] = set()
    for tick, index, row in parsed:
        if tick in seen_ticks:
            limitations.append(f"duplicate_tick_{tick}_row_{index}_ignored")
            complete = False
            continue
        seen_ticks.add(tick)
        explicit_pre, explicit_pre_source = _row_xy(
            row,
            ("pre_xy", "pose_xy", "pose_before_xy", "robot_xy_m"),
        )
        if previous_tick is not None and tick != previous_tick + 1:
            limitations.append(f"trajectory_tick_gap:{previous_tick}->{tick}")
            complete = False
            if explicit_pre is None:
                previous_post = None
                previous_post_source = None
        pre = explicit_pre if explicit_pre is not None else previous_post
        pre_source = (
            explicit_pre_source
            if explicit_pre is not None
            else (
                first_pre_source
                if previous_tick is None
                else (
                    None
                    if previous_post_source is None
                    else f"previous_tick:{previous_post_source}"
                )
            )
        )
        if (
            explicit_pre is not None
            and previous_post is not None
            and previous_tick is not None
            and math.dist(explicit_pre, previous_post) > 1e-3
        ):
            limitations.append(f"trajectory_pose_discontinuity_at_tick_{tick}")
            complete = False

        post, post_source = _row_xy(
            row,
            ("post_xy", "pose_after_xy", "post_pose_xy"),
        )
        state = str(row.get("state", row.get("event", ""))).upper()
        if post is None and state == "CLAIM" and pre is not None:
            post = pre
            post_source = "stationary_claim_event"
        if post is None:
            limitations.append(f"post_pose_missing_at_tick_{tick}")
            complete = False
        if pre is None:
            limitations.append(f"pre_pose_missing_at_tick_{tick}")
            complete = False
        if post_source == "post_xy":
            limitations.append(
                "legacy_post_xy_precision_is_rounded_or_unspecified"
            )
        points.append(
            TickTrajectoryPoint(
                tick=tick,
                state=state,
                pre_xy_m=pre,
                post_xy_m=post,
                pre_source=pre_source,
                post_source=post_source,
            )
        )
        previous_tick = tick
        previous_post = post
        previous_post_source = post_source

    final_xy = _xy(result.get("final_xy"))
    if final_xy is not None and points and points[-1].post_xy_m is not None:
        if math.dist(final_xy, points[-1].post_xy_m) > 0.005:
            limitations.append(
                "result_final_xy_disagrees_with_reconstructed_trajectory"
            )
            complete = False
    return tuple(points), complete, tuple(_unique_strings(limitations))


def _accepted_claim_events(
    result: Mapping[str, Any],
    log: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    str,
    tuple[ProxyStrictDiscrepancy, ...],
]:
    discrepancies: list[ProxyStrictDiscrepancy] = []
    result_events = result.get("beacon_claims", result.get("claim_events"))
    accepted_result_events: list[Mapping[str, Any]] = []
    if isinstance(result_events, Sequence) and not isinstance(
        result_events, (str, bytes)
    ):
        accepted_result_events = [
            event
            for event in result_events
            if isinstance(event, Mapping) and _event_is_proxy_accepted(event)
        ]
    log_events = [row for row in log if _is_claim_row(row)]
    if accepted_result_events:
        if log_events and len(log_events) != len(accepted_result_events):
            discrepancies.append(
                ProxyStrictDiscrepancy(
                    code="result_log_claim_event_count_mismatch",
                    detail=(
                        f"result events={len(accepted_result_events)}, "
                        f"log claim rows={len(log_events)}"
                    ),
                )
            )
        return (
            tuple(accepted_result_events),
            "result.beacon_claims",
            tuple(discrepancies),
        )
    return tuple(log_events), "log.CLAIM", tuple(discrepancies)


def _verify_claim_event(
    event: Mapping[str, Any],
    *,
    event_index: int,
    source: str,
    result: Mapping[str, Any],
    trajectory_by_tick: Mapping[int, TickTrajectoryPoint],
    scene_manifest: SceneManifest,
    scene_graph: SceneGraph,
    alias_map: Mapping[str, BoxObject],
    claim_radius_m: float,
    distractors_occlude: bool,
) -> tuple[
    ClaimEventVerification,
    tuple[ProxyStrictDiscrepancy, ...],
    tuple[str, ...],
]:
    discrepancies: list[ProxyStrictDiscrepancy] = []
    limitations: list[str] = []
    tick = _optional_int(event.get("tick"))
    target_reference = _claim_target_reference(event, result=result)
    landmark = (
        alias_map.get(target_reference.lower())
        if target_reference is not None
        else None
    )
    if landmark is None:
        limitations.append(
            f"claim_event_{event_index}_target_unresolved:{target_reference}"
        )

    event_pose, event_pose_source = _row_xy(
        event,
        ("claim_pose_xy", "pose_xy", "pre_xy", "robot_xy_m"),
    )
    point = trajectory_by_tick.get(tick) if tick is not None else None
    if point is not None and point.pre_xy_m is not None:
        pose = point.pre_xy_m
        pose_source = (
            None
            if point.pre_source is None
            else f"trajectory.pre:{point.pre_source}"
        )
        if event_pose is not None and math.dist(event_pose, pose) > 1e-3:
            discrepancies.append(
                ProxyStrictDiscrepancy(
                    code="claim_pose_trajectory_mismatch",
                    detail=(
                        f"event pose={event_pose}, trajectory pre={pose}"
                    ),
                    tick=tick,
                    target_object_id=(None if landmark is None else landmark.object_id),
                )
            )
    else:
        pose = event_pose
        pose_source = event_pose_source
    if pose is None:
        limitations.append(f"claim_event_{event_index}_pose_unavailable")

    proxy_distance = _optional_float(
        event.get("dist_to_target_m", event.get("distance_m"))
    )
    true_distance: float | None = None
    within_radius: bool | None = None
    line_of_sight: bool | None = None
    strict_accepted: bool | None = None
    rejection_reasons: list[str] = []
    if landmark is None:
        rejection_reasons.append("target_unresolved")
    if pose is None:
        rejection_reasons.append("claim_pose_unavailable")
    if landmark is not None and pose is not None:
        target_xy = (
            float(landmark.center_xyz_m[0]),
            float(landmark.center_xyz_m[1]),
        )
        line_of_sight = _true_line_of_sight(
            scene_graph,
            scene_manifest,
            pose,
            target_xy,
            exclude_landmark_xy=target_xy,
            distractors_occlude=distractors_occlude,
        )
        result_claim = strict_ground_truth_claim(
            StrictClaimObservation(
                target_id=landmark.object_id,
                robot_xy_m=pose,
                target_xy_m=target_xy,
                line_of_sight=line_of_sight,
            ),
            claim_radius_m=claim_radius_m,
        )
        true_distance = result_claim.distance_m
        within_radius = result_claim.within_claim_radius
        strict_accepted = result_claim.accepted
        legacy_radius_uncertainty = (
            1e-3
            if pose_source is not None and "post_xy" in pose_source
            else 0.0
        )
        if (
            line_of_sight
            and abs(true_distance - float(claim_radius_m))
            <= legacy_radius_uncertainty
        ):
            strict_accepted = None
            rejection_reasons.append("legacy_pose_precision_straddles_claim_radius")
            limitations.append(
                f"claim_event_{event_index}_requires_full_precision_pose"
            )
        if not within_radius:
            rejection_reasons.append("outside_claim_radius")
        if not line_of_sight:
            rejection_reasons.append("line_of_sight_blocked")
        if proxy_distance is not None and abs(proxy_distance - true_distance) > 0.002:
            discrepancies.append(
                ProxyStrictDiscrepancy(
                    code="proxy_claim_distance_mismatch",
                    detail=(
                        f"proxy={proxy_distance:.6f}m, "
                        f"reconstructed={true_distance:.6f}m"
                    ),
                    tick=tick,
                    target_object_id=landmark.object_id,
                )
            )
        if strict_accepted is False:
            discrepancies.append(
                ProxyStrictDiscrepancy(
                    code="proxy_claim_rejected_by_strict_geometry",
                    detail=",".join(rejection_reasons),
                    tick=tick,
                    target_object_id=landmark.object_id,
                )
            )
        elif strict_accepted is None:
            discrepancies.append(
                ProxyStrictDiscrepancy(
                    code="proxy_claim_unverifiable_at_strict_boundary",
                    detail=",".join(rejection_reasons),
                    tick=tick,
                    target_object_id=landmark.object_id,
                )
            )
    else:
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code="proxy_claim_unverifiable",
                detail=",".join(rejection_reasons),
                tick=tick,
                target_object_id=(None if landmark is None else landmark.object_id),
            )
        )
    return (
        ClaimEventVerification(
            event_index=event_index,
            tick=tick,
            source=source,
            target_reference=target_reference,
            target_object_id=(None if landmark is None else landmark.object_id),
            pose_xy_m=pose,
            pose_source=pose_source,
            proxy_distance_m=proxy_distance,
            true_distance_m=true_distance,
            within_claim_radius=within_radius,
            line_of_sight=line_of_sight,
            strict_accepted=strict_accepted,
            rejection_reasons=tuple(rejection_reasons),
        ),
        tuple(discrepancies),
        tuple(limitations),
    )


def _score_coverage(
    trajectory: Sequence[TickTrajectoryPoint],
    *,
    trajectory_complete: bool,
    audit: FixedSpawnAuditReport,
    limitations: list[str],
) -> tuple[
    float | None,
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
]:
    if not trajectory_complete or not trajectory:
        limitations.append("coverage_unavailable_without_complete_trajectory")
        return (None, None, None, None, None, None)
    if audit.coverage_reachable_cell_count <= 0:
        limitations.append("coverage_unavailable_empty_reachable_component")
        return (None, None, None, None, None, None)
    assert trajectory[0].pre_xy_m is not None
    positions = [trajectory[0].pre_xy_m]
    positions.extend(point.post_xy_m for point in trajectory)
    if any(position is None for position in positions):
        limitations.append("coverage_unavailable_trajectory_pose_gap")
        return (None, None, None, None, None, None)
    concrete_positions = [position for position in positions if position is not None]
    final_metric = reachable_area_normalized_coverage(
        concrete_positions,
        audit=audit,
    )
    visited: set[tuple[int, int]] = set()
    fractions: list[float] = []
    reachable = set(audit.coverage_reachable_cells)
    denominator = audit.coverage_reachable_cell_count
    previous: tuple[float, float] | None = None
    for position in concrete_positions:
        if previous is None:
            swept = {audit.world_to_coverage_grid(position)}
        else:
            swept = supercover_segment_cells(
                previous,
                position,
                origin_xy_m=audit.coverage_grid_origin_xy_m,
                cell_size_m=audit.config.coverage_cell_size_m,
            )
        visited.update(swept & reachable)
        fractions.append(len(visited) / denominator)
        previous = position
    if len(fractions) == 1:
        normalized_auc = fractions[0]
    else:
        normalized_auc = sum(
            0.5 * (left + right)
            for left, right in zip(fractions, fractions[1:])
        ) / (len(fractions) - 1)
    return (
        final_metric.fraction,
        float(normalized_auc),
        final_metric.visited_reachable_cell_count,
        final_metric.reachable_cell_count,
        final_metric.unique_pose_cell_count,
        final_metric.unique_swept_cell_count,
    )


def _canonical_geometry_collisions(
    trajectory: Sequence[TickTrajectoryPoint],
    *,
    trajectory_complete: bool,
    scene_manifest: SceneManifest,
    geometry_contract: GeometryContract,
    limitations: list[str],
) -> tuple[tuple[int, ...] | None, float | None]:
    if not trajectory_complete or not trajectory:
        limitations.append(
            "canonical_collisions_unavailable_without_complete_trajectory"
        )
        return None, None
    config = geometry_contract.configuration_space
    grid = InflatedOccupancyGrid(
        scene_manifest,
        cell_size_m=float(config.oracle_cell_size_m),
        inflation_m=float(config.body_inflation_radius_m),
        treat_landmarks_as_obstacles=bool(config.landmarks_are_obstacles),
        treat_distractors_as_obstacles=bool(config.distractors_are_obstacles),
    )
    max_step = float(
        geometry_contract.kinematic_execution.maximum_translation_substep_m
    )
    radius = float(config.body_inflation_radius_m)
    (x_lo, y_lo), (x_hi, y_hi) = scene_manifest.world_bounds_xy_m
    collision_ticks: list[int] = []
    minimum_clearance = math.inf
    for point in trajectory:
        if point.pre_xy_m is None or point.post_xy_m is None:
            return None, None
        distance = math.dist(point.pre_xy_m, point.post_xy_m)
        steps = max(1, int(math.ceil(distance / max_step)))
        collided = False
        for step in range(steps + 1):
            alpha = step / steps
            xy = (
                point.pre_xy_m[0]
                + alpha * (point.post_xy_m[0] - point.pre_xy_m[0]),
                point.pre_xy_m[1]
                + alpha * (point.post_xy_m[1] - point.pre_xy_m[1]),
            )
            clearance = float(grid.obstacle_clearance_m(xy) - radius)
            minimum_clearance = min(minimum_clearance, clearance)
            in_bounds = (
                float(x_lo) <= xy[0] <= float(x_hi)
                and float(y_lo) <= xy[1] <= float(y_hi)
            )
            if not in_bounds or clearance < 0.0:
                collided = True
        if collided:
            collision_ticks.append(point.tick)
    return (
        tuple(collision_ticks),
        None if math.isinf(minimum_clearance) else float(minimum_clearance),
    )


def _logged_boolean_ticks(
    log: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    non_action_claim_rows_are_false: bool,
) -> tuple[int, ...] | None:
    evidence_available = False
    ticks: list[int] = []
    for row in log:
        tick = _optional_int(row.get("tick"))
        if tick is None:
            continue
        present = [key for key in keys if key in row]
        if present:
            evidence_available = True
            if any(bool(row.get(key)) for key in present):
                ticks.append(tick)
        elif non_action_claim_rows_are_false and _is_claim_row(row):
            continue
    return tuple(ticks) if evidence_available else None


def _compare_logged_summary_count(
    discrepancies: list[ProxyStrictDiscrepancy],
    *,
    code: str,
    label: str,
    proxy_count: int | None,
    logged_ticks: tuple[int, ...] | None,
) -> None:
    if proxy_count is None or logged_ticks is None:
        return
    if proxy_count != len(logged_ticks):
        discrepancies.append(
            ProxyStrictDiscrepancy(
                code=code,
                detail=(
                    f"summary {label} count={proxy_count}, "
                    f"per-tick count={len(logged_ticks)}"
                ),
            )
        )


def _proxy_claimed_object_ids(
    result: Mapping[str, Any],
    *,
    alias_map: Mapping[str, BoxObject],
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    raw = result.get("claimed_colors", result.get("claimed_target_ids"))
    if raw is None:
        return None, ("proxy_claimed_target_list_missing",)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None, ("proxy_claimed_target_list_malformed",)
    ids: set[str] = set()
    limitations: list[str] = []
    for value in raw:
        landmark = alias_map.get(str(value).lower())
        if landmark is None:
            limitations.append(f"proxy_claimed_target_unresolved:{value}")
        else:
            ids.add(landmark.object_id)
    return tuple(sorted(ids)), tuple(limitations)


def _requested_target_ids(
    result: Mapping[str, Any],
    *,
    alias_map: Mapping[str, BoxObject],
    all_target_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = result.get("target_colors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        single = result.get("target_color")
        raw = [] if single is None else [single]
    if not raw or any(str(value).lower() == "all" for value in raw):
        return tuple(all_target_ids), ()
    ids: set[str] = set()
    limitations: list[str] = []
    for value in raw:
        landmark = alias_map.get(str(value).lower())
        if landmark is None:
            limitations.append(f"requested_target_unresolved:{value}")
        else:
            ids.add(landmark.object_id)
    return tuple(sorted(ids)), tuple(limitations)


def _claim_target_reference(
    event: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
) -> str | None:
    for key in (
        "target_object_id",
        "target_id",
        "object_id",
        "target_color",
        "color",
    ):
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    target_index = _optional_int(event.get("target_index"))
    target_colors = result.get("target_colors")
    if (
        target_index is not None
        and isinstance(target_colors, Sequence)
        and not isinstance(target_colors, (str, bytes))
        and 0 <= target_index < len(target_colors)
    ):
        return str(target_colors[target_index])
    single = result.get("target_color")
    if single is not None and str(single).lower() != "all":
        return str(single)
    return None


def _landmark_alias_map(manifest: SceneManifest) -> dict[str, BoxObject]:
    candidates: dict[str, list[BoxObject]] = {}
    known_colors = ("red", "green", "blue", "yellow")
    for landmark in manifest.landmarks:
        aliases = {
            landmark.object_id.lower(),
            landmark.material_id.lower(),
        }
        material = landmark.material_id.lower()
        if material.startswith("landmark_"):
            aliases.add(material.removeprefix("landmark_"))
        object_id = landmark.object_id.lower()
        for color in known_colors:
            if color in material or color in object_id:
                aliases.add(color)
        for alias in aliases:
            candidates.setdefault(alias, []).append(landmark)
    return {
        alias: landmarks[0]
        for alias, landmarks in candidates.items()
        if len(landmarks) == 1
    }


def _true_line_of_sight(
    scene_graph: SceneGraph,
    manifest: SceneManifest,
    src_xy: tuple[float, float],
    dst_xy: tuple[float, float],
    *,
    exclude_landmark_xy: tuple[float, float],
    distractors_occlude: bool,
) -> bool:
    if not scene_graph.has_line_of_sight(
        src_xy,
        dst_xy,
        exclude_landmark_xy=exclude_landmark_xy,
    ):
        return False
    if not distractors_occlude or manifest.visual_randomization is None:
        return True
    return not any(
        _segment_intersects_box(src_xy, dst_xy, distractor)
        for distractor in manifest.visual_randomization.distractor_objects
    )


def _segment_intersects_box(
    src_xy: tuple[float, float],
    dst_xy: tuple[float, float],
    box: BoxObject,
) -> bool:
    cos_yaw = math.cos(-float(box.yaw_rad))
    sin_yaw = math.sin(-float(box.yaw_rad))

    def local(xy: tuple[float, float]) -> tuple[float, float]:
        dx = float(xy[0]) - float(box.center_xyz_m[0])
        dy = float(xy[1]) - float(box.center_xyz_m[1])
        return (
            cos_yaw * dx - sin_yaw * dy,
            sin_yaw * dx + cos_yaw * dy,
        )

    start = local(src_xy)
    end = local(dst_xy)
    half_extents = (
        0.5 * float(box.size_xyz_m[0]),
        0.5 * float(box.size_xyz_m[1]),
    )
    t_min, t_max = 0.0, 1.0
    for start_axis, end_axis, half_extent in zip(
        start,
        end,
        half_extents,
    ):
        delta = end_axis - start_axis
        if abs(delta) <= 1e-12:
            if abs(start_axis) > half_extent:
                return False
            continue
        enter = (-half_extent - start_axis) / delta
        exit_ = (half_extent - start_axis) / delta
        if enter > exit_:
            enter, exit_ = exit_, enter
        t_min = max(t_min, enter)
        t_max = min(t_max, exit_)
        if t_min > t_max:
            return False
    return True


def _event_is_proxy_accepted(event: Mapping[str, Any]) -> bool:
    if event.get("accepted") is False:
        return False
    return True


def _is_claim_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("state", row.get("event", ""))).upper() == "CLAIM"


def _row_xy(
    row: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[tuple[float, float] | None, str | None]:
    for key in keys:
        if key in row:
            value = _xy(row.get(key))
            if value is not None:
                return value, key
    pose = row.get("pose")
    if isinstance(pose, Mapping):
        for key in keys:
            if key in pose:
                value = _xy(pose.get(key))
                if value is not None:
                    return value, f"pose.{key}"
    return None, None


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        values = (value["x"], value["y"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            return None
        values = (value[0], value[1])
    else:
        return None
    try:
        xy = (float(values[0]), float(values[1]))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(component) for component in xy) else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _first_int(mapping: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in mapping:
            return _optional_int(mapping.get(key))
    return None


def _is_sealed_context(
    scene_manifest: SceneManifest,
    benchmark_manifest: Mapping[str, Any] | None,
) -> bool:
    split = str(scene_manifest.split or "").strip().lower().replace("-", "_")
    if split in {"sealed_test", "test_sealed", "final_sealed_test"}:
        return True
    return bool(
        isinstance(benchmark_manifest, Mapping)
        and benchmark_manifest.get("schema") == _SEALED_SCHEMA
    )


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--authorize-sealed-final-evaluation",
        action="store_true",
        help="One-shot authorization required to score a sealed-test manifest.",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = parse_scene_manifest_dict(
        json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    )
    contract = load_geometry_contract(args.geometry_contract)
    benchmark_manifest = (
        None
        if args.benchmark_manifest is None
        else json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    )
    score = score_result_payload(
        payload,
        scene_manifest=manifest,
        geometry_contract=contract,
        benchmark_manifest=benchmark_manifest,
        authorize_sealed_final_evaluation=bool(
            args.authorize_sealed_final_evaluation
        ),
    )
    rendered = json.dumps(score.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ClaimEventVerification",
    "ProxyStrictDiscrepancy",
    "SealedEvaluationAuthorizationError",
    "StrictResultScore",
    "TickTrajectoryPoint",
    "main",
    "score_result_payload",
]
