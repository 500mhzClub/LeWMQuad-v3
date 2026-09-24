"""Deterministic metadata-only scene-local B4 place schedule for V5."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from lewm.datasets.go2_memory_role_place_triplets_v1 import (
    PlaceTripletRow,
    canonical_json_sha256,
)


SCHEMA_V5 = "lewm_go2_memory_role_scene_local_place_schedule_v5"
SELECTION_SEED_V5 = (
    "lewm_go2_rgb_memory_role_factorized_joint_jepa_v5_scene_local_b4"
)
FAMILIES_V5 = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
EXPECTED_SOURCE_ROWS_BY_FAMILY_V5 = {
    "large_enclosed_maze": 242,
    "local_composite_motifs": 456,
    "loop_alias_stress": 325,
    "medium_enclosed_maze": 323,
    "open_obstacle_field": 528,
    "rough_local_dynamics": 527,
    "small_enclosed_maze": 387,
    "visual_sensor_stress": 412,
}
GROUP_SIZE_V5 = 4
GROUPS_PER_FAMILY_V5 = 100
TOTAL_GROUPS_V5 = len(FAMILIES_V5) * GROUPS_PER_FAMILY_V5
TOTAL_ROW_PRESENTATIONS_V5 = TOTAL_GROUPS_V5 * GROUP_SIZE_V5


class SceneLocalPlaceScheduleError(RuntimeError):
    """The frozen train rows cannot produce the exact V5 schedule."""


@dataclass(frozen=True, slots=True)
class SceneLocalPlaceGroupV5:
    index: int
    family: str
    scene_id: str
    rows: tuple[PlaceTripletRow, ...]


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _score(*parts: object) -> str:
    payload = "\0".join((SELECTION_SEED_V5, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _maximum_unique_rows(place_counts: Sequence[int], group_count: int) -> int:
    return min(
        GROUP_SIZE_V5 * group_count,
        sum(min(count, group_count) for count in place_counts),
    )


def _allocate_scene_group_counts(
    family: str,
    scene_place_rows: Mapping[str, Mapping[str, Sequence[PlaceTripletRow]]],
) -> dict[str, int]:
    eligible = {
        scene_id: tuple(len(rows) for rows in places.values())
        for scene_id, places in scene_place_rows.items()
        if len(places) >= GROUP_SIZE_V5
    }
    if not eligible:
        raise SceneLocalPlaceScheduleError(
            f"{family} has no scene with four distinct place identities"
        )
    assigned = {scene_id: 0 for scene_id in eligible}
    for _ in range(GROUPS_PER_FAMILY_V5):
        ranked: list[tuple[int, int, str, str]] = []
        for scene_id, counts in eligible.items():
            previous = _maximum_unique_rows(counts, assigned[scene_id])
            following = _maximum_unique_rows(counts, assigned[scene_id] + 1)
            ranked.append(
                (
                    -(following - previous),
                    assigned[scene_id],
                    _score("scene-allocation", family, scene_id),
                    scene_id,
                )
            )
        selected = min(ranked)[3]
        assigned[selected] += 1
    return assigned


def _allocate_appearances(
    family: str,
    scene_id: str,
    place_rows: Mapping[str, Sequence[PlaceTripletRow]],
    group_count: int,
) -> tuple[dict[str, int], dict[str, int]]:
    capacities = {
        place: min(len(rows), group_count) for place, rows in place_rows.items()
    }
    unique_target = min(
        GROUP_SIZE_V5 * group_count,
        sum(capacities.values()),
    )
    unique = {place: 0 for place in place_rows}
    for _ in range(unique_target):
        candidates = [
            place for place in place_rows if unique[place] < capacities[place]
        ]
        if not candidates:
            raise SceneLocalPlaceScheduleError("unique-row allocation exhausted early")
        selected = min(
            candidates,
            key=lambda place: (
                unique[place],
                _score("unique-appearance", family, scene_id, place),
            ),
        )
        unique[selected] += 1

    appearances = dict(unique)
    presentation_target = GROUP_SIZE_V5 * group_count
    for _ in range(presentation_target - unique_target):
        candidates = [
            place
            for place in place_rows
            if unique[place] > 0 and appearances[place] < group_count
        ]
        if not candidates:
            raise SceneLocalPlaceScheduleError("repeat-row allocation exhausted early")
        selected = min(
            candidates,
            key=lambda place: (
                appearances[place],
                _score("repeat-appearance", family, scene_id, place),
            ),
        )
        appearances[selected] += 1
    return unique, appearances


def _build_scene_groups(
    family: str,
    scene_id: str,
    place_rows: Mapping[str, Sequence[PlaceTripletRow]],
    group_count: int,
) -> tuple[tuple[PlaceTripletRow, ...], ...]:
    if group_count == 0:
        return ()
    unique, appearances = _allocate_appearances(
        family, scene_id, place_rows, group_count
    )
    ordered_rows = {
        place: tuple(
            sorted(
                rows,
                key=lambda row: _score(
                    "row", family, scene_id, place, row.content_sha256
                ),
            )
        )
        for place, rows in place_rows.items()
    }
    remaining = dict(appearances)
    occurrence = {place: 0 for place in place_rows}
    groups: list[tuple[PlaceTripletRow, ...]] = []
    for group_index in range(group_count):
        candidates = [place for place, count in remaining.items() if count > 0]
        if len(candidates) < GROUP_SIZE_V5:
            raise SceneLocalPlaceScheduleError(
                "place-appearance allocation cannot fill a distinct B4 group"
            )
        selected_places = sorted(
            candidates,
            key=lambda place: (
                -remaining[place],
                _score("group-place", family, scene_id, group_index, place),
            ),
        )[:GROUP_SIZE_V5]
        group: list[PlaceTripletRow] = []
        for place in selected_places:
            offset = occurrence[place]
            unique_count = unique[place]
            rows = ordered_rows[place]
            row = (
                rows[offset]
                if offset < unique_count
                else rows[(offset - unique_count) % unique_count]
            )
            group.append(row)
            occurrence[place] += 1
            remaining[place] -= 1
        groups.append(tuple(group))
    if any(remaining.values()):
        raise SceneLocalPlaceScheduleError("place appearances remain unscheduled")
    return tuple(groups)


def _validate_source_rows(
    rows: tuple[PlaceTripletRow, ...],
) -> dict[str, tuple[PlaceTripletRow, ...]]:
    if len(rows) != sum(EXPECTED_SOURCE_ROWS_BY_FAMILY_V5.values()):
        raise SceneLocalPlaceScheduleError("V5 requires the exact 3,200 train rows")
    if [row.index for row in rows] != list(range(len(rows))):
        raise SceneLocalPlaceScheduleError("train rows left frozen index order")
    if any(
        not isinstance(row, PlaceTripletRow)
        or row.role != "train"
        or row.family not in FAMILIES_V5
        or type(row.scene_id) is not str
        or not row.scene_id
        or not _is_sha256(row.content_sha256)
        or not _is_sha256(row.place_identity_sha256)
        for row in rows
    ):
        raise SceneLocalPlaceScheduleError("train row metadata contract changed")
    if len({row.content_sha256 for row in rows}) != len(rows):
        raise SceneLocalPlaceScheduleError("train rows repeat a content identity")
    by_family = {
        family: tuple(row for row in rows if row.family == family)
        for family in FAMILIES_V5
    }
    counts = {family: len(values) for family, values in by_family.items()}
    if counts != EXPECTED_SOURCE_ROWS_BY_FAMILY_V5:
        raise SceneLocalPlaceScheduleError("frozen train family counts changed")
    return by_family


def build_scene_local_place_schedule_v5(
    train_rows: Sequence[PlaceTripletRow],
) -> tuple[tuple[SceneLocalPlaceGroupV5, ...], dict[str, Any]]:
    """Build exactly 800 same-scene B4 groups without opening RGB leaves."""

    rows = tuple(train_rows)
    by_family = _validate_source_rows(rows)
    family_groups: dict[str, tuple[tuple[PlaceTripletRow, ...], ...]] = {}

    for family in FAMILIES_V5:
        scene_place_rows: dict[str, dict[str, list[PlaceTripletRow]]] = {}
        for row in by_family[family]:
            scene_place_rows.setdefault(row.scene_id, {}).setdefault(
                row.place_identity_sha256, []
            ).append(row)
        allocations = _allocate_scene_group_counts(family, scene_place_rows)
        groups_by_scene = {
            scene_id: list(
                _build_scene_groups(
                    family,
                    scene_id,
                    places,
                    allocations.get(scene_id, 0),
                )
            )
            for scene_id, places in scene_place_rows.items()
            if allocations.get(scene_id, 0)
        }
        scene_order = sorted(
            groups_by_scene,
            key=lambda scene_id: _score("scene-order", family, scene_id),
        )
        interleaved: list[tuple[PlaceTripletRow, ...]] = []
        while len(interleaved) < GROUPS_PER_FAMILY_V5:
            progressed = False
            for scene_id in scene_order:
                if groups_by_scene[scene_id]:
                    interleaved.append(groups_by_scene[scene_id].pop(0))
                    progressed = True
            if not progressed:
                raise SceneLocalPlaceScheduleError("family group interleave exhausted")
        if any(groups_by_scene.values()):
            raise SceneLocalPlaceScheduleError("family group interleave left rows")
        family_groups[family] = tuple(interleaved)

    ordered_groups: list[SceneLocalPlaceGroupV5] = []
    for family_offset in range(GROUPS_PER_FAMILY_V5):
        for family in FAMILIES_V5:
            group_rows = family_groups[family][family_offset]
            scene_ids = {row.scene_id for row in group_rows}
            places = {row.place_identity_sha256 for row in group_rows}
            if len(group_rows) != GROUP_SIZE_V5 or len(scene_ids) != 1 or len(places) != 4:
                raise SceneLocalPlaceScheduleError("constructed group is not scene-local B4")
            ordered_groups.append(
                SceneLocalPlaceGroupV5(
                    index=len(ordered_groups),
                    family=family,
                    scene_id=next(iter(scene_ids)),
                    rows=group_rows,
                )
            )

    scheduled_rows = [row for group in ordered_groups for row in group.rows]
    by_family_accounting: dict[str, dict[str, int]] = {}
    for family in FAMILIES_V5:
        source = by_family[family]
        presented = [row for row in scheduled_rows if row.family == family]
        unique_presented = {row.content_sha256 for row in presented}
        source_scenes = {row.scene_id for row in source}
        eligible_scenes = {
            scene_id
            for scene_id in source_scenes
            if len(
                {
                    row.place_identity_sha256
                    for row in source
                    if row.scene_id == scene_id
                }
            )
            >= GROUP_SIZE_V5
        }
        by_family_accounting[family] = {
            "source_row_count": len(source),
            "scheduled_group_count": GROUPS_PER_FAMILY_V5,
            "scheduled_row_presentation_count": len(presented),
            "unique_source_row_presented_count": len(unique_presented),
            "repeated_row_presentation_count": len(presented)
            - len(unique_presented),
            "dropped_source_row_count": len(source) - len(unique_presented),
            "source_scene_count": len(source_scenes),
            "eligible_source_scene_count": len(eligible_scenes),
            "scheduled_scene_count": len({row.scene_id for row in presented}),
        }

    group_binding = [
        {
            "index": group.index,
            "family": group.family,
            "scene_id": group.scene_id,
            "row_content_sha256": [row.content_sha256 for row in group.rows],
            "place_identity_sha256": [
                row.place_identity_sha256 for row in group.rows
            ],
        }
        for group in ordered_groups
    ]
    repeat_total = sum(
        value["repeated_row_presentation_count"]
        for value in by_family_accounting.values()
    )
    drop_total = sum(
        value["dropped_source_row_count"]
        for value in by_family_accounting.values()
    )
    receipt_core: dict[str, Any] = {
        "schema": SCHEMA_V5,
        "status": "PASS_METADATA_ONLY",
        "selection_seed": SELECTION_SEED_V5,
        "input": {
            "role": "train",
            "row_count": len(rows),
            "ordered_content_sha256": canonical_json_sha256(
                [row.content_sha256 for row in rows]
            ),
            "family_row_counts": dict(EXPECTED_SOURCE_ROWS_BY_FAMILY_V5),
        },
        "schedule": {
            "group_size": GROUP_SIZE_V5,
            "group_count": len(ordered_groups),
            "row_presentation_count": len(scheduled_rows),
            "groups_per_family": {
                family: GROUPS_PER_FAMILY_V5 for family in FAMILIES_V5
            },
            "row_presentations_per_family": {
                family: GROUPS_PER_FAMILY_V5 * GROUP_SIZE_V5
                for family in FAMILIES_V5
            },
            "ordered_group_binding_sha256": canonical_json_sha256(group_binding),
        },
        "accounting": {
            "by_family": by_family_accounting,
            "repeated_row_presentation_count": repeat_total,
            "dropped_source_row_count": drop_total,
        },
        "access_ledger": {
            "rgb_byte_open_count": 0,
            "rgb_decode_count": 0,
            "privileged_cell_yaw_emitted_count": 0,
            "heldout_or_sealed_open_count": 0,
        },
        "integrity": {
            "exact_800_b4_groups": len(ordered_groups) == TOTAL_GROUPS_V5,
            "exact_3200_row_presentations": len(scheduled_rows)
            == TOTAL_ROW_PRESENTATIONS_V5,
            "exact_100_groups_per_family": all(
                sum(group.family == family for group in ordered_groups)
                == GROUPS_PER_FAMILY_V5
                for family in FAMILIES_V5
            ),
            "every_group_one_scene": all(
                len({row.scene_id for row in group.rows}) == 1
                for group in ordered_groups
            ),
            "every_group_four_distinct_place_identities": all(
                len({row.place_identity_sha256 for row in group.rows})
                == GROUP_SIZE_V5
                for group in ordered_groups
            ),
        },
    }
    if not all(receipt_core["integrity"].values()):
        raise SceneLocalPlaceScheduleError("terminal schedule integrity failed")
    receipt = dict(receipt_core)
    receipt["content_sha256"] = canonical_json_sha256(receipt_core)
    return tuple(ordered_groups), receipt


__all__ = [
    "EXPECTED_SOURCE_ROWS_BY_FAMILY_V5",
    "FAMILIES_V5",
    "GROUPS_PER_FAMILY_V5",
    "GROUP_SIZE_V5",
    "SCHEMA_V5",
    "SELECTION_SEED_V5",
    "SceneLocalPlaceGroupV5",
    "SceneLocalPlaceScheduleError",
    "TOTAL_GROUPS_V5",
    "TOTAL_ROW_PRESENTATIONS_V5",
    "build_scene_local_place_schedule_v5",
]
