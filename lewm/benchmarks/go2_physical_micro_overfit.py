"""Pure contracts and metrics for the train-only Go2 micro-overfit probe.

This module performs no file I/O. The preparer owns the global row-index
metadata boundary; the runner receives only the derived train-role panel.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks.go2_physical_spatial_grounding import (
    canonical_json_sha256,
    empty_loss_accumulator,
    finalize_loss_accumulator,
    loss_accumulator_for_batch,
    merge_accumulator,
)


PANEL_SCHEMA = "lewm_go2_physical_micro_overfit_panel_v1"
RESULT_SCHEMA = "lewm_go2_physical_micro_overfit_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_physical_micro_overfit_smoke_result_v1"
ROW_SCHEMA = "lewm_go2_physical_micro_overfit_row_v1"
SELECTION_SEED = "go2_physical_microfit_patch7_v1"
SELECTION_UNIT = "one_transition_per_env_episode_reset_stream"
ROWS_PER_FAMILY_PANEL = 32
TRAIN_SCENES_PER_FAMILY = 9
FIT_SAME_POOL_SCENES = 4
SCENE_POOL_POLICY = "hash_rank_4_fit_same_5_cross_v1"
AUTHORITATIVE_EXECUTION = {
    "batch_size": 4,
    "faithful_steps": 2000,
    "ceiling_steps": 3000,
    "evaluation_interval": 100,
}
SMOKE_EXECUTION = {
    "batch_size": 4,
    "faithful_steps": 3,
    "ceiling_steps": 3,
    "evaluation_interval": 1,
}
PANELS = ("fit", "same_scene_holdout", "cross_scene_holdout")
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
TRAINING_WEIGHTS = {
    "unknown_known": (0.13853880763053894, 1.8614611625671387),
    "free_occupied": (0.21841949224472046, 1.7815805673599243),
}
DISTANCE_BINS_M = (
    ("0.0_to_0.5", 0.0, 0.5),
    ("0.5_to_1.0", 0.5, 1.0),
    ("1.0_to_2.0", 1.0, 2.0),
    ("2.0_to_3.0", 2.0, 3.0),
    ("3.0_plus", 3.0, None),
)
GATED_DISTANCE_BIN_NAMES = ("1.0_to_2.0", "2.0_to_3.0", "3.0_plus")

_ROW_FIELDS = (
    "scene_id",
    "family",
    "dataset_role",
    "global_row",
    "env_index",
    "episode_id",
    "reset_count",
    "current_episode_step",
    "next_episode_step",
    "current_frame_index",
    "next_frame_index",
    "current_timestamp_ns",
    "next_timestamp_ns",
    "primitive",
    "relative_se2_current_frame",
    "label_shard_path",
    "label_shard_sha256",
    "label_shard_row",
    "current_image_path",
    "current_image_sha256",
    "next_image_path",
    "next_image_sha256",
)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _hash_rank(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def _normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _ROW_FIELDS if name not in row]
    if missing:
        raise ValueError(f"micro-overfit row lacks fields: {missing}")
    if str(row["dataset_role"]) != "train":
        raise ValueError("micro-overfit panels may contain only train-role rows")
    for name in (
        "label_shard_sha256",
        "current_image_sha256",
        "next_image_sha256",
    ):
        if not _is_sha256(row[name]):
            raise ValueError(f"row has malformed {name}")
    delta = tuple(float(value) for value in row["relative_se2_current_frame"])
    if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
        raise ValueError("relative SE(2) delta must contain three finite values")
    return {
        "schema": ROW_SCHEMA,
        "scene_id": str(row["scene_id"]),
        "family": str(row["family"]),
        "dataset_role": "train",
        "global_row": int(row["global_row"]),
        "env_index": int(row["env_index"]),
        "episode_id": str(row["episode_id"]),
        "reset_count": int(row["reset_count"]),
        "current_episode_step": int(row["current_episode_step"]),
        "next_episode_step": int(row["next_episode_step"]),
        "current_frame_index": int(row["current_frame_index"]),
        "next_frame_index": int(row["next_frame_index"]),
        "current_timestamp_ns": int(row["current_timestamp_ns"]),
        "next_timestamp_ns": int(row["next_timestamp_ns"]),
        "primitive": str(row["primitive"]),
        "relative_se2_current_frame": list(delta),
        "label_shard_path": str(row["label_shard_path"]),
        "label_shard_sha256": str(row["label_shard_sha256"]),
        "label_shard_row": int(row["label_shard_row"]),
        "current_image_path": str(row["current_image_path"]),
        "current_image_sha256": str(row["current_image_sha256"]),
        "next_image_path": str(row["next_image_path"]),
        "next_image_sha256": str(row["next_image_sha256"]),
    }


def _row_image_hashes(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["current_image_sha256"]), str(row["next_image_sha256"]))


def _stream_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["env_index"]),
        str(row["episode_id"]),
        int(row["reset_count"]),
    )


def _rows_by_stream(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, int], list[Mapping[str, Any]]]:
    result: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault(_stream_key(row), []).append(row)
    return result


def _ranked_scene_streams(
    scene_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: str,
    family: str,
) -> tuple[
    list[str],
    dict[str, dict[tuple[int, str, int], list[Mapping[str, Any]]]],
    dict[str, list[tuple[int, str, int]]],
]:
    if len(scene_rows) != TRAIN_SCENES_PER_FAMILY:
        raise ValueError(
            f"family {family} must contain exactly {TRAIN_SCENES_PER_FAMILY} "
            "train scenes for the frozen pool partition"
        )
    ranked_scenes = sorted(
        scene_rows,
        key=lambda scene: (
            _hash_rank(seed, "pool-scene", family, scene),
            scene,
        ),
    )
    rows_by_scene_stream = {
        scene: _rows_by_stream(rows) for scene, rows in scene_rows.items()
    }
    ranked_streams = {
        scene: sorted(
            rows_by_scene_stream[scene],
            key=lambda stream: (
                _hash_rank(seed, "pool-stream", family, scene, *stream),
                stream,
            ),
        )
        for scene in ranked_scenes
    }
    return ranked_scenes, rows_by_scene_stream, ranked_streams


def _assigned_stream_pool(
    *,
    panel: str,
    fit_same_scenes: Sequence[str],
    cross_scenes: Sequence[str],
    ranked_streams: Mapping[str, Sequence[tuple[int, str, int]]],
) -> list[tuple[str, tuple[int, str, int]]]:
    if panel == "fit":
        return [
            (scene, stream)
            for scene in fit_same_scenes
            for index, stream in enumerate(ranked_streams[scene])
            if index % 2 == 0
        ]
    if panel == "same_scene_holdout":
        return [
            (scene, stream)
            for scene in fit_same_scenes
            for index, stream in enumerate(ranked_streams[scene])
            if index % 2 == 1
        ]
    if panel == "cross_scene_holdout":
        return [
            (scene, stream)
            for scene in cross_scenes
            for stream in ranked_streams[scene]
        ]
    raise ValueError(f"unsupported panel: {panel}")


def _select_pooled_stream_rows(
    rows_by_scene_stream: Mapping[
        str, Mapping[tuple[int, str, int], Sequence[Mapping[str, Any]]]
    ],
    assigned_streams: Sequence[tuple[str, tuple[int, str, int]]],
    *,
    required_count: int,
    family: str,
    panel: str,
    seed: str,
    required_scenes: Sequence[str],
    used_global_rows: set[int],
    used_image_hashes: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, tuple[int, str, int]]],
    dict[str, Any],
]:
    ranked = sorted(
        assigned_streams,
        key=lambda item: (
            _hash_rank(
                seed,
                "panel-prefix",
                panel,
                family,
                item[0],
                *item[1],
            ),
            item[0],
            item[1],
        ),
    )
    minimum_per_scene = 2
    if required_count < minimum_per_scene * len(required_scenes):
        raise ValueError("panel row budget cannot supply two rows per pool scene")
    selected: list[dict[str, Any]] = []
    selected_streams: list[tuple[str, tuple[int, str, int]]] = []
    selected_stream_set: set[tuple[str, tuple[int, str, int]]] = set()
    unusable_stream_set: set[tuple[str, tuple[int, str, int]]] = set()
    skipped: list[dict[str, Any]] = []

    def try_stream(scene: str, stream: tuple[int, str, int]) -> bool:
        if (scene, stream) in unusable_stream_set:
            return False
        candidates = sorted(
            rows_by_scene_stream[scene][stream],
            key=lambda row: (
                _hash_rank(
                    seed,
                    "row",
                    panel,
                    family,
                    scene,
                    *stream,
                    int(row["global_row"]),
                ),
                int(row["global_row"]),
            ),
        )
        chosen = None
        rejection_counts: Counter[str] = Counter()
        for raw_row in candidates:
            row = _normalized_row(raw_row)
            global_row = int(row["global_row"])
            current_hash, next_hash = _row_image_hashes(row)
            if global_row in used_global_rows:
                rejection_counts["global_row_already_used"] += 1
                continue
            if current_hash == next_hash:
                rejection_counts["current_equals_next_image_hash"] += 1
                continue
            if current_hash in used_image_hashes or next_hash in used_image_hashes:
                rejection_counts["endpoint_image_hash_already_used"] += 1
                continue
            chosen = row
            break
        if chosen is None:
            unusable_stream_set.add((scene, stream))
            skipped.append(
                {
                    "scene_id": scene,
                    "stream": list(stream),
                    "candidate_row_count": len(candidates),
                    "rejection_counts": dict(sorted(rejection_counts.items())),
                }
            )
            return False
        selected.append(chosen)
        selected_streams.append((scene, stream))
        selected_stream_set.add((scene, stream))
        used_global_rows.add(int(chosen["global_row"]))
        used_image_hashes.update(_row_image_hashes(chosen))
        return True

    # A same-scene wrong-view control requires two distinct transitions in
    # every scene represented by the frozen pool. Establish that support first,
    # then fill the remaining budget from the global hash-ranked prefix.
    for required_scene in required_scenes:
        chosen_for_scene = 0
        for scene, stream in ranked:
            if scene != required_scene:
                continue
            if try_stream(scene, stream):
                chosen_for_scene += 1
                if chosen_for_scene == minimum_per_scene:
                    break
        if chosen_for_scene != minimum_per_scene:
            raise ValueError(
                f"family {family}/{panel} cannot supply two metadata-valid "
                f"transitions for scene {required_scene}"
            )

    for scene, stream in ranked:
        if len(selected) == int(required_count):
            break
        if (scene, stream) in selected_stream_set:
            continue
        try_stream(scene, stream)
    if len(selected) != int(required_count):
        raise ValueError(
            f"family {family}/{panel} supplied {len(selected)}/{required_count} "
            f"selectable pooled episode streams after scanning {len(ranked)}"
        )
    scene_counts = Counter(scene for scene, _stream in selected_streams)
    return selected, selected_streams, {
        "assigned_stream_count": len(assigned_streams),
        "ranked_stream_count": len(ranked),
        "examined_stream_count": len(selected_streams) + len(skipped),
        "skipped_unusable_stream_count": len(skipped),
        "skipped_unusable_streams": skipped,
        "selected_stream_count": len(selected_streams),
        "minimum_selected_rows_per_pool_scene": minimum_per_scene,
        "selected_rows_by_scene": dict(sorted(scene_counts.items())),
        "selected_streams_sha256": canonical_json_sha256(
            [[scene, list(stream)] for scene, stream in selected_streams]
        ),
    }


def select_train_only_panels(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    seed: str = SELECTION_SEED,
    families: Sequence[str] = FAMILIES,
    rows_per_family_panel: int = ROWS_PER_FAMILY_PANEL,
) -> dict[str, Any]:
    """Select label-independent train-only fit and holdout panels.

    The global JSONL parser necessarily materializes complete row metadata,
    including non-train path strings. Non-train paths are never emitted into
    the panel, dereferenced, hashed, decoded, or opened.
    """

    requested_families = tuple(map(str, families))
    if str(seed) != SELECTION_SEED:
        raise ValueError("micro-overfit selection seed differs from the frozen contract")
    if requested_families != FAMILIES:
        raise ValueError("micro-overfit family sequence differs from the frozen contract")
    if int(rows_per_family_panel) != ROWS_PER_FAMILY_PANEL:
        raise ValueError("micro-overfit panel row count differs from the frozen contract")
    if len(set(requested_families)) != len(requested_families):
        raise ValueError("micro-overfit families must be unique")
    if rows_per_family_panel <= 0:
        raise ValueError("rows_per_family_panel must be positive")

    role_metadata_counts: Counter[str] = Counter()
    train_by_family_scene: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        family: {} for family in requested_families
    }
    train_primitives: set[str] = set()
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if scene_id not in assignments:
            raise ValueError(f"row scene lacks a role assignment: {scene_id!r}")
        role = str(assignments[scene_id])
        direct_role = str(row.get("dataset_role", ""))
        if direct_role != role:
            raise ValueError(f"row and scene role disagree for {scene_id}")
        role_metadata_counts[role] += 1
        if role != "train":
            continue
        train_primitives.add(str(row["primitive"]))
        family = str(row["family"])
        if family in train_by_family_scene:
            train_by_family_scene[family].setdefault(scene_id, []).append(row)

    if not train_primitives:
        raise ValueError("train-role primitive vocabulary is empty")

    pool_contract: dict[str, Any] = {}
    for family in requested_families:
        by_scene = train_by_family_scene[family]
        ranked_scenes, rows_by_scene_stream, ranked_streams = (
            _ranked_scene_streams(by_scene, seed=seed, family=family)
        )
        fit_same_scenes = ranked_scenes[:FIT_SAME_POOL_SCENES]
        cross_scenes = ranked_scenes[FIT_SAME_POOL_SCENES:]
        pool_contract[family] = {
            "scene_pool_policy": SCENE_POOL_POLICY,
            "ranked_train_scenes": ranked_scenes,
            "ranked_train_scenes_sha256": canonical_json_sha256(ranked_scenes),
            "fit_same_pool_scenes": fit_same_scenes,
            "cross_pool_scenes": cross_scenes,
            "fit_same_pool_scene_count": len(fit_same_scenes),
            "cross_pool_scene_count": len(cross_scenes),
            "fit_same_stream_assignment": "even_fit_odd_same_per_scene_hash_rank",
            "cross_stream_assignment": "all_streams_from_cross_pool",
            "scene_row_counts": {
                scene: len(by_scene[scene]) for scene in ranked_scenes
            },
            "scene_stream_counts": {
                scene: len(rows_by_scene_stream[scene]) for scene in ranked_scenes
            },
        }

    panels: dict[str, list[dict[str, Any]]] = {name: [] for name in PANELS}
    selection_reports: dict[str, dict[str, Any]] = {name: {} for name in PANELS}
    used_global_rows: set[int] = set()
    used_image_hashes: set[str] = set()
    for panel in PANELS:
        for family in requested_families:
            by_scene = train_by_family_scene[family]
            ranked_scenes, rows_by_scene_stream, ranked_streams = (
                _ranked_scene_streams(by_scene, seed=seed, family=family)
            )
            fit_same_scenes = ranked_scenes[:FIT_SAME_POOL_SCENES]
            cross_scenes = ranked_scenes[FIT_SAME_POOL_SCENES:]
            assigned = _assigned_stream_pool(
                panel=panel,
                fit_same_scenes=fit_same_scenes,
                cross_scenes=cross_scenes,
                ranked_streams=ranked_streams,
            )
            selected_rows, selected_streams, report = _select_pooled_stream_rows(
                rows_by_scene_stream,
                assigned,
                required_count=rows_per_family_panel,
                family=family,
                panel=panel,
                seed=seed,
                required_scenes=(
                    cross_scenes
                    if panel == "cross_scene_holdout"
                    else fit_same_scenes
                ),
                used_global_rows=used_global_rows,
                used_image_hashes=used_image_hashes,
            )
            panels[panel].extend(selected_rows)
            selection_reports[panel][family] = {
                **report,
                "selected_streams": [
                    [scene, list(stream)] for scene, stream in selected_streams
                ],
            }

    for panel_rows in panels.values():
        panel_rows.sort(key=lambda row: int(row["global_row"]))
    panel_records = {
        name: {
            "row_count": len(panel_rows),
            "frame_count": 2 * len(panel_rows),
            "rows_sha256": canonical_json_sha256(panel_rows),
            "rows": panel_rows,
        }
        for name, panel_rows in panels.items()
    }
    return {
        "selection_seed": str(seed),
        "families": list(requested_families),
        "rows_per_family_panel": int(rows_per_family_panel),
        "selection_unit": SELECTION_UNIT,
        "scene_pool_policy": SCENE_POOL_POLICY,
        "train_scenes_per_family": TRAIN_SCENES_PER_FAMILY,
        "fit_same_pool_scene_count": FIT_SAME_POOL_SCENES,
        "cross_pool_scene_count": TRAIN_SCENES_PER_FAMILY - FIT_SAME_POOL_SCENES,
        "pool_contract": pool_contract,
        "selection_reports": selection_reports,
        "primitive_vocabulary": sorted(train_primitives),
        "panels": panel_records,
        "metadata_access": {
            "global_row_index_metadata_read": True,
            "role_row_counts": dict(sorted(role_metadata_counts.items())),
            "full_row_objects_parsed_including_non_train_path_metadata": True,
            "non_train_artifact_paths_emitted_to_panel": False,
            "non_train_artifact_paths_dereferenced": False,
            "non_train_label_shard_byte_opens": 0,
            "non_train_image_byte_opens": 0,
            "non_train_model_outputs": 0,
        },
    }


def validate_panel_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != PANEL_SCHEMA:
        raise ValueError("unsupported physical micro-overfit panel schema")
    core = dict(payload)
    declared = str(core.pop("content_sha256", ""))
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("physical micro-overfit panel content hash mismatch")
    families = tuple(map(str, payload.get("families", ())))
    if families != FAMILIES:
        raise ValueError("physical micro-overfit family contract changed")
    if str(payload.get("selection_seed", "")) != SELECTION_SEED:
        raise ValueError("physical micro-overfit selection seed changed")
    if str(payload.get("selection_unit", "")) != SELECTION_UNIT:
        raise ValueError("physical micro-overfit selection unit changed")
    panel_payload = payload.get("panels")
    if not isinstance(panel_payload, Mapping) or set(panel_payload) != set(PANELS):
        raise ValueError("physical micro-overfit panels are incomplete")

    expected_per_family = int(payload.get("rows_per_family_panel", 0))
    if expected_per_family != ROWS_PER_FAMILY_PANEL:
        raise ValueError("physical micro-overfit rows-per-family contract changed")
    all_rows: dict[str, list[Mapping[str, Any]]] = {}
    global_rows: set[int] = set()
    image_hashes: set[str] = set()
    for panel in PANELS:
        record = panel_payload[panel]
        if not isinstance(record, Mapping) or not isinstance(record.get("rows"), list):
            raise ValueError(f"malformed panel: {panel}")
        rows = [_normalized_row(row) for row in record["rows"]]
        if int(record.get("row_count", -1)) != len(rows):
            raise ValueError(f"row count mismatch for panel {panel}")
        if int(record.get("frame_count", -1)) != 2 * len(rows):
            raise ValueError(f"frame count mismatch for panel {panel}")
        if str(record.get("rows_sha256", "")) != canonical_json_sha256(rows):
            raise ValueError(f"row content hash mismatch for panel {panel}")
        family_counts = Counter(str(row["family"]) for row in rows)
        if family_counts != Counter({family: expected_per_family for family in FAMILIES}):
            raise ValueError(f"family balance mismatch for panel {panel}")
        for row in rows:
            global_row = int(row["global_row"])
            if global_row in global_rows:
                raise ValueError("global rows overlap across micro-overfit panels")
            global_rows.add(global_row)
            for image_hash in _row_image_hashes(row):
                if image_hash in image_hashes:
                    raise ValueError("image hashes overlap across micro-overfit panels")
                image_hashes.add(image_hash)
        all_rows[panel] = rows

        streams = {
            (str(row["scene_id"]), *_stream_key(row))
            for row in rows
        }
        if len(streams) != len(rows):
            raise ValueError(f"panel reuses an episode stream: {panel}")

    if str(payload.get("scene_pool_policy", "")) != SCENE_POOL_POLICY:
        raise ValueError("physical micro-overfit scene-pool policy changed")
    if int(payload.get("train_scenes_per_family", -1)) != TRAIN_SCENES_PER_FAMILY:
        raise ValueError("physical micro-overfit train-scene count changed")
    if int(payload.get("fit_same_pool_scene_count", -1)) != FIT_SAME_POOL_SCENES:
        raise ValueError("physical micro-overfit fit/same pool size changed")
    if int(payload.get("cross_pool_scene_count", -1)) != (
        TRAIN_SCENES_PER_FAMILY - FIT_SAME_POOL_SCENES
    ):
        raise ValueError("physical micro-overfit cross pool size changed")
    pool_contract = payload.get("pool_contract")
    reports = payload.get("selection_reports")
    if not isinstance(pool_contract, Mapping) or set(pool_contract) != set(FAMILIES):
        raise ValueError("panel scene-pool contract is incomplete")
    if not isinstance(reports, Mapping) or set(reports) != set(PANELS):
        raise ValueError("panel selection reports are incomplete")

    selected_streams_global: set[tuple[str, int, str, int]] = set()
    for family in FAMILIES:
        pool = pool_contract[family]
        if not isinstance(pool, Mapping) or pool.get("scene_pool_policy") != SCENE_POOL_POLICY:
            raise ValueError(f"scene-pool contract changed for {family}")
        ranked_scenes = list(map(str, pool.get("ranked_train_scenes", ())))
        if len(ranked_scenes) != TRAIN_SCENES_PER_FAMILY or len(
            set(ranked_scenes)
        ) != TRAIN_SCENES_PER_FAMILY:
            raise ValueError(f"scene-pool ranking is malformed for {family}")
        expected_ranking = sorted(
            ranked_scenes,
            key=lambda scene: (
                _hash_rank(SELECTION_SEED, "pool-scene", family, scene),
                scene,
            ),
        )
        if ranked_scenes != expected_ranking or canonical_json_sha256(
            ranked_scenes
        ) != str(pool.get("ranked_train_scenes_sha256", "")):
            raise ValueError(f"scene-pool ranking hash mismatch for {family}")
        fit_same_pool = list(map(str, pool.get("fit_same_pool_scenes", ())))
        cross_pool = list(map(str, pool.get("cross_pool_scenes", ())))
        if fit_same_pool != ranked_scenes[:FIT_SAME_POOL_SCENES] or cross_pool != (
            ranked_scenes[FIT_SAME_POOL_SCENES:]
        ):
            raise ValueError(f"scene-pool partition mismatch for {family}")

        for panel in PANELS:
            panel_reports = reports[panel]
            if not isinstance(panel_reports, Mapping):
                raise ValueError(f"malformed selection report for {panel}")
            report = panel_reports.get(family)
            if not isinstance(report, Mapping):
                raise ValueError(f"missing selection report for {family}/{panel}")
            raw_streams = report.get("selected_streams")
            if not isinstance(raw_streams, list) or len(raw_streams) != (
                ROWS_PER_FAMILY_PANEL
            ):
                raise ValueError(f"selected stream count mismatch for {family}/{panel}")
            parsed_streams: list[tuple[str, int, str, int]] = []
            for raw in raw_streams:
                if not isinstance(raw, list) or len(raw) != 2:
                    raise ValueError(f"malformed selected stream for {family}/{panel}")
                scene = str(raw[0])
                stream = raw[1]
                if not isinstance(stream, list) or len(stream) != 3:
                    raise ValueError(f"malformed stream key for {family}/{panel}")
                parsed_streams.append(
                    (scene, int(stream[0]), str(stream[1]), int(stream[2]))
                )
            if len(set(parsed_streams)) != ROWS_PER_FAMILY_PANEL:
                raise ValueError(f"selected streams repeat for {family}/{panel}")
            if selected_streams_global & set(parsed_streams):
                raise ValueError("episode streams overlap across micro-overfit panels")
            selected_streams_global.update(parsed_streams)
            actual = {
                (str(row["scene_id"]), *_stream_key(row))
                for row in all_rows[panel]
                if str(row["family"]) == family
            }
            if set(parsed_streams) != actual:
                raise ValueError(f"selected stream rows disagree for {family}/{panel}")
            allowed_scenes = (
                set(cross_pool)
                if panel == "cross_scene_holdout"
                else set(fit_same_pool)
            )
            scene_counts = Counter(scene for scene, *_rest in parsed_streams)
            if set(scene_counts) != allowed_scenes or min(scene_counts.values()) < 2:
                raise ValueError(f"scene coverage/control support failed for {family}/{panel}")
            if report.get("selected_rows_by_scene") != dict(sorted(scene_counts.items())):
                raise ValueError(f"selected scene counts disagree for {family}/{panel}")
            expected_stream_hash = canonical_json_sha256(
                [[scene, [env, episode, reset]] for scene, env, episode, reset in parsed_streams]
            )
            if expected_stream_hash != str(report.get("selected_streams_sha256", "")):
                raise ValueError(f"selected stream hash mismatch for {family}/{panel}")
    return {panel: list(rows) for panel, rows in all_rows.items()}


def frame_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        normalized = _normalized_row(row)
        for side in ("current", "next"):
            records.append(
                {
                    "scene_id": normalized["scene_id"],
                    "family": normalized["family"],
                    "global_row": normalized["global_row"],
                    "side": side,
                    "image_path": normalized[f"{side}_image_path"],
                    "image_sha256": normalized[f"{side}_image_sha256"],
                    "label_shard_path": normalized["label_shard_path"],
                    "label_shard_sha256": normalized["label_shard_sha256"],
                    "label_shard_row": normalized["label_shard_row"],
                }
            )
    records.sort(key=lambda item: (int(item["global_row"]), str(item["side"])))
    return records


def attach_role_global_shuffle(
    records: Sequence[Mapping[str, Any]], *, seed: int, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a deterministic zero-match cross-scene image control."""

    copied = [dict(record) for record in records]
    if len(copied) < 2:
        raise ValueError("role-global shuffle requires at least two frames")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(copied):
        grouped.setdefault(str(record["scene_id"]), []).append(index)
    maximum = max(map(len, grouped.values()))
    if maximum * 2 > len(copied):
        raise ValueError("no zero-match cross-scene permutation exists")
    scene_order = sorted(
        grouped,
        key=lambda scene: (_hash_rank(seed, namespace, "scene", scene), scene),
    )
    ordered = []
    for scene in scene_order:
        ordered.extend(
            sorted(
                grouped[scene],
                key=lambda index: (
                    _hash_rank(seed, namespace, "frame", copied[index]["image_sha256"]),
                    int(copied[index]["global_row"]),
                    str(copied[index]["side"]),
                ),
            )
        )
    permutation = np.empty(len(copied), dtype=np.int64)
    for position, target in enumerate(ordered):
        permutation[target] = ordered[(position + maximum) % len(ordered)]

    same_image = same_scene = same_transition = 0
    for target_index, source_index_value in enumerate(permutation):
        source_index = int(source_index_value)
        target = copied[target_index]
        source = copied[source_index]
        target["control_image_path"] = str(source["image_path"])
        target["control_image_sha256"] = str(source["image_sha256"])
        target["control_scene_id"] = str(source["scene_id"])
        target["control_global_row"] = int(source["global_row"])
        target["control_side"] = str(source["side"])
        same_image += int(target["image_sha256"] == source["image_sha256"])
        same_scene += int(target["scene_id"] == source["scene_id"])
        same_transition += int(
            target["scene_id"] == source["scene_id"]
            and int(target["global_row"]) == int(source["global_row"])
        )
    if same_image or same_scene or same_transition:
        raise ValueError("role-global shuffle failed its zero-match contract")
    report = {
        "schema": "lewm_go2_micro_overfit_shuffle_v1",
        "seed": int(seed),
        "namespace": str(namespace),
        "record_count": len(copied),
        "permutation_sha256": canonical_json_sha256(permutation.tolist()),
        "same_image_pairs": same_image,
        "same_scene_pairs": same_scene,
        "same_transition_pairs": same_transition,
    }
    return copied, report


def attach_same_scene_wrong_view(
    records: Sequence[Mapping[str, Any]], *, seed: int, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a deterministic wrong-transition control within each scene."""

    copied = [dict(record) for record in records]
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(copied):
        grouped.setdefault(str(record["scene_id"]), []).append(index)
    permutation = np.empty(len(copied), dtype=np.int64)
    scene_reports = {}
    for scene_id, indices in sorted(grouped.items()):
        transition_counts = Counter(int(copied[index]["global_row"]) for index in indices)
        maximum = max(transition_counts.values())
        if maximum * 2 > len(indices):
            raise ValueError(f"scene {scene_id} cannot support a wrong-view control")
        ordered = sorted(
            indices,
            key=lambda index: (
                int(copied[index]["global_row"]),
                _hash_rank(
                    seed,
                    namespace,
                    scene_id,
                    copied[index]["side"],
                    copied[index]["image_sha256"],
                ),
            ),
        )
        for position, target in enumerate(ordered):
            permutation[target] = ordered[(position + maximum) % len(ordered)]
        scene_reports[scene_id] = {
            "frame_count": len(indices),
            "transition_count": len(transition_counts),
            "rotation": maximum,
        }

    same_image = same_transition = different_scene = 0
    for target_index, source_index_value in enumerate(permutation):
        source_index = int(source_index_value)
        target = copied[target_index]
        source = copied[source_index]
        target["same_scene_control_image_path"] = str(source["image_path"])
        target["same_scene_control_image_sha256"] = str(source["image_sha256"])
        target["same_scene_control_global_row"] = int(source["global_row"])
        target["same_scene_control_side"] = str(source["side"])
        same_image += int(target["image_sha256"] == source["image_sha256"])
        same_transition += int(
            int(target["global_row"]) == int(source["global_row"])
        )
        different_scene += int(target["scene_id"] != source["scene_id"])
    if same_image or same_transition or different_scene:
        raise ValueError("same-scene wrong-view control violated its pairing contract")
    return copied, {
        "schema": "lewm_go2_micro_overfit_same_scene_wrong_view_v1",
        "seed": int(seed),
        "namespace": str(namespace),
        "record_count": len(copied),
        "permutation_sha256": canonical_json_sha256(permutation.tolist()),
        "same_image_pairs": same_image,
        "same_transition_pairs": same_transition,
        "different_scene_pairs": different_scene,
        "scenes": scene_reports,
    }


def empty_raw_accumulator() -> dict[str, Any]:
    return {
        "loss": empty_loss_accumulator(),
        "joint_confusion": np.zeros((3, 3), dtype=np.int64),
        "unknown_known_confusion": np.zeros((2, 2), dtype=np.int64),
        "free_occupied_confusion": np.zeros((2, 2), dtype=np.int64),
        "distance_free_true": {name: 0 for name, _low, _high in DISTANCE_BINS_M},
        "distance_free_correct": {name: 0 for name, _low, _high in DISTANCE_BINS_M},
        "free_scores": [],
        "free_targets": [],
        "occupied_scores": [],
        "occupied_targets": [],
        "posterior_by_truth": {
            truth: {predicted: [] for predicted in range(3)} for truth in range(3)
        },
    }


def update_raw_accumulator(
    accumulator: dict[str, Any],
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    distances_m: np.ndarray,
    *,
    unknown_known_weights: Sequence[float] = TRAINING_WEIGHTS["unknown_known"],
    free_occupied_weights: Sequence[float] = TRAINING_WEIGHTS["free_occupied"],
) -> None:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if distances_m.shape != truth.shape[-2:]:
        raise ValueError("distance grid does not match occupancy grid")
    merge_accumulator(
        accumulator["loss"],
        loss_accumulator_for_batch(
            values,
            truth,
            valid,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
        ),
    )
    prediction = values.argmax(axis=1)
    known_logit = np.logaddexp(values[:, 1], values[:, 2])
    unknown_known_prediction = (known_logit > values[:, 0]).astype(np.int64)
    unknown_known_truth = (truth != 0).astype(np.int64)
    free_occupied_prediction = values[:, 1:].argmax(axis=1)
    free_occupied_truth = np.clip(truth - 1, 0, 1)
    known = valid & (truth != 0)
    maximum = values.max(axis=1, keepdims=True)
    probabilities = np.exp(values - maximum)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    accumulator["free_scores"].append(probabilities[:, 1][valid])
    accumulator["free_targets"].append((truth[valid] == 1).astype(np.int8))
    accumulator["occupied_scores"].append(probabilities[:, 2][valid])
    accumulator["occupied_targets"].append((truth[valid] == 2).astype(np.int8))
    for actual in range(3):
        actual_mask = valid & (truth == actual)
        for predicted in range(3):
            accumulator["posterior_by_truth"][actual][predicted].append(
                probabilities[:, predicted][actual_mask]
            )
    for actual in range(3):
        for predicted in range(3):
            accumulator["joint_confusion"][actual, predicted] += int(
                (valid & (truth == actual) & (prediction == predicted)).sum()
            )
    for actual in range(2):
        for predicted in range(2):
            accumulator["unknown_known_confusion"][actual, predicted] += int(
                (
                    valid
                    & (unknown_known_truth == actual)
                    & (unknown_known_prediction == predicted)
                ).sum()
            )
            accumulator["free_occupied_confusion"][actual, predicted] += int(
                (
                    known
                    & (free_occupied_truth == actual)
                    & (free_occupied_prediction == predicted)
                ).sum()
            )
    distance_grid = np.broadcast_to(distances_m, truth.shape)
    for name, lower, upper in DISTANCE_BINS_M:
        in_bin = distance_grid >= lower
        if upper is not None:
            in_bin &= distance_grid < upper
        free = valid & in_bin & (truth == 1)
        accumulator["distance_free_true"][name] += int(free.sum())
        accumulator["distance_free_correct"][name] += int(
            (free & (prediction == 1)).sum()
        )


def _balanced_accuracy(confusion: np.ndarray) -> float | None:
    recalls = []
    for class_index in range(confusion.shape[0]):
        support = int(confusion[class_index].sum())
        if support:
            recalls.append(float(confusion[class_index, class_index]) / support)
    return None if not recalls else float(np.mean(recalls))


def _concatenate(chunks: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(chunk) for chunk in chunks if np.asarray(chunk).size]
    return np.concatenate(nonempty) if nonempty else np.asarray([], dtype=np.float64)


def _average_precision(targets: np.ndarray, scores: np.ndarray) -> float | None:
    truth = np.asarray(targets, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or truth.ndim != 1:
        raise ValueError("average-precision inputs must be matching vectors")
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    ranked = truth[order]
    true_positives = np.cumsum(ranked == 1)
    ranks = np.arange(1, len(ranked) + 1, dtype=np.float64)
    return float((true_positives[ranked == 1] / ranks[ranked == 1]).sum() / positives)


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return None
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def finalize_raw_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    result = finalize_loss_accumulator(accumulator["loss"])
    joint = np.asarray(accumulator["joint_confusion"], dtype=np.int64)
    uk = np.asarray(accumulator["unknown_known_confusion"], dtype=np.int64)
    fo = np.asarray(accumulator["free_occupied_confusion"], dtype=np.int64)
    class_recall = {}
    class_precision = {}
    for index, name in enumerate(("unknown", "free", "occupied")):
        support = int(joint[index].sum())
        class_recall[name] = None if support == 0 else float(joint[index, index]) / support
        predicted = int(joint[:, index].sum())
        class_precision[name] = (
            None if predicted == 0 else float(joint[index, index]) / predicted
        )
    free_scores = _concatenate(accumulator["free_scores"])
    free_targets = _concatenate(accumulator["free_targets"]).astype(np.int8)
    occupied_scores = _concatenate(accumulator["occupied_scores"])
    occupied_targets = _concatenate(accumulator["occupied_targets"]).astype(np.int8)
    class_names = ("unknown", "free", "occupied")
    posterior_quantiles = {
        class_names[truth]: {
            class_names[predicted]: _quantiles(
                _concatenate(accumulator["posterior_by_truth"][truth][predicted])
            )
            for predicted in range(3)
        }
        for truth in range(3)
    }
    result.update(
        {
            "joint_confusion": joint.tolist(),
            "unknown_known_confusion": uk.tolist(),
            "free_occupied_confusion": fo.tolist(),
            "unknown_known_balanced_accuracy": _balanced_accuracy(uk),
            "free_occupied_balanced_accuracy": _balanced_accuracy(fo),
            "class_recall": class_recall,
            "class_precision": class_precision,
            "free_average_precision": _average_precision(free_targets, free_scores),
            "occupied_average_precision": _average_precision(
                occupied_targets, occupied_scores
            ),
            "posterior_quantiles_by_truth_class": posterior_quantiles,
            "distance_free_recall": {
                name: (
                    None
                    if int(accumulator["distance_free_true"][name]) == 0
                    else float(accumulator["distance_free_correct"][name])
                    / int(accumulator["distance_free_true"][name])
                )
                for name, _lower, _upper in DISTANCE_BINS_M
            },
            "distance_free_support": dict(accumulator["distance_free_true"]),
        }
    )
    return result


def fit_gate(
    metrics: Mapping[str, Any],
    *,
    cross_scene_shuffled_nll: float,
    same_scene_shuffled_nll: float,
) -> dict[str, Any]:
    checks = {
        "raw_hierarchical_balanced_nll_le_0_03": (
            metrics.get("raw_hierarchical_balanced_nll") is not None
            and float(metrics["raw_hierarchical_balanced_nll"]) <= 0.03
        ),
        "unknown_known_balanced_accuracy_ge_0_99": (
            metrics.get("unknown_known_balanced_accuracy") is not None
            and float(metrics["unknown_known_balanced_accuracy"]) >= 0.99
        ),
        "free_occupied_balanced_accuracy_ge_0_99": (
            metrics.get("free_occupied_balanced_accuracy") is not None
            and float(metrics["free_occupied_balanced_accuracy"]) >= 0.99
        ),
    }
    recalls = metrics.get("class_recall", {})
    for name in ("unknown", "free", "occupied"):
        value = recalls.get(name) if isinstance(recalls, Mapping) else None
        checks[f"{name}_recall_ge_0_98"] = value is not None and float(value) >= 0.98
    distance_recalls = metrics.get("distance_free_recall", {})
    for name in GATED_DISTANCE_BIN_NAMES:
        value = distance_recalls.get(name) if isinstance(distance_recalls, Mapping) else None
        checks[f"{name}_free_recall_ge_0_95"] = value is not None and float(value) >= 0.95
    correct_nll = metrics.get("raw_hierarchical_balanced_nll")
    cross_delta = (
        None
        if correct_nll is None
        else float(cross_scene_shuffled_nll) - float(correct_nll)
    )
    same_delta = (
        None
        if correct_nll is None
        else float(same_scene_shuffled_nll) - float(correct_nll)
    )
    checks["cross_scene_shuffled_minus_correct_nll_ge_0_25"] = (
        cross_delta is not None and cross_delta >= 0.25
    )
    checks["same_scene_wrong_view_minus_correct_nll_ge_0_25"] = (
        same_delta is not None and same_delta >= 0.25
    )
    return {
        "schema": "lewm_go2_physical_micro_overfit_fit_gate_v1",
        "passes": all(checks.values()),
        "checks": checks,
        "cross_scene_shuffled_minus_correct_raw_hierarchical_balanced_nll": (
            cross_delta
        ),
        "same_scene_wrong_view_minus_correct_raw_hierarchical_balanced_nll": (
            same_delta
        ),
    }


_ARMS = ("patch14_8x8", "patch7_16x16")
_CLASS_NAMES = ("unknown", "free", "occupied")


def _stage_fit_pass(stage: Mapping[str, Any], arm: str) -> bool:
    arm_result = stage.get(arm)
    return isinstance(arm_result, Mapping) and bool(
        arm_result.get("fit_gate_passed_terminal_three_evaluations", False)
    )


def _family_correct_metrics(
    stage: Mapping[str, Any], arm: str, panel: str
) -> dict[str, Mapping[str, Any]]:
    try:
        families = stage[arm]["final_panels"][panel]["families"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"stage lacks {arm}/{panel} family metrics") from exc
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError(f"{arm}/{panel} must contain exactly the five families")
    result = {}
    for family in FAMILIES:
        try:
            metrics = families[family]["conditions"]["correct_rgb"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"stage lacks {arm}/{panel}/{family} correct-RGB metrics"
            ) from exc
        if not isinstance(metrics, Mapping):
            raise ValueError(f"malformed {arm}/{panel}/{family} metrics")
        result[family] = metrics
    return result


def _required_metric(metrics: Mapping[str, Any], *path: str) -> float:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"metrics lack {'/'.join(path)}")
        value = value[key]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"metrics contain invalid {'/'.join(path)}")
    return float(value)


def _holdout_patch7_checks(
    stage: Mapping[str, Any], panel: str
) -> dict[str, Any]:
    if panel not in ("same_scene_holdout", "cross_scene_holdout"):
        raise ValueError(f"unsupported holdout panel: {panel}")
    baseline = _family_correct_metrics(stage, "patch14_8x8", panel)
    patch7 = _family_correct_metrics(stage, "patch7_16x16", panel)
    per_family = {}
    for family in FAMILIES:
        baseline_metrics = baseline[family]
        patch7_metrics = patch7[family]
        baseline_nll = _required_metric(
            baseline_metrics, "raw_hierarchical_balanced_nll"
        )
        patch7_nll = _required_metric(
            patch7_metrics, "raw_hierarchical_balanced_nll"
        )
        baseline_far = _required_metric(
            baseline_metrics, "distance_free_recall", "3.0_plus"
        )
        patch7_far = _required_metric(
            patch7_metrics, "distance_free_recall", "3.0_plus"
        )
        class_deltas = {
            name: _required_metric(patch7_metrics, "class_recall", name)
            - _required_metric(baseline_metrics, "class_recall", name)
            for name in _CLASS_NAMES
        }
        per_family[family] = {
            "patch14_hierarchical_nll": baseline_nll,
            "patch7_hierarchical_nll": patch7_nll,
            "patch14_far_free_recall": baseline_far,
            "patch7_far_free_recall": patch7_far,
            "patch7_minus_patch14_class_recall": class_deltas,
            "strictly_lower_nll_and_higher_far_free": (
                patch7_nll < baseline_nll and patch7_far > baseline_far
            ),
        }

    macro = {
        "patch14_hierarchical_nll": float(
            np.mean([per_family[family]["patch14_hierarchical_nll"] for family in FAMILIES])
        ),
        "patch7_hierarchical_nll": float(
            np.mean([per_family[family]["patch7_hierarchical_nll"] for family in FAMILIES])
        ),
        "patch14_far_free_recall": float(
            np.mean([per_family[family]["patch14_far_free_recall"] for family in FAMILIES])
        ),
        "patch7_far_free_recall": float(
            np.mean([per_family[family]["patch7_far_free_recall"] for family in FAMILIES])
        ),
        "patch7_minus_patch14_class_recall": {
            name: float(
                np.mean(
                    [
                        per_family[family]["patch7_minus_patch14_class_recall"][name]
                        for family in FAMILIES
                    ]
                )
            )
            for name in _CLASS_NAMES
        },
    }
    baseline_macro_nll = float(macro["patch14_hierarchical_nll"])
    ratio = (
        None
        if baseline_macro_nll <= 0.0
        else float(macro["patch7_hierarchical_nll"]) / baseline_macro_nll
    )
    far_delta = float(macro["patch7_far_free_recall"]) - float(
        macro["patch14_far_free_recall"]
    )
    favorable_count = sum(
        bool(record["strictly_lower_nll_and_higher_far_free"])
        for record in per_family.values()
    )
    required_favorable = 5 if panel == "cross_scene_holdout" else 4
    checks = {
        "equal_weight_family_macro_nll_ratio_le_0_80": (
            ratio is not None and ratio <= 0.80
        ),
        "equal_weight_family_macro_far_free_delta_ge_0_10": far_delta >= 0.10,
        "every_macro_class_recall_delta_ge_neg_0_01": min(
            macro["patch7_minus_patch14_class_recall"].values()
        )
        >= -0.01,
        "no_family_class_recall_delta_lt_neg_0_01": min(
            delta
            for record in per_family.values()
            for delta in record["patch7_minus_patch14_class_recall"].values()
        )
        >= -0.01,
        f"strict_family_nll_and_far_improvement_ge_{required_favorable}_of_5": (
            favorable_count >= required_favorable
        ),
    }
    return {
        "panel": panel,
        "passes": all(checks.values()),
        "checks": checks,
        "family_macro_weighting": "equal_weight_across_five_families",
        "macro": macro,
        "patch7_to_patch14_macro_hierarchical_nll_ratio": ratio,
        "patch7_minus_patch14_macro_far_free_recall": far_delta,
        "strictly_favorable_family_count": favorable_count,
        "strictly_favorable_family_requirement": required_favorable,
        "cross_scene_observed_one_sided_exact_sign_p": (
            1.0 / 32.0
            if panel == "cross_scene_holdout" and favorable_count == 5
            else None
        ),
        "registered_all_five_one_sided_exact_sign_p": (
            1.0 / 32.0 if panel == "cross_scene_holdout" else None
        ),
        "ties_count_as_failure": True,
        "per_family": per_family,
    }


def _near_fit_gate(stage: Mapping[str, Any], arm: str) -> bool:
    arm_result = stage.get(arm)
    if not isinstance(arm_result, Mapping):
        return False
    fit = arm_result.get("final_panels", {}).get("fit", {})
    conditions = fit.get("conditions", {}) if isinstance(fit, Mapping) else {}
    if not isinstance(conditions, Mapping):
        return False
    correct = conditions.get("correct_rgb")
    cross = conditions.get("role_global_shuffled_rgb")
    same = conditions.get("same_scene_wrong_view_rgb")
    if not all(isinstance(value, Mapping) for value in (correct, cross, same)):
        return False
    try:
        nll = _required_metric(correct, "raw_hierarchical_balanced_nll")
        class_recall = {
            name: _required_metric(correct, "class_recall", name)
            for name in _CLASS_NAMES
        }
        distance_recall = {
            name: _required_metric(correct, "distance_free_recall", name)
            for name in GATED_DISTANCE_BIN_NAMES
        }
        return all(
            (
                nll <= 0.033,
                _required_metric(correct, "unknown_known_balanced_accuracy") >= 0.989,
                _required_metric(correct, "free_occupied_balanced_accuracy") >= 0.989,
                min(class_recall.values()) >= 0.978,
                min(distance_recall.values()) >= 0.945,
                _required_metric(cross, "raw_hierarchical_balanced_nll") - nll
                >= 0.225,
                _required_metric(same, "raw_hierarchical_balanced_nll") - nll
                >= 0.225,
            )
        )
    except ValueError:
        return False


def _near_holdout_support(checks: Mapping[str, Mapping[str, Any]]) -> bool:
    relaxed_panels = []
    for panel in ("same_scene_holdout", "cross_scene_holdout"):
        record = checks[panel]
        ratio = record["patch7_to_patch14_macro_hierarchical_nll_ratio"]
        far_delta = float(record["patch7_minus_patch14_macro_far_free_recall"])
        class_deltas = record["macro"]["patch7_minus_patch14_class_recall"]
        relaxed_panels.append(
            ratio is not None
            and float(ratio) <= 0.85
            and far_delta >= 0.075
            and min(float(value) for value in class_deltas.values()) >= -0.015
        )
    return all(relaxed_panels) or int(
        checks["cross_scene_holdout"]["strictly_favorable_family_count"]
    ) >= 4


def classify_cross_arm_decision(
    faithful: Mapping[str, Any],
    ceiling: Mapping[str, Any] | None,
    *,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Adjudicate one seed without ever issuing a full-training license."""

    if int(seed) not in (20260710, 20260711):
        raise ValueError("authoritative micro-overfit seed must be 20260710 or 20260711")
    faithful_passes = {arm: _stage_fit_pass(faithful, arm) for arm in _ARMS}
    if not all(faithful_passes.values()) and ceiling is None:
        raise ValueError("ceiling optimizer is mandatory when either faithful arm fails")
    ceiling_passes = (
        None
        if ceiling is None
        else {arm: _stage_fit_pass(ceiling, arm) for arm in _ARMS}
    )
    expressive = {
        arm: faithful_passes[arm]
        or bool(ceiling_passes is not None and ceiling_passes[arm])
        for arm in _ARMS
    }
    common_stage = None
    common_stage_payload = None
    if all(faithful_passes.values()):
        common_stage = "production_faithful"
        common_stage_payload = faithful
    elif ceiling_passes is not None and all(ceiling_passes.values()):
        common_stage = "ceiling_optimizer"
        common_stage_payload = ceiling

    holdout_checks = None
    holdout_support = False
    if common_stage_payload is not None:
        holdout_checks = {
            panel: _holdout_patch7_checks(common_stage_payload, panel)
            for panel in ("same_scene_holdout", "cross_scene_holdout")
        }
        holdout_support = all(record["passes"] for record in holdout_checks.values())

    causal_support = not expressive["patch14_8x8"] and expressive["patch7_16x16"]
    provisional_support = causal_support or (
        all(expressive.values()) and holdout_support
    )
    provisional_basis = None
    qualifying_stage = None
    if causal_support:
        classification = "patch7_tokenization_bundle_causal_support"
        provisional_basis = "causal_fit"
        qualifying_stage = (
            "production_faithful"
            if faithful_passes["patch7_16x16"]
            else "ceiling_optimizer"
        )
    elif expressive["patch14_8x8"] and not expressive["patch7_16x16"]:
        classification = "patch14_expressive_patch7_tokenization_bundle_negative"
    elif not any(expressive.values()):
        classification = "both_arms_fail_patch7_tokenization_bundle_insufficient"
    elif all(expressive.values()) and common_stage is None:
        classification = "both_expressive_no_common_stage_comparison"
    elif all(expressive.values()) and holdout_support:
        classification = "patch7_tokenization_bundle_holdout_support"
        provisional_basis = "matched_holdout"
        qualifying_stage = common_stage
    else:
        classification = "both_expressive_no_patch7_tokenization_bundle_support"

    stage_disagreement = ceiling_passes is not None and any(
        faithful_passes[arm] != ceiling_passes[arm] for arm in _ARMS
    )
    near_gate = []
    for stage_name, stage_payload, passes in (
        ("production_faithful", faithful, faithful_passes),
        ("ceiling_optimizer", ceiling, ceiling_passes),
    ):
        if stage_payload is None or passes is None:
            continue
        for arm in _ARMS:
            if not passes[arm] and _near_fit_gate(stage_payload, arm):
                near_gate.append({"stage": stage_name, "arm": arm})
    near_holdout = bool(
        holdout_checks is not None and _near_holdout_support(holdout_checks)
    )
    replication_reasons = []
    if provisional_support:
        replication_reasons.append("provisional_support_requires_seed_20260711")
    if stage_disagreement:
        replication_reasons.append("faithful_and_ceiling_fit_gates_disagree")
    if near_gate:
        replication_reasons.append("arm_within_registered_near_fit_margin")
    if near_holdout and not holdout_support:
        replication_reasons.append("comparison_within_registered_near_holdout_margin")
    support_mechanism = (
        None
        if provisional_basis is None
        else f"{provisional_basis}:{qualifying_stage}"
    )
    return {
        "schema": "lewm_go2_physical_micro_overfit_cross_arm_decision_v2",
        "seed": int(seed),
        "classification": classification,
        "causal_claim": "patch7_16x16_tokenization_and_patch_embedding_bundle",
        "provisional_patch7_support": provisional_support,
        "provisional_support_basis": provisional_basis,
        "qualifying_stage": qualifying_stage,
        "support_mechanism": support_mechanism,
        "patch7_full_train_candidate_licensed": False,
        "single_seed_artifact_cannot_license": True,
        "fit_gate_passes": {
            "production_faithful": faithful_passes,
            "ceiling_optimizer": ceiling_passes,
        },
        "per_arm_expressive_faithful_or_ceiling": expressive,
        "matched_holdout_stage": common_stage,
        "holdout_patch7_checks": holdout_checks,
        "near_fit_gate_arms": near_gate,
        "near_holdout_margin": near_holdout,
        "second_seed_needed": int(seed) == 20260710 and bool(replication_reasons),
        "second_seed": (
            20260711
            if int(seed) == 20260710 and replication_reasons
            else None
        ),
        "second_seed_reasons": replication_reasons,
    }


def aggregate_two_seed_decisions(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the only rule permitted to license the patch7 bundle."""

    decisions = [first, second]
    if any(
        decision.get("schema")
        != "lewm_go2_physical_micro_overfit_cross_arm_decision_v2"
        for decision in decisions
    ):
        raise ValueError("two-seed aggregation requires v2 per-seed decisions")
    by_seed = {int(decision.get("seed", -1)): decision for decision in decisions}
    if set(by_seed) != {20260710, 20260711} or len(by_seed) != 2:
        raise ValueError("two-seed aggregation requires seeds 20260710 and 20260711")
    primary = by_seed[20260710]
    replication = by_seed[20260711]

    def branch_evidence_is_valid(decision: Mapping[str, Any]) -> bool:
        basis = decision.get("provisional_support_basis")
        stage = decision.get("qualifying_stage")
        passes = decision.get("fit_gate_passes")
        if not isinstance(passes, Mapping) or stage not in (
            "production_faithful",
            "ceiling_optimizer",
        ):
            return False
        faithful_passes = passes.get("production_faithful")
        ceiling_passes = passes.get("ceiling_optimizer")
        if not isinstance(faithful_passes, Mapping):
            return False
        if basis == "causal_fit":
            if not isinstance(ceiling_passes, Mapping):
                return False
            return (
                decision.get("classification")
                == "patch7_tokenization_bundle_causal_support"
                and decision.get("support_mechanism") == f"causal_fit:{stage}"
                and not bool(faithful_passes.get("patch14_8x8", False))
                and not bool(ceiling_passes.get("patch14_8x8", False))
                and bool(passes[stage].get("patch7_16x16", False))
            )
        if basis == "matched_holdout":
            holdout = decision.get("holdout_patch7_checks")
            return (
                decision.get("classification")
                == "patch7_tokenization_bundle_holdout_support"
                and decision.get("support_mechanism") == f"matched_holdout:{stage}"
                and decision.get("matched_holdout_stage") == stage
                and isinstance(passes.get(stage), Mapping)
                and all(bool(passes[stage].get(arm, False)) for arm in _ARMS)
                and isinstance(holdout, Mapping)
                and set(holdout)
                == {"same_scene_holdout", "cross_scene_holdout"}
                and all(
                    isinstance(record, Mapping) and bool(record.get("passes", False))
                    for record in holdout.values()
                )
            )
        return False

    checks = {
        "both_seeds_provisionally_support_patch7": all(
            bool(decision.get("provisional_patch7_support", False))
            for decision in (primary, replication)
        ),
        "same_provisional_support_basis": (
            primary.get("provisional_support_basis") is not None
            and primary.get("provisional_support_basis")
            == replication.get("provisional_support_basis")
        ),
        "same_favorable_classification": (
            primary.get("classification") == replication.get("classification")
        ),
        "same_qualifying_optimizer_stage": (
            primary.get("qualifying_stage") is not None
            and primary.get("qualifying_stage")
            == replication.get("qualifying_stage")
        ),
        "same_support_mechanism": (
            primary.get("support_mechanism") is not None
            and primary.get("support_mechanism")
            == replication.get("support_mechanism")
        ),
        "per_seed_artifacts_did_not_self_license": all(
            not bool(decision.get("patch7_full_train_candidate_licensed", False))
            for decision in (primary, replication)
        ),
        "both_seed_branch_evidence_valid": all(
            branch_evidence_is_valid(decision)
            for decision in (primary, replication)
        ),
    }
    licensed = all(checks.values())
    return {
        "schema": "lewm_go2_physical_micro_overfit_two_seed_decision_v1",
        "seeds": [20260710, 20260711],
        "classification": (
            "patch7_tokenization_bundle_two_seed_license"
            if licensed
            else "two_seed_inconclusive"
        ),
        "checks": checks,
        "support_basis": (
            primary.get("provisional_support_basis") if licensed else None
        ),
        "qualifying_stage": primary.get("qualifying_stage") if licensed else None,
        "patch7_full_train_candidate_licensed": licensed,
    }


def aggregate_two_seed_result_artifacts(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate two immutable result payloads before applying the seed rule."""

    artifacts = [first, second]
    normalized = []
    for artifact in artifacts:
        if artifact.get("schema") != RESULT_SCHEMA:
            raise ValueError("two-seed finalizer requires micro-overfit result v1")
        core = dict(artifact)
        declared = str(core.pop("content_sha256", ""))
        if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
            raise ValueError("micro-overfit result content hash mismatch")
        decision = artifact.get("cross_arm_decision")
        if not isinstance(decision, Mapping):
            raise ValueError("micro-overfit result lacks its per-seed decision")
        execution = artifact.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("micro-overfit result lacks execution provenance")
        determinism = execution.get("determinism")
        if not isinstance(determinism, Mapping) or int(
            determinism.get("seed", -1)
        ) != int(decision.get("seed", -2)):
            raise ValueError("result execution seed and decision seed disagree")
        normalized.append(
            {
                "content_sha256": declared,
                "decision": decision,
                "panel_manifest": artifact.get("inputs", {}).get(
                    "panel_manifest"
                ),
                "contract": artifact.get("contract"),
                "source_hashes": artifact.get("source_hashes"),
            }
        )
    for field in ("panel_manifest", "contract", "source_hashes"):
        if normalized[0][field] != normalized[1][field]:
            raise ValueError(f"two-seed result artifacts disagree on {field}")
    aggregate = aggregate_two_seed_decisions(
        normalized[0]["decision"], normalized[1]["decision"]
    )
    return {
        "schema": "lewm_go2_physical_micro_overfit_two_result_finalization_v1",
        "input_result_content_sha256": sorted(
            record["content_sha256"] for record in normalized
        ),
        "common_panel_manifest": normalized[0]["panel_manifest"],
        "common_source_hashes": normalized[0]["source_hashes"],
        "decision": aggregate,
        "patch7_full_train_candidate_licensed": aggregate[
            "patch7_full_train_candidate_licensed"
        ],
    }


__all__ = [
    "AUTHORITATIVE_EXECUTION",
    "DISTANCE_BINS_M",
    "FAMILIES",
    "GATED_DISTANCE_BIN_NAMES",
    "PANELS",
    "PANEL_SCHEMA",
    "RESULT_SCHEMA",
    "ROWS_PER_FAMILY_PANEL",
    "ROW_SCHEMA",
    "SCENE_POOL_POLICY",
    "SELECTION_SEED",
    "SELECTION_UNIT",
    "SMOKE_EXECUTION",
    "SMOKE_RESULT_SCHEMA",
    "TRAINING_WEIGHTS",
    "aggregate_two_seed_decisions",
    "aggregate_two_seed_result_artifacts",
    "attach_role_global_shuffle",
    "attach_same_scene_wrong_view",
    "canonical_json_sha256",
    "classify_cross_arm_decision",
    "empty_raw_accumulator",
    "finalize_raw_accumulator",
    "fit_gate",
    "frame_records",
    "select_train_only_panels",
    "update_raw_accumulator",
    "validate_panel_manifest",
]
