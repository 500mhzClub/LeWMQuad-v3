"""Causal endpoint adapter for the frozen recurrent-H4 V1 RGB schedule.

V1 already supplies deterministic, reset-safe six-command grouping, ranking,
scene selection, and custody checks.  V2 preserves those logical candidates
and adapts only their endpoints.  An action now joins the real final frame
immediately before its request to the fifth/final frame produced by that
requested block.  The first V1-phase sextet in each episode is discarded as a
unit because its first command has no recorded pre-request boundary.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (
    BLOCK_SIZE,
    COMMAND_DT_S,
    ENVS_PER_SOURCE,
    EXPECTED_SCENES,
    FAMILIES,
    PRIMITIVES,
    ROLES,
    ROWS_PER_SOURCE,
    SourceRef,
    TIME_TOLERANCE_S,
    discover_sources,
)
from lewm.datasets import go2_recurrent_h4_rgb_sequences as _v1


SCHEMA = "lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
ALIGNMENT = (
    "pre_request_final_previous_block -> fifth_tick_current_block; "
    "requested_target_block_action; no_next_block_tick"
)
SEED = _v1.SEED
TRAIN_PER_FAMILY = _v1.TRAIN_PER_FAMILY
VAL_PER_FAMILY = _v1.VAL_PER_FAMILY
PER_SCENE_CANDIDATE_CAP = _v1.PER_SCENE_CANDIDATE_CAP
ACTION_TO_INDEX = _v1.ACTION_TO_INDEX
Endpoint = _v1.Endpoint
SequenceContractError = _v1.SequenceContractError
CENSUS_SOURCE_BINDING_SHA256 = (
    "0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696"
)


@dataclass(frozen=True)
class H6Window:
    rank: str
    role: str
    family: str
    scene_id: str
    endpoints: tuple[Endpoint, ...]
    actions: tuple[int, ...]

    def to_row(self) -> dict[str, Any]:
        _validate_causal_window(self)
        return {
            "schema": SCHEMA,
            "role": self.role,
            "family": self.family,
            "scene_id": self.scene_id,
            "rgb": [
                (
                    f"{self.scene_id}/rgb/"
                    f"frame_{endpoint.frame_index:06d}_env_{endpoint.env_index:02d}.png"
                )
                for endpoint in self.endpoints
            ],
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class _FrameRecord:
    key: tuple[int, str, int]
    endpoint: Endpoint
    signature: tuple[str, str]
    primitive: str
    block_size: int
    command_dt_s: float
    request_timestamp_ns: int


def _validate_causal_window(window: H6Window) -> None:
    # V1's structural validator is independent of the serialized schema.
    _v1._validate_window(window)
    for left, right in zip(window.endpoints, window.endpoints[1:]):
        if (
            right.env_index != left.env_index
            or right.frame_index != left.frame_index + BLOCK_SIZE * ENVS_PER_SOURCE
            or right.episode_step != left.episode_step + BLOCK_SIZE
            or abs(
                (right.timestamp_ns - left.timestamp_ns) * 1e-9
                - BLOCK_SIZE * COMMAND_DT_S
            )
            > TIME_TOLERANCE_S
        ):
            raise SequenceContractError("V2 window is not six causal five-tick edges")


def _wanted_frame_indices(windows: Sequence[_v1.H6Window]) -> set[int]:
    wanted: set[int] = set()
    for window in windows:
        first = window.endpoints[0]
        if first.episode_step == 1:
            continue
        wanted.add(first.frame_index - ENVS_PER_SOURCE)
        for command_first in window.endpoints[:6]:
            wanted.update(
                command_first.frame_index + tick * ENVS_PER_SOURCE
                for tick in range(BLOCK_SIZE)
            )
    return wanted


def _load_wanted_records(
    repo_root: Path,
    source: SourceRef,
    *,
    expected_scene_id: str,
    expected_manifest: str,
    wanted: set[int],
) -> tuple[dict[int, _FrameRecord], int, str]:
    scene_id, manifest = _v1._load_source_identity(repo_root, source)
    if scene_id != expected_scene_id or manifest != expected_manifest:
        raise SequenceContractError("V1 and V2 source identity disagree")

    source_fd = _v1._open_absolute_directory(_v1._source_directory(repo_root, source))
    try:
        frames_fd = os.open("frames.jsonl", _v1._FILE_FLAGS, dir_fd=source_fd)
    finally:
        os.close(source_fd)
    initial_info = os.fstat(frames_fd)
    if not stat.S_ISREG(initial_info.st_mode) or int(initial_info.st_size) != source.byte_count:
        os.close(frames_fd)
        raise SequenceContractError("frames leaf changed before V2 endpoint scan")

    records: dict[int, _FrameRecord] = {}
    content_digest = hashlib.sha256()
    content_byte_count = 0
    final_info: os.stat_result | None = None
    try:
        with os.fdopen(frames_fd, "rb", closefd=True) as stream:
            for raw_line in stream:
                content_digest.update(raw_line)
                content_byte_count += len(raw_line)
                payload = _v1._strict_json_loads(raw_line)
                if not isinstance(payload, dict):
                    raise SequenceContractError("frame row is not an object")
                frame_index = payload.get("frame_index")
                if frame_index not in wanted:
                    continue
                (
                    key,
                    endpoint,
                    signature,
                    primitive,
                    block_size,
                    command_dt_s,
                ) = _v1._parse_frame(payload, source=source, manifest=manifest)
                context = payload.get("command_context")
                request_timestamp_ns = (
                    context.get("timestamp_ns") if isinstance(context, Mapping) else None
                )
                if (
                    not isinstance(request_timestamp_ns, int)
                    or isinstance(request_timestamp_ns, bool)
                    or endpoint.frame_index in records
                ):
                    raise SequenceContractError("requested-command timestamp or frame identity changed")
                records[endpoint.frame_index] = _FrameRecord(
                    key=key,
                    endpoint=endpoint,
                    signature=signature,
                    primitive=primitive,
                    block_size=block_size,
                    command_dt_s=command_dt_s,
                    request_timestamp_ns=request_timestamp_ns,
                )
            final_info = os.fstat(stream.fileno())
    except BaseException:
        try:
            os.close(frames_fd)
        except OSError:
            pass
        raise

    if set(records) != wanted:
        raise SequenceContractError("a required real V2 boundary row is absent")
    if (
        final_info is None
        or not stat.S_ISREG(final_info.st_mode)
        or int(final_info.st_size) != source.byte_count
        or final_info.st_dev != initial_info.st_dev
        or final_info.st_ino != initial_info.st_ino
        or content_byte_count != initial_info.st_size
    ):
        raise SequenceContractError("frames leaf changed during V2 endpoint scan")
    return records, content_byte_count, content_digest.hexdigest()


def _adapt_legacy_window(
    legacy: _v1.H6Window,
    records: Mapping[int, _FrameRecord],
) -> H6Window | None:
    legacy_first = legacy.endpoints[0]
    if legacy_first.episode_step == 1:
        return None

    preceding_index = legacy_first.frame_index - ENVS_PER_SOURCE
    preceding = records.get(preceding_index)
    if preceding is None:
        raise SequenceContractError("noninitial V1 group lacks its real predecessor")
    stream_key = preceding.key
    endpoints: list[Endpoint] = [preceding.endpoint]

    for position, command_first in enumerate(legacy.endpoints[:6]):
        rows = [
            records.get(command_first.frame_index + tick * ENVS_PER_SOURCE)
            for tick in range(BLOCK_SIZE)
        ]
        if any(row is None for row in rows):
            raise SequenceContractError("causal command block is incomplete")
        command_rows = tuple(row for row in rows if row is not None)
        if command_rows[0].endpoint.identity != command_first.identity:
            raise SequenceContractError("legacy command anchor changed")
        if any(
            row.key != stream_key
            or row.signature != command_rows[0].signature
            or row.primitive != command_rows[0].primitive
            or row.block_size != BLOCK_SIZE
            or abs(row.command_dt_s - COMMAND_DT_S) > TIME_TOLERANCE_S
            or row.request_timestamp_ns != command_rows[0].request_timestamp_ns
            for row in command_rows
        ):
            raise SequenceContractError("target command context is not stable and reset-safe")
        if ACTION_TO_INDEX[command_rows[0].primitive] != legacy.actions[position]:
            raise SequenceContractError("requested target-block primitive changed")
        if command_rows[0].request_timestamp_ns != endpoints[-1].timestamp_ns:
            raise SequenceContractError("source endpoint is not the pre-request boundary")

        chain = (endpoints[-1],) + tuple(row.endpoint for row in command_rows)
        for left, right in zip(chain, chain[1:]):
            if (
                right.env_index != left.env_index
                or right.frame_index != left.frame_index + ENVS_PER_SOURCE
                or right.episode_step != left.episode_step + 1
                or abs(
                    (right.timestamp_ns - left.timestamp_ns) * 1e-9
                    - COMMAND_DT_S
                )
                > TIME_TOLERANCE_S
            ):
                raise SequenceContractError("causal edge does not contain five regular ticks")
        endpoints.append(command_rows[-1].endpoint)

    adapted = H6Window(
        rank=legacy.rank,
        role=legacy.role,
        family=legacy.family,
        scene_id=legacy.scene_id,
        endpoints=tuple(endpoints),
        actions=legacy.actions,
    )
    _validate_causal_window(adapted)
    return adapted


def scan_source_candidates(
    repo_root: str,
    source: SourceRef,
    per_scene_cap: int,
) -> dict[str, Any]:
    """Adapt the complete ordered V1 candidate list before applying the cap."""

    legacy = _v1.scan_source_candidates(repo_root, source, ROWS_PER_SOURCE)
    legacy_candidates: list[_v1.H6Window] = legacy["candidates"]
    if len(legacy_candidates) != int(legacy["candidate_count"]):
        raise SequenceContractError("full V1 candidate order was not materialized")

    records, source_content_byte_count, source_content_sha256 = _load_wanted_records(
        Path(repo_root),
        source,
        expected_scene_id=str(legacy["scene_id"]),
        expected_manifest=str(legacy["manifest_sha256"]),
        wanted=_wanted_frame_indices(legacy_candidates),
    )
    adapted: list[H6Window] = []
    discarded_initial = 0
    for legacy_window in legacy_candidates:
        window = _adapt_legacy_window(legacy_window, records)
        if window is None:
            discarded_initial += 1
        else:
            adapted.append(window)

    if any(left.rank > right.rank for left, right in zip(adapted, adapted[1:])):
        raise SequenceContractError("V1 candidate rank order changed")
    if len(adapted) < per_scene_cap:
        raise SequenceContractError(
            "fewer than the required causal-valid candidates remain before the cap"
        )
    return {
        "source_ordinal": source.ordinal,
        "role": source.role,
        "family": source.family,
        "scene_id": legacy["scene_id"],
        "manifest_sha256": legacy["manifest_sha256"],
        "legacy_candidate_count": len(legacy_candidates),
        "candidate_count": len(adapted),
        "emitted_causal_group_count": len(adapted),
        "discarded_initial_group_count": discarded_initial,
        "v2_actual_metadata_row_count": len(records),
        "source_content_byte_count": source_content_byte_count,
        "source_content_sha256": source_content_sha256,
        "candidates": adapted[:per_scene_cap],
    }


def _select_role_family(
    records: Sequence[dict[str, Any]],
    *,
    target: int,
) -> list[H6Window]:
    return _v1._select_role_family(records, target=target)


def _interleave_families(
    by_family: Mapping[str, Sequence[H6Window]],
) -> list[H6Window]:
    return _v1._interleave_families(by_family)


def _coverage(windows: Sequence[H6Window]) -> dict[str, Any]:
    return _v1._coverage(windows)


def validate_selected_rgb(
    repo_root: Path,
    windows: Sequence[H6Window],
) -> dict[str, int]:
    return _v1.validate_selected_rgb(repo_root, windows)


def _require_census_source_binding(
    sources: Sequence[SourceRef],
    results: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str = CENSUS_SOURCE_BINDING_SHA256,
) -> tuple[str, int]:
    ordered_sources = sorted(sources, key=lambda source: source.ordinal)
    result_by_ordinal = {
        int(result["source_ordinal"]): result for result in results
    }
    if (
        len(result_by_ordinal) != len(results)
        or len(ordered_sources) != len(results)
        or set(result_by_ordinal) != {source.ordinal for source in ordered_sources}
    ):
        raise SequenceContractError("observed source binding rows are incomplete or duplicated")

    binding_rows: list[list[Any]] = []
    for source in ordered_sources:
        result = result_by_ordinal[source.ordinal]
        byte_count = int(result["source_content_byte_count"])
        content_sha256 = str(result["source_content_sha256"])
        if (
            result["role"] != source.role
            or result["family"] != source.family
            or byte_count != source.byte_count
            or len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
        ):
            raise SequenceContractError("observed source content identity changed")
        binding_rows.append(
            [
                source.role,
                source.family,
                source.chunk,
                source.sequence,
                byte_count,
                content_sha256,
            ]
        )

    observed_sha256 = hashlib.sha256(
        json.dumps(
            binding_rows,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SequenceContractError("live source-content binding disagrees with frozen census")
    return observed_sha256, len(binding_rows)


def build_index(
    repo_root: Path,
    *,
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, list[H6Window]], dict[str, Any]]:
    sources, discovery_access = discover_sources(repo_root)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                scan_source_candidates,
                str(repo_root),
                source,
                PER_SCENE_CANDIDATE_CAP,
            ): source.ordinal
            for source in sources
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress is not None:
                progress(len(results), len(sources))
    results.sort(key=lambda item: int(item["source_ordinal"]))
    observed_binding_sha256, observed_binding_row_count = (
        _require_census_source_binding(sources, results)
    )

    indexes: dict[str, list[H6Window]] = {}
    scenes_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    manifests_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    for result in results:
        role = str(result["role"])
        scene = str(result["scene_id"])
        manifest_sha256 = str(result["manifest_sha256"])
        if scene in scenes_by_role[role] or manifest_sha256 in manifests_by_role[role]:
            raise SequenceContractError("duplicate scene or manifest within role")
        scenes_by_role[role].add(scene)
        manifests_by_role[role].add(manifest_sha256)
    if (
        scenes_by_role["train"] & scenes_by_role["val"]
        or manifests_by_role["train"] & manifests_by_role["val"]
    ):
        raise SequenceContractError("train/validation scene or manifest overlap")

    for role, target in (("train", TRAIN_PER_FAMILY), ("val", VAL_PER_FAMILY)):
        selected_by_family: dict[str, list[H6Window]] = {}
        for family in FAMILIES:
            family_records = [
                result
                for result in results
                if result["role"] == role and result["family"] == family
            ]
            if len(family_records) != EXPECTED_SCENES[role][family]:
                raise SequenceContractError("role/family scene count changed")
            selected_by_family[family] = _select_role_family(
                family_records,
                target=target,
            )
        indexes[role] = _interleave_families(selected_by_family)

    train_coverage = _coverage(indexes["train"])
    val_coverage = _coverage(indexes["val"])
    if (
        train_coverage["missing_action_position_cells"]
        or val_coverage["missing_action_position_cells"]
    ):
        raise SequenceContractError("selected schedule lacks an action-position cell")
    rgb = validate_selected_rgb(repo_root, [*indexes["train"], *indexes["val"]])
    manifest = {
        "schema": SCHEMA,
        "seed": SEED,
        "selection": {
            "method": "v1_rank_and_phase_then_causal_filter_and_v1_backfill",
            "legacy_rank_anchor": "first_post_command_frame_of_group_p0",
            "transition_disjoint_h6": True,
            "adjacent_packed_groups_share_one_boundary_rgb": True,
            "initial_six_command_group_per_episode_discarded": True,
            "candidate_materialization": (
                "complete_per_source_v1_rank_order_in_worker_then_v2_filter_then_cap"
            ),
            "minimum_causal_valid_candidates_per_scene": PER_SCENE_CANDIDATE_CAP,
            "per_scene_candidate_cap": PER_SCENE_CANDIDATE_CAP,
            "train_per_family": TRAIN_PER_FAMILY,
            "val_per_family": VAL_PER_FAMILY,
            "train_presentation_cap": 16_000,
        },
        "sequence_schema": {
            "action_vocabulary": list(PRIMITIVES),
            "rgb_count": 7,
            "action_count": 6,
            "alignment": ALIGNMENT,
            "history": "rgb[0:3] with actions[0:2]",
            "future": "predict rgb[3:7] with actions[2:6]",
        },
        "train": train_coverage,
        "val": val_coverage,
        "rgb_validation": {
            "method": "exact_leaf_nofollow_regular_file_and_png_signature",
            **rgb,
        },
        "source": {
            "scene_count": len(results),
            "observed_ordered_source_content_binding_sha256": (
                observed_binding_sha256
            ),
            "observed_source_content_binding_row_count": observed_binding_row_count,
            "legacy_candidate_count": sum(
                int(result["legacy_candidate_count"]) for result in results
            ),
            "candidate_count": sum(int(result["candidate_count"]) for result in results),
            "emitted_causal_group_count": sum(
                int(result["emitted_causal_group_count"]) for result in results
            ),
            "discarded_initial_group_count": sum(
                int(result["discarded_initial_group_count"]) for result in results
            ),
            "v2_actual_metadata_row_count": sum(
                int(result["v2_actual_metadata_row_count"]) for result in results
            ),
            "frames_jsonl_full_pass_count": 2 * len(results),
            "discovery_access": discovery_access,
            "rgb_root_list_count": 0,
            "untrusted_row_string_path_open_count": 0,
            "validated_numeric_row_derived_rgb_open_count": rgb["unique_rgb_count"],
            "test_heldout_sealed_open_count": 0,
            "label_or_raw_message_open_count": 0,
        },
    }
    return indexes, manifest


def canonical_row_bytes(window: H6Window) -> bytes:
    return (
        json.dumps(
            window.to_row(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "ACTION_TO_INDEX",
    "ALIGNMENT",
    "Endpoint",
    "H6Window",
    "PER_SCENE_CANDIDATE_CAP",
    "SCHEMA",
    "SEED",
    "SequenceContractError",
    "TRAIN_PER_FAMILY",
    "VAL_PER_FAMILY",
    "build_index",
    "canonical_row_bytes",
    "scan_source_candidates",
    "validate_selected_rgb",
]
