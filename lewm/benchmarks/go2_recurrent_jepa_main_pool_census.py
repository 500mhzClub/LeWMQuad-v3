"""Reset-safe temporal census for the public datagen_full train/val pool.

The census deliberately reads only ``frames.jsonl`` metadata below two exact
roles.  It never follows paths from a row and never visits render, RGB, label,
raw-message, test, held-out, or sealed roots.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Callable, Mapping


SCHEMA = "lewm_go2_recurrent_jepa_main_pool_census_v2"
ROLES = ("train", "val")
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
PRIMITIVES = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
EXPECTED_SCENES = {
    "train": {
        "large_enclosed_maze": 150,
        "local_composite_motifs": 150,
        "loop_alias_stress": 100,
        "medium_enclosed_maze": 250,
        "open_obstacle_field": 100,
        "rough_local_dynamics": 50,
        "small_enclosed_maze": 150,
        "visual_sensor_stress": 50,
    },
    "val": {
        "large_enclosed_maze": 22,
        "local_composite_motifs": 23,
        "loop_alias_stress": 15,
        "medium_enclosed_maze": 38,
        "open_obstacle_field": 15,
        "rough_local_dynamics": 7,
        "small_enclosed_maze": 23,
        "visual_sensor_stress": 7,
    },
}

ROWS_PER_SOURCE = 48_000
ENVS_PER_SOURCE = 48
STEPS_PER_STREAM = 1_000
BLOCK_SIZE = 5
COMMAND_DT_S = 0.1
TIME_TOLERANCE_S = 2e-4
TRANSITIONS_PER_STREAM = 199
SLIDING_H6_PER_STREAM = 194
PACKED_H6_PER_STREAM = 33

_CHUNK_RE = re.compile(r"^chunk_[0-9]{4}$")
_SHA12_RE = r"[0-9a-f]{12}"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSET = object()
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class CensusContractError(RuntimeError):
    """The allowlisted corpus layout or row schema violated the contract."""


@dataclass(frozen=True)
class SourceRef:
    role: str
    family: str
    chunk: str
    sequence: str
    byte_count: int
    ordinal: int


@dataclass(frozen=True)
class RowMeta:
    step: int
    timestamp_ns: int
    frame_index: int
    sequence_id: str | None
    primitive: str
    block_size: int | None
    command_dt_s: float | None

    @property
    def signature(self) -> tuple[str | None, str]:
        return self.sequence_id, self.primitive

    @property
    def endpoint_id(self) -> tuple[int, int, int]:
        return self.step, self.timestamp_ns, self.frame_index


@dataclass
class RunState:
    signature: tuple[str | None, str]
    primitive: str
    block_size: int | None
    command_dt_s: float | None
    rows: list[RowMeta]
    context_stable: bool = True


@dataclass
class StreamState:
    scene_id: Any
    manifest_sha256: str
    split: str
    last: RowMeta
    run: RunState
    path_length: int = 0
    path_actions: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    last_transition_endpoint: tuple[int, int, int] | None = None
    contributed_h6: bool = False


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {".", "..", ""} for part in path.parts[1:]):
        raise CensusContractError("directory path must be canonical and absolute")
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise CensusContractError("descriptor-relative no-follow opens are required")
    descriptor = os.open(path.anchor, _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise CensusContractError("non-canonical directory component")
    return os.open(name, _DIR_FLAGS, dir_fd=parent)


def _valid_chunk(family: str, name: str) -> bool:
    return bool(_CHUNK_RE.fullmatch(name)) or (
        family == "rough_local_dynamics" and name == "chunk_backfill"
    )


def _sequence_pattern(family: str) -> re.Pattern[str]:
    return re.compile(rf"^[0-9]{{6}}_{re.escape(family)}_{_SHA12_RE}$")


def discover_sources(repo_root: Path) -> tuple[list[SourceRef], dict[str, int]]:
    """Inventory exact allowlisted leaves without traversing sibling roots."""

    repo_root = Path(repo_root)
    access: Counter[str] = Counter()
    sources: list[SourceRef] = []
    ordinal = 0
    for role in ROLES:
        split_path = repo_root / ".generated" / "datagen_full" / "rollout" / role
        access["split_directory_open_attempt_count"] += 1
        split_fd = _open_absolute_directory(split_path)
        access["split_directory_open_success_count"] += 1
        try:
            for family in FAMILIES:
                access["family_directory_open_attempt_count"] += 1
                family_fd = _open_child_directory(split_fd, family)
                access["family_directory_open_success_count"] += 1
                try:
                    access["family_name_listing_count"] += 1
                    chunk_names = os.listdir(family_fd)
                    accepted_chunks = sorted(
                        name for name in chunk_names if _valid_chunk(family, name)
                    )
                    access["ignored_family_entry_name_count"] += (
                        len(chunk_names) - len(accepted_chunks)
                    )
                    pattern = _sequence_pattern(family)
                    for chunk in accepted_chunks:
                        access["chunk_directory_open_attempt_count"] += 1
                        chunk_fd = _open_child_directory(family_fd, chunk)
                        access["chunk_directory_open_success_count"] += 1
                        try:
                            access["plan_directory_open_attempt_count"] += 1
                            plan_fd = _open_child_directory(chunk_fd, "plan")
                            access["plan_directory_open_success_count"] += 1
                            try:
                                access["plan_name_listing_count"] += 1
                                plan_names = os.listdir(plan_fd)
                                sequence_names = sorted(
                                    name for name in plan_names if pattern.fullmatch(name)
                                )
                                access["ignored_plan_entry_name_count"] += (
                                    len(plan_names) - len(sequence_names)
                                )
                                for sequence in sequence_names:
                                    access["sequence_directory_open_attempt_count"] += 1
                                    sequence_fd = _open_child_directory(plan_fd, sequence)
                                    access["sequence_directory_open_success_count"] += 1
                                    try:
                                        access["inventory_frames_open_attempt_count"] += 1
                                        frames_fd = os.open(
                                            "frames.jsonl", _FILE_FLAGS, dir_fd=sequence_fd
                                        )
                                        access["inventory_frames_open_success_count"] += 1
                                        try:
                                            info = os.fstat(frames_fd)
                                            if not stat.S_ISREG(info.st_mode):
                                                access["nonregular_frames_rejection_count"] += 1
                                                continue
                                            ordinal += 1
                                            sources.append(
                                                SourceRef(
                                                    role=role,
                                                    family=family,
                                                    chunk=chunk,
                                                    sequence=sequence,
                                                    byte_count=int(info.st_size),
                                                    ordinal=ordinal,
                                                )
                                            )
                                        finally:
                                            os.close(frames_fd)
                                    finally:
                                        os.close(sequence_fd)
                            finally:
                                os.close(plan_fd)
                        finally:
                            os.close(chunk_fd)
                finally:
                    os.close(family_fd)
        finally:
            os.close(split_fd)
    access["row_derived_path_open_attempt_count"] = 0
    access["forbidden_open_or_stat_attempt_count"] = 0
    return sources, dict(sorted(access.items()))


def _open_source(repo_root: Path, source: SourceRef) -> tuple[int, int]:
    directory = (
        repo_root
        / ".generated"
        / "datagen_full"
        / "rollout"
        / source.role
        / source.family
        / source.chunk
        / "plan"
        / source.sequence
    )
    directory_fd = _open_absolute_directory(directory)
    try:
        descriptor = os.open("frames.jsonl", _FILE_FLAGS, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise CensusContractError("frames leaf is not a regular file")
    return descriptor, int(info.st_size)


def _new_run(row: RowMeta) -> RunState:
    return RunState(
        signature=row.signature,
        primitive=row.primitive,
        block_size=row.block_size,
        command_dt_s=row.command_dt_s,
        rows=[row],
    )


class _SourceScanner:
    def __init__(self, role: str, family: str) -> None:
        self.role = role
        self.family = family
        self.counts: Counter[str] = Counter()
        self.integrity: Counter[str] = Counter()
        self.window_counts: Counter[str] = Counter()
        self.action_position_counts: Counter[str] = Counter()
        self.action_position_presence: set[str] = set()
        self.maximal_path_histogram: Counter[str] = Counter()
        self.packed_leftover_histogram: Counter[str] = Counter()
        self.streams: dict[tuple[int, str, int], StreamState] = {}
        self.env_indices: set[int] = set()
        self.env_row_counts: Counter[int] = Counter()
        self.h6_streams: set[tuple[int, str, int]] = set()
        self.file_scene_id: Any = _UNSET
        self.file_manifest_sha256: str | None = None
        self.last_global_frame_index: int | None = None
        self.first_global_frame_index: int | None = None

    def malformed(self, name: str) -> None:
        self.integrity[name] += 1

    def _finish_path(self, state: StreamState) -> None:
        if not state.path_length:
            return
        length = state.path_length
        self.maximal_path_histogram[str(length)] += 1
        self.counts["packed_h6"] += length // 6
        self.packed_leftover_histogram[str(length % 6)] += 1
        self.counts["accepted_transitions_covered_by_paths"] += length
        state.path_length = 0
        state.path_actions.clear()
        state.last_transition_endpoint = None

    def _emit_transition(
        self,
        key: tuple[int, str, int],
        state: StreamState,
        current: RowMeta,
        endpoint: RowMeta,
        primitive: str,
    ) -> None:
        if (
            state.path_length
            and state.last_transition_endpoint != current.endpoint_id
        ):
            self.integrity["nonconsecutive_transition_join_count"] += 1
            self._finish_path(state)
        state.path_length += 1
        state.path_actions.append(primitive)
        state.last_transition_endpoint = endpoint.endpoint_id
        self.counts["primitive_transitions"] += 1
        for horizon in range(1, min(6, state.path_length) + 1):
            self.window_counts[f"h{horizon}"] += 1
        if state.path_length >= 6:
            state.contributed_h6 = True
            self.h6_streams.add(key)
            actions = tuple(state.path_actions)
            for position in range(2, 6):
                cell = f"p{position}:{actions[position]}"
                self.action_position_counts[cell] += 1
                self.action_position_presence.add(cell)

    def _finalize_run(
        self,
        key: tuple[int, str, int],
        state: StreamState,
        endpoint: RowMeta | None,
    ) -> None:
        run = state.run
        valid = True
        if run.signature[0] is None or not run.primitive:
            self.integrity["incomplete_command_context_count"] += 1
            valid = False
        if run.primitive not in PRIMITIVES:
            self.integrity["unknown_primitive_count"] += 1
            valid = False
        if run.block_size != BLOCK_SIZE:
            self.integrity["invalid_block_size_count"] += 1
            valid = False
        if (
            run.command_dt_s is None
            or not math.isfinite(run.command_dt_s)
            or abs(run.command_dt_s - COMMAND_DT_S) > TIME_TOLERANCE_S
        ):
            self.integrity["invalid_command_dt_count"] += 1
            valid = False
        if not run.context_stable:
            self.integrity["command_context_drift_count"] += 1
            valid = False
        if len(run.rows) != BLOCK_SIZE:
            self.integrity["run_length_mismatch_count"] += 1
            valid = False

        if endpoint is None:
            if valid:
                self.counts["terminal_runs_without_endpoint"] += 1
            self._finish_path(state)
            return

        window = [*run.rows, endpoint]
        if any(
            right.step != left.step + 1 for left, right in zip(window, window[1:])
        ):
            self.integrity["run_nonconsecutive_step_count"] += 1
            valid = False
        if run.command_dt_s is not None and math.isfinite(run.command_dt_s):
            deltas = [
                (right.timestamp_ns - left.timestamp_ns) * 1e-9
                for left, right in zip(window, window[1:])
            ]
            if any(
                abs(delta - run.command_dt_s) > TIME_TOLERANCE_S
                for delta in deltas
            ):
                self.integrity["run_irregular_tick_timing_count"] += 1
                valid = False
            actual_duration = (
                endpoint.timestamp_ns - run.rows[0].timestamp_ns
            ) * 1e-9
            expected_duration = BLOCK_SIZE * run.command_dt_s
            if (
                abs(expected_duration - 0.5) > TIME_TOLERANCE_S
                or abs(actual_duration - expected_duration) > TIME_TOLERANCE_S
            ):
                self.integrity["run_wrong_duration_count"] += 1
                valid = False
        if valid:
            self._emit_transition(
                key, state, run.rows[0], endpoint, run.primitive
            )
        else:
            self._finish_path(state)

    def _parse_row(self, payload: Mapping[str, Any]) -> tuple[
        tuple[int, str, int], Any, str, str, RowMeta
    ] | None:
        episode = payload.get("episode")
        context = payload.get("command_context")
        if not isinstance(episode, Mapping):
            self.malformed("missing_episode_count")
            return None
        if not isinstance(context, Mapping):
            self.malformed("missing_command_context_count")
            return None
        env_index = payload.get("env_index")
        frame_index = payload.get("frame_index")
        timestamp_ns = payload.get("timestamp_ns")
        step = episode.get("episode_step")
        reset_count = episode.get("reset_count")
        episode_id = episode.get("episode_id")
        split = episode.get("split")
        manifest = episode.get("manifest_sha256")
        if not all(
            _is_plain_int(value)
            for value in (env_index, frame_index, timestamp_ns, step, reset_count)
        ):
            self.malformed("invalid_integer_field_count")
            return None
        if not isinstance(episode_id, (str, int)) or isinstance(episode_id, bool):
            self.malformed("invalid_episode_id_count")
            return None
        if not isinstance(split, str) or not isinstance(manifest, str):
            self.malformed("invalid_episode_context_count")
            return None
        scene_id = episode.get("scene_id")
        if (
            not isinstance(scene_id, (str, int))
            or isinstance(scene_id, bool)
            or (isinstance(scene_id, str) and not scene_id)
        ):
            self.malformed("invalid_scene_id_count")
            return None
        if not _SHA256_RE.fullmatch(manifest):
            self.malformed("invalid_manifest_sha256_count")
            return None
        if int(step) < 1 or int(reset_count) < 0:
            self.malformed("invalid_episode_counter_count")
            return None
        sequence_value = context.get("sequence_id")
        sequence_id = (
            str(sequence_value)
            if isinstance(sequence_value, (str, int)) and not isinstance(sequence_value, bool)
            else None
        )
        primitive = context.get("primitive_name")
        primitive = primitive if isinstance(primitive, str) else ""
        block_value = context.get("block_size")
        block_size = int(block_value) if _is_plain_int(block_value) else None
        dt_value = context.get("command_dt_s")
        command_dt_s = (
            float(dt_value)
            if isinstance(dt_value, (int, float)) and not isinstance(dt_value, bool)
            else None
        )
        row = RowMeta(
            step=int(step),
            timestamp_ns=int(timestamp_ns),
            frame_index=int(frame_index),
            sequence_id=sequence_id,
            primitive=primitive,
            block_size=block_size,
            command_dt_s=command_dt_s,
        )
        key = int(env_index), str(episode_id), int(reset_count)
        return key, scene_id, manifest, split, row

    def add_payload(self, payload: Mapping[str, Any]) -> None:
        parsed = self._parse_row(payload)
        if parsed is None:
            return
        key, scene_id, manifest, split, row = parsed
        env_index = key[0]
        self.env_indices.add(env_index)
        self.env_row_counts[env_index] += 1
        if not 0 <= env_index < ENVS_PER_SOURCE:
            self.integrity["env_index_out_of_range_count"] += 1
        if split != self.role:
            self.integrity["role_mismatch_count"] += 1
        if self.file_scene_id is _UNSET:
            self.file_scene_id = scene_id
        elif scene_id != self.file_scene_id:
            self.integrity["file_scene_drift_count"] += 1
        if self.file_manifest_sha256 is None:
            self.file_manifest_sha256 = manifest
        elif manifest != self.file_manifest_sha256:
            self.integrity["file_manifest_drift_count"] += 1

        if self.first_global_frame_index is None:
            self.first_global_frame_index = row.frame_index
        if (
            self.last_global_frame_index is not None
            and row.frame_index != self.last_global_frame_index + 1
        ):
            self.integrity["global_frame_index_discontinuity_count"] += 1
        self.last_global_frame_index = row.frame_index

        state = self.streams.get(key)
        if state is None:
            if row.step != 1:
                self.integrity["stream_first_step_not_one_count"] += 1
            self.streams[key] = StreamState(
                scene_id=scene_id,
                manifest_sha256=manifest,
                split=split,
                last=row,
                run=_new_run(row),
            )
            return
        if (
            state.scene_id != scene_id
            or state.manifest_sha256 != manifest
            or state.split != split
        ):
            self.integrity["stream_context_drift_count"] += 1
        discontinuous = False
        if row.step <= state.last.step:
            self.integrity["duplicate_or_nonmonotone_stream_step_count"] += 1
            discontinuous = True
        elif row.step != state.last.step + 1:
            self.integrity["stream_step_gap_count"] += 1
            discontinuous = True
        if row.timestamp_ns <= state.last.timestamp_ns:
            self.integrity["nonmonotone_stream_timestamp_count"] += 1
            discontinuous = True
        if discontinuous:
            state.run.context_stable = False
            self._finalize_run(key, state, None)
            state.run = _new_run(row)
        elif row.signature == state.run.signature:
            if (
                row.block_size != state.run.block_size
                or row.command_dt_s != state.run.command_dt_s
            ):
                state.run.context_stable = False
            state.run.rows.append(row)
        else:
            self._finalize_run(key, state, row)
            state.run = _new_run(row)
        state.last = row

    def finish(self) -> dict[str, Any]:
        for key, state in self.streams.items():
            self._finalize_run(key, state, None)
        env_row_count_violation_count = sum(
            count != STEPS_PER_STREAM for count in self.env_row_counts.values()
        )
        scene_id = None if self.file_scene_id is _UNSET else self.file_scene_id
        return {
            "role": self.role,
            "family": self.family,
            "row_count": self.counts["row_count"],
            "byte_count": self.counts["byte_count"],
            "stream_count": len(self.streams),
            "env_count": len(self.env_indices),
            "env_row_count_violation_count": env_row_count_violation_count,
            "first_global_frame_index": self.first_global_frame_index,
            "last_global_frame_index": self.last_global_frame_index,
            "primitive_transitions": self.counts["primitive_transitions"],
            "terminal_runs_without_endpoint": self.counts[
                "terminal_runs_without_endpoint"
            ],
            "accepted_transitions_covered_by_paths": self.counts[
                "accepted_transitions_covered_by_paths"
            ],
            "window_counts": dict(sorted(self.window_counts.items())),
            "packed_h6": self.counts["packed_h6"],
            "contributing_h6_stream_count": len(self.h6_streams),
            "action_position_counts": dict(sorted(self.action_position_counts.items())),
            "action_position_presence": sorted(self.action_position_presence),
            "maximal_path_histogram": dict(sorted(self.maximal_path_histogram.items())),
            "packed_leftover_histogram": dict(
                sorted(self.packed_leftover_histogram.items())
            ),
            "integrity": dict(sorted(self.integrity.items())),
            "_scene_id_type": "unset" if scene_id is None else type(scene_id).__name__,
            "_scene_id": scene_id,
            "_manifest_sha256": self.file_manifest_sha256,
        }


def _strict_json_loads(raw: bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def scan_binary_stream(stream: BinaryIO, *, role: str, family: str) -> dict[str, Any]:
    scanner = _SourceScanner(role, family)
    content_digest = hashlib.sha256()
    for raw_line in stream:
        content_digest.update(raw_line)
        scanner.counts["byte_count"] += len(raw_line)
        if not raw_line.strip():
            scanner.integrity["blank_line_count"] += 1
            continue
        scanner.counts["row_count"] += 1
        try:
            payload = _strict_json_loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            scanner.integrity["json_decode_failure_count"] += 1
            continue
        if not isinstance(payload, dict):
            scanner.integrity["non_object_row_count"] += 1
            continue
        scanner.add_payload(payload)
    result = scanner.finish()
    result["_content_sha256"] = content_digest.hexdigest()
    return result


def scan_source(repo_root: str, source: SourceRef) -> dict[str, Any]:
    descriptor, observed_size = _open_source(Path(repo_root), source)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            result = scan_binary_stream(
                stream, role=source.role, family=source.family
            )
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    result["source_ordinal"] = source.ordinal
    result["inventory_byte_count"] = source.byte_count
    result["scan_fstat_byte_count"] = observed_size
    return result


def _empty_family_result() -> dict[str, Any]:
    return {
        "source_count": 0,
        "byte_count": 0,
        "row_count": 0,
        "stream_count": 0,
        "primitive_transitions": 0,
        "terminal_runs_without_endpoint": 0,
        "window_counts": {f"h{h}": 0 for h in range(1, 7)},
        "packed_h6": 0,
        "contributing_h6_source_count": 0,
        "contributing_h6_stream_count": 0,
        "action_position_counts": {
            f"p{position}:{primitive}": 0
            for position in range(2, 6)
            for primitive in PRIMITIVES
        },
        "action_position_source_counts": {
            f"p{position}:{primitive}": 0
            for position in range(2, 6)
            for primitive in PRIMITIVES
        },
        "maximal_path_histogram": {},
        "packed_leftover_histogram": {},
        "source_shape_failure_count": 0,
    }


def _merge_counter(target: dict[str, int], values: Mapping[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + int(value)


def aggregate_results(
    sources: list[SourceRef],
    results: list[dict[str, Any]],
    access: Mapping[str, int],
) -> dict[str, Any]:
    by_role_family = {
        role: {family: _empty_family_result() for family in FAMILIES}
        for role in ROLES
    }
    integrity: Counter[str] = Counter()
    scene_ids_by_role: dict[str, set[tuple[str, str]]] = {
        role: set() for role in ROLES
    }
    manifests_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    scene_ids_by_cell: dict[tuple[str, str], set[tuple[str, str]]] = {
        (role, family): set() for role in ROLES for family in FAMILIES
    }
    manifests_by_cell: dict[tuple[str, str], set[str]] = {
        (role, family): set() for role in ROLES for family in FAMILIES
    }
    duplicate_scene_id_count = 0
    duplicate_manifest_count = 0
    binding_rows: list[list[Any]] = []
    source_by_ordinal = {source.ordinal: source for source in sources}
    for result in results:
        source = source_by_ordinal[int(result["source_ordinal"])]
        family_result = by_role_family[source.role][source.family]
        scene_key = (str(result["_scene_id_type"]), str(result["_scene_id"]))
        manifest = str(result["_manifest_sha256"])
        if scene_key in scene_ids_by_role[source.role]:
            duplicate_scene_id_count += 1
        if manifest in manifests_by_role[source.role]:
            duplicate_manifest_count += 1
        scene_ids_by_role[source.role].add(scene_key)
        manifests_by_role[source.role].add(manifest)
        scene_ids_by_cell[(source.role, source.family)].add(scene_key)
        manifests_by_cell[(source.role, source.family)].add(manifest)
        binding_rows.append(
            [
                source.role,
                source.family,
                source.chunk,
                source.sequence,
                int(result["byte_count"]),
                str(result["_content_sha256"]),
            ]
        )
        family_result["source_count"] += 1
        for key in (
            "byte_count",
            "row_count",
            "stream_count",
            "primitive_transitions",
            "terminal_runs_without_endpoint",
            "packed_h6",
            "contributing_h6_stream_count",
        ):
            family_result[key] += int(result[key])
        if int(result["window_counts"].get("h6", 0)) > 0:
            family_result["contributing_h6_source_count"] += 1
        _merge_counter(family_result["window_counts"], result["window_counts"])
        _merge_counter(
            family_result["action_position_counts"],
            result["action_position_counts"],
        )
        for cell in result["action_position_presence"]:
            family_result["action_position_source_counts"][cell] += 1
        _merge_counter(
            family_result["maximal_path_histogram"],
            result["maximal_path_histogram"],
        )
        _merge_counter(
            family_result["packed_leftover_histogram"],
            result["packed_leftover_histogram"],
        )
        integrity.update(result["integrity"])

        shape_ok = all(
            (
                result["inventory_byte_count"] == result["scan_fstat_byte_count"],
                result["scan_fstat_byte_count"] == result["byte_count"],
                result["row_count"] == ROWS_PER_SOURCE,
                result["env_count"] == ENVS_PER_SOURCE,
                result["env_row_count_violation_count"] == 0,
                result["first_global_frame_index"] == 0,
                result["last_global_frame_index"] == ROWS_PER_SOURCE - 1,
                result["terminal_runs_without_endpoint"] == result["stream_count"],
                result["accepted_transitions_covered_by_paths"]
                == result["primitive_transitions"],
            )
        )
        if not shape_ok:
            family_result["source_shape_failure_count"] += 1

    failures: list[str] = []
    if len(results) != len(sources):
        failures.append("not_all_discovered_sources_scanned")
    if any(value for value in integrity.values()):
        failures.append("nonzero_temporal_or_row_integrity_count")
    if duplicate_scene_id_count:
        failures.append("duplicate_scene_identity_within_role")
    if duplicate_manifest_count:
        failures.append("duplicate_manifest_identity_within_role")
    cross_role_scene_id_overlap = len(
        scene_ids_by_role["train"] & scene_ids_by_role["val"]
    )
    cross_role_manifest_overlap = len(
        manifests_by_role["train"] & manifests_by_role["val"]
    )
    if cross_role_scene_id_overlap:
        failures.append("train_val_scene_identity_overlap")
    if cross_role_manifest_overlap:
        failures.append("train_val_manifest_identity_overlap")
    for role in ROLES:
        for family in FAMILIES:
            item = by_role_family[role][family]
            expected_sources = EXPECTED_SCENES[role][family]
            item["distinct_scene_identity_count"] = len(
                scene_ids_by_cell[(role, family)]
            )
            item["distinct_manifest_identity_count"] = len(
                manifests_by_cell[(role, family)]
            )
            if item["source_count"] != expected_sources:
                failures.append(f"source_count:{role}:{family}")
            if item["distinct_scene_identity_count"] != expected_sources:
                failures.append(f"scene_identity_count:{role}:{family}")
            if item["distinct_manifest_identity_count"] != expected_sources:
                failures.append(f"manifest_identity_count:{role}:{family}")
            if item["row_count"] != expected_sources * ROWS_PER_SOURCE:
                failures.append(f"row_count:{role}:{family}")
            if item["source_shape_failure_count"]:
                failures.append(f"source_shape:{role}:{family}")
            min_sources = 32 if role == "train" else 4
            min_packed = 1_024 if role == "train" else 128
            min_occurrences = 64 if role == "train" else 16
            min_action_sources = 8 if role == "train" else 3
            if item["contributing_h6_source_count"] < min_sources:
                failures.append(f"h6_source_breadth:{role}:{family}")
            if item["packed_h6"] < min_packed:
                failures.append(f"packed_h6:{role}:{family}")
            for position in range(2, 6):
                for primitive in PRIMITIVES:
                    cell = f"p{position}:{primitive}"
                    if item["action_position_counts"][cell] < min_occurrences:
                        failures.append(f"action_count:{role}:{family}:{cell}")
                    if (
                        item["action_position_source_counts"][cell]
                        < min_action_sources
                    ):
                        failures.append(f"action_source_breadth:{role}:{family}:{cell}")

    decision = (
        "MAIN_POOL_H4_METADATA_FEASIBLE"
        if not failures
        else "STOP_MAIN_POOL_H4_METADATA_INADEQUATE"
    )
    total_bytes = sum(int(result["byte_count"]) for result in results)
    total_rows = sum(int(result["row_count"]) for result in results)
    source_binding_sha256 = hashlib.sha256(
        json.dumps(
            binding_rows,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "decision": decision,
        "scope": {
            "roles": list(ROLES),
            "families": list(FAMILIES),
            "metadata_leaf": "frames.jsonl",
            "rgb_open_count": 0,
            "label_open_count": 0,
            "test_or_heldout_open_count": 0,
            "sealed_open_count": 0,
            "training_or_gpu_run_count": 0,
        },
        "definition": {
            "history_observations": 3,
            "history_transitions": 2,
            "future_transitions": 4,
            "total_transition_window": 6,
            "observation_endpoints": 7,
            "future_positions": ["p2", "p3", "p4", "p5"],
            "nominal_future_horizon_s": 2.0,
            "row_disjoint_packing": "each maximal path at offsets 0,6,12,...",
        },
        "totals": {
            "source_count": len(results),
            "byte_count": total_bytes,
            "row_count": total_rows,
            "primitive_transitions": sum(
                int(result["primitive_transitions"]) for result in results
            ),
            "sliding_h6": sum(
                int(result["window_counts"].get("h6", 0)) for result in results
            ),
            "packed_h6": sum(int(result["packed_h6"]) for result in results),
        },
        "identity": {
            "train_distinct_scene_identity_count": len(scene_ids_by_role["train"]),
            "val_distinct_scene_identity_count": len(scene_ids_by_role["val"]),
            "train_distinct_manifest_identity_count": len(manifests_by_role["train"]),
            "val_distinct_manifest_identity_count": len(manifests_by_role["val"]),
            "duplicate_scene_identity_within_role_count": duplicate_scene_id_count,
            "duplicate_manifest_identity_within_role_count": duplicate_manifest_count,
            "cross_role_scene_identity_overlap_count": cross_role_scene_id_overlap,
            "cross_role_manifest_identity_overlap_count": cross_role_manifest_overlap,
            "ordered_source_content_binding_sha256": source_binding_sha256,
            "raw_identifiers_persisted": False,
            "per_source_hashes_persisted": False,
        },
        "integrity": dict(sorted(integrity.items())),
        "access": {
            **dict(sorted(access.items())),
            "scan_frames_open_attempt_count": len(sources),
            "scan_frames_open_success_count": len(results),
            "scan_frames_byte_count": total_bytes,
            "scan_frames_row_count": total_rows,
        },
        "thresholds": {
            "train_min_h6_sources_per_family": 32,
            "val_min_h6_sources_per_family": 4,
            "train_min_packed_h6_per_family": 1_024,
            "val_min_packed_h6_per_family": 128,
            "train_min_action_position_occurrences": 64,
            "val_min_action_position_occurrences": 16,
            "train_min_action_position_sources": 8,
            "val_min_action_position_sources": 3,
            "integrity_counts_must_all_be_zero": True,
        },
        "by_role_family": by_role_family,
        "failed_predicates": failures,
        "authority": (
            "This receipt establishes metadata feasibility only. It does not "
            "authorize training, qualification, navigation, held-out access, "
            "benchmark opening, promotion, or deployment."
        ),
    }


def run_census(
    repo_root: Path,
    *,
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    sources, access = discover_sources(repo_root)
    results: list[dict[str, Any]] = []
    worker_count = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(scan_source, str(repo_root), source): source.ordinal
            for source in sources
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress is not None:
                progress(len(results), len(sources))
    results.sort(key=lambda item: int(item["source_ordinal"]))
    return aggregate_results(sources, results, access)


__all__ = [
    "CensusContractError",
    "EXPECTED_SCENES",
    "FAMILIES",
    "PRIMITIVES",
    "ROLES",
    "SCHEMA",
    "SourceRef",
    "aggregate_results",
    "discover_sources",
    "run_census",
    "scan_binary_stream",
    "scan_source",
]
