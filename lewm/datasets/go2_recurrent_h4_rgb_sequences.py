"""Deterministic reset-safe H6-to-RGB index for datagen_full train/val.

Only exact train/validation ``frames.jsonl`` and their literal replay-plan,
render-summary, and selected RGB leaves are opened.  The flat render root is
never listed and no path from a metadata row is followed.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (
    BLOCK_SIZE,
    COMMAND_DT_S,
    ENVS_PER_SOURCE,
    EXPECTED_SCENES,
    FAMILIES,
    PRIMITIVES,
    ROWS_PER_SOURCE,
    ROLES,
    STEPS_PER_STREAM,
    SourceRef,
    TIME_TOLERANCE_S,
    _open_absolute_directory,
    _strict_json_loads,
    discover_sources,
)


SCHEMA = "lewm_go2_recurrent_h4_rgb_sequence_index_v1"
SEED = "go2_recurrent_h4_rgb_sequence_index_v1_20260727"
TRAIN_PER_FAMILY = 2_000
VAL_PER_FAMILY = 256
PER_SCENE_CANDIDATE_CAP = 64
ACTION_TO_INDEX = {name: index for index, name in enumerate(PRIMITIVES)}
_SCENE_RE = re.compile(r"^(?:" + "|".join(map(re.escape, FAMILIES)) + r")_[0-9a-f]{12}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class SequenceContractError(RuntimeError):
    """The exact metadata, temporal, or RGB binding contract failed."""


@dataclass(frozen=True)
class Endpoint:
    frame_index: int
    env_index: int
    episode_step: int
    timestamp_ns: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return self.env_index, self.episode_step, self.timestamp_ns, self.frame_index


@dataclass(frozen=True)
class Transition:
    current: Endpoint
    next: Endpoint
    primitive: str


@dataclass(frozen=True)
class H6Window:
    rank: str
    role: str
    family: str
    scene_id: str
    endpoints: tuple[Endpoint, ...]
    actions: tuple[int, ...]

    def to_row(self) -> dict[str, Any]:
        rgb = [
            (
                f"{self.scene_id}/rgb/"
                f"frame_{endpoint.frame_index:06d}_env_{endpoint.env_index:02d}.png"
            )
            for endpoint in self.endpoints
        ]
        return {
            "schema": SCHEMA,
            "role": self.role,
            "family": self.family,
            "scene_id": self.scene_id,
            "rgb": rgb,
            "actions": list(self.actions),
        }


@dataclass
class _Run:
    signature: tuple[str, str]
    primitive: str
    block_size: int
    command_dt_s: float
    rows: list[Endpoint]


@dataclass
class _Stream:
    last: Endpoint
    run: _Run
    packed: list[Transition] = field(default_factory=list)


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SequenceContractError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _source_directory(repo_root: Path, source: SourceRef) -> Path:
    _validate_source_ref(source)
    return (
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


def _validate_source_ref(source: SourceRef) -> None:
    if source.role not in ROLES or source.family not in FAMILIES:
        raise SequenceContractError("source role or family left the allowlist")
    valid_chunk = bool(re.fullmatch(r"chunk_[0-9]{4}", source.chunk)) or (
        source.family == "rough_local_dynamics" and source.chunk == "chunk_backfill"
    )
    sequence_pattern = re.compile(
        rf"^[0-9]{{6}}_{re.escape(source.family)}_[0-9a-f]{{12}}$"
    )
    if not valid_chunk or not sequence_pattern.fullmatch(source.sequence):
        raise SequenceContractError("source chunk or sequence left the allowlist")
    if source.byte_count <= 0 or source.ordinal <= 0:
        raise SequenceContractError("source inventory binding is invalid")


def _validate_window(window: H6Window) -> None:
    if window.role not in ROLES or window.family not in FAMILIES:
        raise SequenceContractError("window role or family left the allowlist")
    if (
        not _SCENE_RE.fullmatch(window.scene_id)
        or not window.scene_id.startswith(f"{window.family}_")
    ):
        raise SequenceContractError("window scene identity is invalid")
    if len(window.endpoints) != 7 or len(window.actions) != 6:
        raise SequenceContractError("window must bind seven RGB endpoints and six actions")
    if any(
        not isinstance(action, int)
        or isinstance(action, bool)
        or not 0 <= action < len(PRIMITIVES)
        for action in window.actions
    ):
        raise SequenceContractError("window action index is invalid")
    for endpoint in window.endpoints:
        if (
            not 0 <= endpoint.frame_index < ROWS_PER_SOURCE
            or not 0 <= endpoint.env_index < ENVS_PER_SOURCE
            or endpoint.frame_index % ENVS_PER_SOURCE != endpoint.env_index
            or endpoint.episode_step < 1
            or endpoint.timestamp_ns < 0
        ):
            raise SequenceContractError("window endpoint is invalid")


def _open_literal_child_directory(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SequenceContractError("non-canonical directory component")
    return os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_fd,
    )


def _load_source_identity(repo_root: Path, source: SourceRef) -> tuple[str, str]:
    source_directory = _source_directory(repo_root, source)
    source_fd = _open_absolute_directory(source_directory)
    try:
        plan_raw = _read_regular_at(source_fd, "render_replay_plan.json")
    finally:
        os.close(source_fd)
    plan = _strict_json_loads(plan_raw)
    if not isinstance(plan, dict):
        raise SequenceContractError("render plan is not an object")
    expected_scene = source.sequence.split("_", 1)[1]
    scene_id = plan.get("scene_id")
    manifest = plan.get("manifest_sha256")
    if (
        plan.get("schema") != "lewm_render_replay_plan_v0"
        or plan.get("split") != source.role
        or plan.get("scene_family") != source.family
        or plan.get("frame_count") != ROWS_PER_SOURCE
        or plan.get("source_env_count") != ENVS_PER_SOURCE
        or scene_id != expected_scene
        or not isinstance(scene_id, str)
        or not _SCENE_RE.fullmatch(scene_id)
        or not isinstance(manifest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest)
        or plan.get("frames_jsonl") != str(source_directory / "frames.jsonl")
    ):
        raise SequenceContractError("render plan identity changed")

    render_scene = (
        repo_root
        / ".generated"
        / "datagen_full"
        / "render_textured_v03"
        / scene_id
    )
    render_fd = _open_absolute_directory(render_scene)
    try:
        summary_raw = _read_regular_at(render_fd, "summary.json")
    finally:
        os.close(render_fd)
    summary = _strict_json_loads(summary_raw)
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != "lewm_rendered_vision_v03"
        or summary.get("render_status") != "complete"
        or summary.get("scene_id") != scene_id
        or summary.get("split") != source.role
        or summary.get("family") != source.family
        or summary.get("frame_count") != ROWS_PER_SOURCE
        or summary.get("resolution") != 224
        or summary.get("visuals") != "textured_v03"
        or summary.get("textures_enabled") is not True
        or summary.get("plan") != str(source_directory / "render_replay_plan.json")
    ):
        raise SequenceContractError("render summary identity changed")
    return scene_id, manifest


def _parse_frame(
    payload: Mapping[str, Any], *, source: SourceRef, manifest: str
) -> tuple[tuple[int, str, int], Endpoint, tuple[str, str], str, int, float]:
    episode = payload.get("episode")
    context = payload.get("command_context")
    if not isinstance(episode, Mapping) or not isinstance(context, Mapping):
        raise SequenceContractError("frame lacks episode or command context")
    integers = (
        payload.get("frame_index"),
        payload.get("env_index"),
        payload.get("timestamp_ns"),
        episode.get("episode_step"),
        episode.get("reset_count"),
    )
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in integers):
        raise SequenceContractError("frame integer field changed")
    frame_index, env_index, timestamp_ns, step, reset_count = map(int, integers)
    if (
        not 0 <= frame_index < ROWS_PER_SOURCE
        or not 0 <= env_index < ENVS_PER_SOURCE
        or frame_index % ENVS_PER_SOURCE != env_index
        or step < 1
        or reset_count < 0
        or episode.get("split") != source.role
        or episode.get("manifest_sha256") != manifest
    ):
        raise SequenceContractError("frame identity or role changed")
    episode_id = episode.get("episode_id")
    sequence_id = context.get("sequence_id")
    primitive = context.get("primitive_name")
    block_size = context.get("block_size")
    command_dt_s = context.get("command_dt_s")
    if (
        not isinstance(episode_id, (str, int))
        or isinstance(episode_id, bool)
        or not isinstance(sequence_id, (str, int))
        or isinstance(sequence_id, bool)
        or primitive not in ACTION_TO_INDEX
        or block_size != BLOCK_SIZE
        or not isinstance(command_dt_s, (int, float))
        or isinstance(command_dt_s, bool)
        or not math.isfinite(float(command_dt_s))
        or abs(float(command_dt_s) - COMMAND_DT_S) > TIME_TOLERANCE_S
    ):
        raise SequenceContractError("primitive context changed")
    endpoint = Endpoint(frame_index, env_index, step, timestamp_ns)
    key = env_index, str(episode_id), reset_count
    return (
        key,
        endpoint,
        (str(sequence_id), str(primitive)),
        str(primitive),
        int(block_size),
        float(command_dt_s),
    )


def _validate_run(run: _Run, endpoint: Endpoint | None) -> bool:
    if len(run.rows) != BLOCK_SIZE:
        raise SequenceContractError("primitive run is not one complete block")
    if endpoint is None:
        return False
    window = [*run.rows, endpoint]
    if any(right.episode_step != left.episode_step + 1 for left, right in zip(window, window[1:])):
        raise SequenceContractError("primitive run crosses an episode-step gap")
    deltas = [
        (right.timestamp_ns - left.timestamp_ns) * 1e-9
        for left, right in zip(window, window[1:])
    ]
    if any(abs(delta - run.command_dt_s) > TIME_TOLERANCE_S for delta in deltas):
        raise SequenceContractError("primitive run has irregular timing")
    duration = (endpoint.timestamp_ns - run.rows[0].timestamp_ns) * 1e-9
    if abs(duration - 0.5) > TIME_TOLERANCE_S:
        raise SequenceContractError("primitive run is not half a second")
    return True


def _window_rank(
    source: SourceRef, stream_key: tuple[int, str, int], first: Endpoint
) -> str:
    raw = json.dumps(
        [
            SEED,
            source.role,
            source.family,
            source.ordinal,
            stream_key[0],
            stream_key[1],
            stream_key[2],
            first.frame_index,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def scan_source_candidates(
    repo_root: str, source: SourceRef, per_scene_cap: int
) -> dict[str, Any]:
    root = Path(repo_root)
    scene_id, manifest = _load_source_identity(root, source)
    source_fd = _open_absolute_directory(_source_directory(root, source))
    try:
        frames_fd = os.open("frames.jsonl", _FILE_FLAGS, dir_fd=source_fd)
    finally:
        os.close(source_fd)
    streams: dict[tuple[int, str, int], _Stream] = {}
    env_rows: Counter[int] = Counter()
    candidates: list[H6Window] = []
    row_count = 0
    last_frame_index = -1
    final_frames_info: os.stat_result | None = None
    initial_frames_info = os.fstat(frames_fd)
    if (
        not stat.S_ISREG(initial_frames_info.st_mode)
        or int(initial_frames_info.st_size) != source.byte_count
    ):
        os.close(frames_fd)
        raise SequenceContractError("frames leaf changed after source discovery")

    def emit_transition(
        key: tuple[int, str, int], state: _Stream, transition: Transition
    ) -> None:
        if state.packed and state.packed[-1].next.identity != transition.current.identity:
            state.packed.clear()
        state.packed.append(transition)
        if len(state.packed) != 6:
            return
        endpoints = (state.packed[0].current,) + tuple(
            item.next for item in state.packed
        )
        actions = tuple(ACTION_TO_INDEX[item.primitive] for item in state.packed)
        candidates.append(
            H6Window(
                rank=_window_rank(source, key, endpoints[0]),
                role=source.role,
                family=source.family,
                scene_id=scene_id,
                endpoints=endpoints,
                actions=actions,
            )
        )
        state.packed.clear()

    try:
        with os.fdopen(frames_fd, "rb", closefd=True) as stream:
            for raw_line in stream:
                payload = _strict_json_loads(raw_line)
                if not isinstance(payload, dict):
                    raise SequenceContractError("frame row is not an object")
                key, endpoint, signature, primitive, block_size, command_dt_s = _parse_frame(
                    payload, source=source, manifest=manifest
                )
                row_count += 1
                env_rows[endpoint.env_index] += 1
                if endpoint.frame_index != last_frame_index + 1:
                    raise SequenceContractError("global frame index is discontinuous")
                last_frame_index = endpoint.frame_index
                state = streams.get(key)
                if state is None:
                    if endpoint.episode_step != 1:
                        raise SequenceContractError("stream does not start at step one")
                    streams[key] = _Stream(
                        last=endpoint,
                        run=_Run(signature, primitive, block_size, command_dt_s, [endpoint]),
                    )
                    continue
                if (
                    endpoint.episode_step != state.last.episode_step + 1
                    or endpoint.timestamp_ns <= state.last.timestamp_ns
                ):
                    raise SequenceContractError("stream continuity changed")
                if signature == state.run.signature:
                    if (
                        primitive != state.run.primitive
                        or block_size != state.run.block_size
                        or command_dt_s != state.run.command_dt_s
                    ):
                        raise SequenceContractError("run context drifted")
                    state.run.rows.append(endpoint)
                else:
                    if _validate_run(state.run, endpoint):
                        emit_transition(
                            key,
                            state,
                            Transition(state.run.rows[0], endpoint, state.run.primitive),
                        )
                    state.run = _Run(
                        signature, primitive, block_size, command_dt_s, [endpoint]
                    )
                state.last = endpoint
            final_frames_info = os.fstat(stream.fileno())
    except BaseException:
        try:
            os.close(frames_fd)
        except OSError:
            pass
        raise

    for state in streams.values():
        _validate_run(state.run, None)
        state.packed.clear()
    if (
        final_frames_info is None
        or not stat.S_ISREG(final_frames_info.st_mode)
        or int(final_frames_info.st_size) != source.byte_count
        or final_frames_info.st_dev != initial_frames_info.st_dev
        or final_frames_info.st_ino != initial_frames_info.st_ino
    ):
        raise SequenceContractError("frames leaf changed during source scan")
    post_source_fd = _open_absolute_directory(_source_directory(root, source))
    try:
        post_frames_fd = os.open("frames.jsonl", _FILE_FLAGS, dir_fd=post_source_fd)
        try:
            post_frames_info = os.fstat(post_frames_fd)
        finally:
            os.close(post_frames_fd)
    finally:
        os.close(post_source_fd)
    if (
        not stat.S_ISREG(post_frames_info.st_mode)
        or int(post_frames_info.st_size) != source.byte_count
        or post_frames_info.st_dev != initial_frames_info.st_dev
        or post_frames_info.st_ino != initial_frames_info.st_ino
    ):
        raise SequenceContractError("frames path was replaced during source scan")
    if (
        row_count != ROWS_PER_SOURCE
        or last_frame_index != ROWS_PER_SOURCE - 1
        or set(env_rows) != set(range(ENVS_PER_SOURCE))
        or any(count != STEPS_PER_STREAM for count in env_rows.values())
    ):
        raise SequenceContractError("source row or environment shape changed")
    candidates.sort(key=lambda item: item.rank)
    return {
        "source_ordinal": source.ordinal,
        "role": source.role,
        "family": source.family,
        "scene_id": scene_id,
        "manifest_sha256": manifest,
        "candidate_count": len(candidates),
        "candidates": candidates[:per_scene_cap],
    }


def _select_role_family(
    records: Sequence[dict[str, Any]], *, target: int
) -> list[H6Window]:
    ordered = sorted(
        records,
        key=lambda item: hashlib.sha256(
            f"{SEED}|scene|{item['scene_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[H6Window] = []
    depth = 0
    while len(selected) < target:
        added = False
        for record in ordered:
            candidates: list[H6Window] = record["candidates"]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != target:
        raise SequenceContractError(
            f"only {len(selected)} candidates available for target {target}"
        )
    return selected


def _interleave_families(by_family: Mapping[str, Sequence[H6Window]]) -> list[H6Window]:
    lengths = {len(values) for values in by_family.values()}
    if len(lengths) != 1 or set(by_family) != set(FAMILIES):
        raise SequenceContractError("family schedule is not rectangular")
    family_order = sorted(
        FAMILIES,
        key=lambda family: hashlib.sha256(
            f"{SEED}|family|{family}".encode("utf-8")
        ).hexdigest(),
    )
    count = next(iter(lengths))
    return [
        by_family[family][index]
        for index in range(count)
        for family in family_order
    ]


def _coverage(windows: Sequence[H6Window]) -> dict[str, Any]:
    action_cells: Counter[str] = Counter()
    scene_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    for window in windows:
        scene_ids.add(window.scene_id)
        family_counts[window.family] += 1
        for position in range(2, 6):
            action_cells[f"{window.family}:p{position}:{PRIMITIVES[window.actions[position]]}"] += 1
    missing = [
        f"{family}:p{position}:{primitive}"
        for family in FAMILIES
        for position in range(2, 6)
        for primitive in PRIMITIVES
        if action_cells[f"{family}:p{position}:{primitive}"] == 0
    ]
    return {
        "row_count": len(windows),
        "scene_count": len(scene_ids),
        "family_counts": dict(sorted(family_counts.items())),
        "action_position_counts": dict(sorted(action_cells.items())),
        "missing_action_position_cells": missing,
    }


def validate_selected_rgb(
    repo_root: Path, windows: Sequence[H6Window]
) -> dict[str, int]:
    render_root = repo_root / ".generated" / "datagen_full" / "render_textured_v03"
    render_fd = _open_absolute_directory(render_root)
    paths_by_scene: dict[str, set[str]] = {}
    for window in windows:
        _validate_window(window)
        bucket = paths_by_scene.setdefault(window.scene_id, set())
        for endpoint in window.endpoints:
            bucket.add(
                f"frame_{endpoint.frame_index:06d}_env_{endpoint.env_index:02d}.png"
            )
    open_count = 0
    byte_count = 0
    try:
        for scene_id in sorted(paths_by_scene):
            scene_fd = _open_literal_child_directory(render_fd, scene_id)
            try:
                rgb_fd = _open_literal_child_directory(scene_fd, "rgb")
                try:
                    for filename in sorted(paths_by_scene[scene_id]):
                        descriptor = os.open(filename, _FILE_FLAGS, dir_fd=rgb_fd)
                        try:
                            info = os.fstat(descriptor)
                            if not stat.S_ISREG(info.st_mode):
                                raise SequenceContractError("scheduled RGB is not regular")
                            if os.read(descriptor, 8) != _PNG_SIGNATURE:
                                raise SequenceContractError("scheduled RGB is not PNG")
                            open_count += 1
                            byte_count += int(info.st_size)
                        finally:
                            os.close(descriptor)
                finally:
                    os.close(rgb_fd)
            finally:
                os.close(scene_fd)
    finally:
        os.close(render_fd)
    return {
        "unique_rgb_count": open_count,
        "unique_rgb_byte_count": byte_count,
    }


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
                family_records, target=target
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
            "method": "per_scene_hash_rank_then_scene_round_robin_then_family_interleave",
            "row_disjoint_h6": True,
            "per_scene_candidate_cap": PER_SCENE_CANDIDATE_CAP,
            "train_per_family": TRAIN_PER_FAMILY,
            "val_per_family": VAL_PER_FAMILY,
            "train_presentation_cap": 16_000,
        },
        "sequence_schema": {
            "action_vocabulary": list(PRIMITIVES),
            "rgb_count": 7,
            "action_count": 6,
            "alignment": "actions[i] joins rgb[i] to rgb[i+1]",
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
            "candidate_count": sum(int(result["candidate_count"]) for result in results),
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
