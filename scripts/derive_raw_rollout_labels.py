#!/usr/bin/env python3
"""Derive privileged per-step labels from a raw rollout.

This is the Phase A1 offline pass: rebuild the scene graph from the recorded
``topology_seed``/``family`` (or an explicit manifest), join it against logged
``/lewm/go2/base_state`` poses, and write labels without touching rollout
generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_worlds.labels.derived import (
    BASE_STATE_TOPIC,
    COMMAND_BLOCK_TOPIC,
    DEFAULT_YAW_BINS,
    EPISODE_INFO_TOPIC,
    EXECUTED_COMMAND_BLOCK_TOPIC,
    DerivedLabelComputer,
    DerivedLabelConfig,
    label_to_jsonable,
    pose_steps_from_message_records,
    pose_steps_from_messages_jsonl,
)
from lewm_worlds.labels.topology import local_graph_type_histogram
from lewm_worlds.manifest import (
    SceneManifest,
    build_scene_manifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)


ENV_TOPIC_RE = re.compile(r"^/env_(\d+)(/.*)$")
LABEL_SOURCE_TOPICS = {
    BASE_STATE_TOPIC,
    COMMAND_BLOCK_TOPIC,
    EXECUTED_COMMAND_BLOCK_TOPIC,
    EPISODE_INFO_TOPIC,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_rollout",
        type=Path,
        help="Raw-rollout directory, compact messages.jsonl, or rosbag2 MCAP directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <raw_rollout>/derived_labels).",
    )
    parser.add_argument("--scene-manifest", type=Path, default=None)
    parser.add_argument("--scene-corpus", type=Path, default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--topology-seed", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--difficulty-tier", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--yaw-bins", type=int, default=DEFAULT_YAW_BINS)
    parser.add_argument("--block-size-ticks", type=int, default=5)
    parser.add_argument("--history-window-ticks", type=int, default=16)
    args = parser.parse_args()

    source_path = args.raw_rollout.resolve()
    summary = _load_sidecar_summary(source_path)
    manifest = _resolve_manifest(args, summary)
    poses, join_summary, source_kind, source_detail = _load_pose_steps(source_path)
    if not poses:
        raise SystemExit(f"no {BASE_STATE_TOPIC} poses found in {source_path}")

    config = DerivedLabelConfig(
        yaw_bins=int(args.yaw_bins),
        block_size_ticks=int(args.block_size_ticks),
        history_window_ticks=int(args.history_window_ticks),
    )
    computer = DerivedLabelComputer(manifest, config=config)

    out_dir = _default_out_dir(source_path) if args.out is None else args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    label_count = 0
    local_graph_counts: dict[str, int] = {}
    with labels_path.open("w", encoding="utf-8") as stream:
        for label in computer.stream(poses):
            payload = label_to_jsonable(label)
            local_graph_counts[payload["local_graph_type"]] = (
                local_graph_counts.get(payload["local_graph_type"], 0) + 1
            )
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            label_count += 1

    summary_payload = {
        "schema": "lewm_derived_labels_v0",
        "source": {
            "path": str(source_path),
            "kind": source_kind,
            "detail": source_detail,
        },
        "scene": {
            "scene_id": manifest.scene_id,
            "family": manifest.family,
            "split": manifest.split,
            "difficulty_tier": manifest.difficulty_tier,
            "topology_seed": manifest.topology_seed,
            "visual_seed": manifest.visual_seed,
            "physics_seed": manifest.physics_seed,
            "manifest_sha256": manifest_sha256(manifest),
            "local_graph_type_histogram": local_graph_type_histogram(manifest),
        },
        "config": asdict(config),
        "pose_join": asdict(join_summary),
        "label_count": label_count,
        "local_graph_type_counts": dict(sorted(local_graph_counts.items())),
        "labels_jsonl": str(labels_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"labels={labels_path}")
    print(f"summary={summary_path}")
    print(
        "derived:"
        f" labels={label_count}"
        f" scene={manifest.scene_id}"
        f" envs={list(join_summary.env_indices)}"
        f" missing_commands={join_summary.missing_command_count}"
    )
    return 0


def _load_pose_steps(source_path: Path):
    messages_path = _messages_jsonl_path(source_path)
    if messages_path is not None:
        poses, summary = pose_steps_from_messages_jsonl(messages_path)
        return poses, summary, "messages_jsonl", str(messages_path)

    if source_path.is_dir():
        records = list(_records_from_mcap_bag(source_path))
        poses, summary = pose_steps_from_message_records(records)
        return poses, summary, "rosbag2_mcap", str(source_path)

    raise SystemExit(
        f"{source_path} is neither a messages.jsonl file nor a raw-rollout/MCAP directory"
    )


def _messages_jsonl_path(source_path: Path) -> Path | None:
    if source_path.is_file() and source_path.name.endswith(".jsonl"):
        return source_path
    candidate = source_path / "messages.jsonl"
    if source_path.is_dir() and candidate.is_file():
        return candidate
    return None


def _default_out_dir(source_path: Path) -> Path:
    if source_path.is_file():
        return source_path.parent / "derived_labels"
    return source_path / "derived_labels"


def _load_sidecar_summary(source_path: Path) -> dict[str, Any]:
    candidates = []
    if source_path.is_dir():
        candidates.append(source_path / "summary.json")
    else:
        candidates.append(source_path.parent / "summary.json")
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _resolve_manifest(args: argparse.Namespace, summary: dict[str, Any]) -> SceneManifest:
    if args.scene_manifest is not None:
        return _load_manifest(args.scene_manifest.resolve())

    family = args.family or summary.get("family") or summary.get("scene_family")
    split = args.split if args.split is not None else summary.get("split")
    difficulty_tier = (
        args.difficulty_tier
        if args.difficulty_tier is not None
        else summary.get("difficulty_tier")
    )
    topology_seed = (
        int(args.topology_seed)
        if args.topology_seed is not None
        else _optional_int(summary.get("topology_seed"))
    )

    if family is not None and topology_seed is not None:
        return build_scene_manifest(
            scene_seed=topology_seed,
            family=str(family),
            split=None if split in ("", None) else str(split),
            difficulty_tier=None if difficulty_tier in ("", None) else str(difficulty_tier),
        )

    if args.scene_corpus is not None:
        scene_id = args.scene_id or summary.get("scene_id")
        manifest = _find_manifest_in_corpus(
            args.scene_corpus.resolve(),
            scene_id=None if scene_id in ("", None) else str(scene_id),
            family=None if family in ("", None) else str(family),
            topology_seed=topology_seed,
        )
        if manifest is not None:
            return manifest

    raise SystemExit(
        "could not resolve scene manifest; provide --scene-manifest, "
        "--family + --topology-seed, a rollout summary.json with those fields, "
        "or --scene-corpus with --scene-id"
    )


def _load_manifest(path: Path) -> SceneManifest:
    return parse_scene_manifest_dict(json.loads(path.read_text(encoding="utf-8")))


def _find_manifest_in_corpus(
    corpus: Path,
    *,
    scene_id: str | None,
    family: str | None,
    topology_seed: int | None,
) -> SceneManifest | None:
    if not corpus.is_dir():
        raise SystemExit(f"scene corpus does not exist: {corpus}")
    for manifest_path in sorted(corpus.rglob("manifest.json")):
        manifest = _load_manifest(manifest_path)
        if scene_id is not None and manifest.scene_id == scene_id:
            return manifest
        if (
            scene_id is None
            and family is not None
            and topology_seed is not None
            and manifest.family == family
            and manifest.topology_seed == topology_seed
        ):
            return manifest
    return None


def _records_from_mcap_bag(bag_dir: Path) -> Iterable[dict[str, Any]]:
    try:
        import yaml
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - depends on sourced ROS env
        raise SystemExit(
            "reading MCAP input requires a sourced ROS 2 Jazzy overlay with "
            "rosbag2_py, rclpy, and rosidl_runtime_py"
        ) from exc

    storage_id = "mcap"
    metadata_path = bag_dir / "metadata.yaml"
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        storage_id = str(
            metadata.get("rosbag2_bagfile_information", {}).get(
                "storage_identifier", "mcap"
            )
        )

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_types = {topic: get_message(type_name) for topic, type_name in topic_types.items()}

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        canonical_topic = _canonical_topic(topic)
        if canonical_topic not in LABEL_SOURCE_TOPICS:
            continue
        msg_type = message_types.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(data, msg_type)
        record = {
            "topic": topic,
            "canonical_topic": canonical_topic,
            "type": topic_types[topic],
            "timestamp_ns": int(timestamp_ns),
            "payload": _jsonable(message_to_ordereddict(msg)),
        }
        env_index = _env_index(topic)
        if env_index is not None:
            record["env_index"] = env_index
        yield record


def _canonical_topic(topic: str) -> str:
    match = ENV_TOPIC_RE.match(topic)
    if match:
        topic = match.group(2)
    return topic


def _env_index(topic: str) -> int | None:
    match = ENV_TOPIC_RE.match(topic)
    if match:
        return int(match.group(1))
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_omitted_count": len(value)}
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
