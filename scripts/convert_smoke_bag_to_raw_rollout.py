#!/usr/bin/env python3
"""Convert a smoke ROS bag into compact raw-rollout JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


COMPACT_TOPICS = {
    "/rgb_image",
    "/d455/image",
    "/d455/depth_image",
    "/d455/points",
    "/velodyne_points/points",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bag_dir = args.bag.resolve()
    if not bag_dir.is_dir():
        raise SystemExit(f"bag directory does not exist: {bag_dir}")
    out_dir = (args.out or bag_dir.with_name(f"{bag_dir.name}_raw_rollout")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_id = _storage_id(bag_dir)
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

    counts: Counter[str] = Counter()
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    messages_path = out_dir / "messages.jsonl"

    with messages_path.open("w", encoding="utf-8") as stream:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            msg_type = message_types.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            record = {
                "topic": topic,
                "type": topic_types[topic],
                "timestamp_ns": int(timestamp_ns),
                "payload": _compact_message(topic, msg),
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            counts[topic] += 1
            if first_timestamp_ns is None:
                first_timestamp_ns = int(timestamp_ns)
            last_timestamp_ns = int(timestamp_ns)

    summary = {
        "schema": "lewm_raw_rollout_smoke_v0",
        "source_bag": str(bag_dir),
        "storage_id": storage_id,
        "message_count": sum(counts.values()),
        "topic_counts": dict(sorted(counts.items())),
        "topic_types": dict(sorted(topic_types.items())),
        "first_timestamp_ns": first_timestamp_ns,
        "last_timestamp_ns": last_timestamp_ns,
        "duration_s": (
            None
            if first_timestamp_ns is None or last_timestamp_ns is None
            else (last_timestamp_ns - first_timestamp_ns) / 1e9
        ),
        "messages_jsonl": str(messages_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")


def _storage_id(bag_dir: Path) -> str:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return "mcap"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    return str(metadata.get("rosbag2_bagfile_information", {}).get("storage_identifier", "mcap"))


def _compact_message(topic: str, msg: Any) -> dict[str, Any]:
    if topic in COMPACT_TOPICS:
        return _compact_sensor_payload(msg)
    payload = message_to_ordereddict(msg)
    return _jsonable(payload)


def _compact_sensor_payload(msg: Any) -> dict[str, Any]:
    payload = message_to_ordereddict(msg)
    data = payload.pop("data", None)
    if data is not None:
        try:
            omitted = len(data)
        except TypeError:
            omitted = 0
        payload["data_omitted_count"] = omitted
    return _jsonable(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_omitted_count": len(value)}
    return value


if __name__ == "__main__":
    main()
