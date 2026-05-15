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

# Topics whose drops corrupt downstream training. Audited after parsing.
COMMAND_BLOCK_TOPIC = "/lewm/go2/command_block"
EXECUTED_COMMAND_BLOCK_TOPIC = "/lewm/go2/executed_command_block"
RESET_EVENT_TOPIC = "/lewm/go2/reset_event"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write summary.json with the contract audit but exit 0 even on gaps.",
    )
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

    command_block_sequence_ids: list[int] = []
    executed_command_sequence_ids: list[int] = []
    reset_counts: list[int] = []

    with messages_path.open("w", encoding="utf-8") as stream:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            msg_type = message_types.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            if topic == COMMAND_BLOCK_TOPIC:
                command_block_sequence_ids.append(int(getattr(msg, "sequence_id", -1)))
            elif topic == EXECUTED_COMMAND_BLOCK_TOPIC:
                executed_command_sequence_ids.append(int(getattr(msg, "sequence_id", -1)))
            elif topic == RESET_EVENT_TOPIC:
                reset_counts.append(int(getattr(msg, "reset_count", -1)))
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

    contract_audit = _audit_contract_topics(
        command_block_sequence_ids=command_block_sequence_ids,
        executed_command_sequence_ids=executed_command_sequence_ids,
        reset_counts=reset_counts,
    )

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
        "contract_audit": contract_audit,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")
    print(
        "contract_audit:"
        f" pass={contract_audit['pass']}"
        f" command_blocks={contract_audit['command_block_count']}"
        f" executed={contract_audit['executed_command_block_count']}"
        f" resets={contract_audit['reset_event_count']}"
    )
    if not contract_audit["pass"]:
        for issue in contract_audit["issues"]:
            print(f"  contract issue: {issue}")
        if not args.no_strict:
            raise SystemExit(2)


def _audit_contract_topics(
    *,
    command_block_sequence_ids: list[int],
    executed_command_sequence_ids: list[int],
    reset_counts: list[int],
) -> dict[str, Any]:
    """Detect dropped or duplicated contract messages.

    Rules:
      * Every requested ``command_block.sequence_id`` must reappear as an
        ``executed_command_block.sequence_id`` (no dropped executions).
      * No ``sequence_id`` may repeat on either topic (each block is unique).
      * ``reset_event.reset_count`` must be strictly increasing by one with no
        gaps (each reset must be observed).

    Orphan executions (executions without a matching command in this bag) are
    reported but do not fail the audit, since the bag may begin partway
    through a primitive.
    """

    issues: list[str] = []

    cmd_seen = list(command_block_sequence_ids)
    exec_seen = list(executed_command_sequence_ids)
    cmd_set = set(cmd_seen)
    exec_set = set(exec_seen)

    missing_executions = sorted(cmd_set - exec_set)
    if missing_executions:
        issues.append(
            f"command_block sequence_ids missing matching execution: {missing_executions}"
        )

    cmd_duplicates = sorted({s for s in cmd_seen if cmd_seen.count(s) > 1})
    if cmd_duplicates:
        issues.append(f"duplicate command_block sequence_ids: {cmd_duplicates}")

    exec_duplicates = sorted({s for s in exec_seen if exec_seen.count(s) > 1})
    if exec_duplicates:
        issues.append(f"duplicate executed_command_block sequence_ids: {exec_duplicates}")

    reset_gaps: list[dict[str, int]] = []
    for prev, curr in zip(reset_counts, reset_counts[1:]):
        if curr != prev + 1:
            reset_gaps.append({"prev": int(prev), "next": int(curr)})
    if reset_gaps:
        issues.append(f"reset_event.reset_count gaps: {reset_gaps}")

    orphan_executions = sorted(exec_set - cmd_set)

    return {
        "pass": not issues,
        "issues": issues,
        "command_block_count": len(cmd_seen),
        "executed_command_block_count": len(exec_seen),
        "reset_event_count": len(reset_counts),
        "missing_executions": missing_executions,
        "duplicate_command_sequence_ids": cmd_duplicates,
        "duplicate_executed_sequence_ids": exec_duplicates,
        "reset_gaps": reset_gaps,
        "orphan_executions": orphan_executions,
    }


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
