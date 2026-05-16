#!/usr/bin/env python3
"""Convert a smoke ROS bag into compact raw-rollout JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
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
ENV_TOPIC_RE = re.compile(r"^/env_(\d+)(/.*)$")


@dataclass(frozen=True)
class TopicQualityRule:
    min_count: int = 1
    min_rate_hz: float | None = None
    max_gap_s: float | None = None
    time_basis: str = "source"


RAW_TOPIC_RULES: dict[str, TopicQualityRule] = {
    "/clock": TopicQualityRule(min_rate_hz=500.0, max_gap_s=0.05),
    "/tf": TopicQualityRule(min_rate_hz=100.0, max_gap_s=0.5, time_basis="record"),
    "/joint_states": TopicQualityRule(min_rate_hz=100.0, max_gap_s=0.05),
    "/imu/data": TopicQualityRule(min_rate_hz=50.0, max_gap_s=0.05),
    "/odom": TopicQualityRule(min_rate_hz=20.0, max_gap_s=0.15),
    "/gazebo/odom": TopicQualityRule(min_rate_hz=20.0, max_gap_s=0.15),
    "/lewm/go2/base_state": TopicQualityRule(min_rate_hz=20.0, max_gap_s=0.15),
    "/lewm/go2/foot_contacts": TopicQualityRule(min_rate_hz=100.0, max_gap_s=0.05),
    "/lewm/go2/mode": TopicQualityRule(min_rate_hz=0.5, max_gap_s=2.0),
    "/lewm/episode_info": TopicQualityRule(min_rate_hz=0.2, max_gap_s=5.0),
    "/cmd_vel": TopicQualityRule(min_count=1, time_basis="record"),
}
GENESIS_RAW_TOPIC_RULES: dict[str, TopicQualityRule] = {
    "/clock": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    COMMAND_BLOCK_TOPIC: TopicQualityRule(min_count=1, time_basis="record"),
    EXECUTED_COMMAND_BLOCK_TOPIC: TopicQualityRule(min_count=1, time_basis="record"),
    RESET_EVENT_TOPIC: TopicQualityRule(min_count=1, time_basis="record"),
    "/joint_states": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/imu/data": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/odom": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/lewm/go2/base_state": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/lewm/go2/foot_contacts": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/lewm/go2/mode": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
    "/lewm/episode_info": TopicQualityRule(min_rate_hz=9.0, max_gap_s=0.15),
}
VISION_TOPIC_RULES: dict[str, TopicQualityRule] = {
    **RAW_TOPIC_RULES,
    "/rgb_image": TopicQualityRule(min_rate_hz=5.0, max_gap_s=0.3),
    "/lewm/go2/camera_info": TopicQualityRule(min_rate_hz=1.0, max_gap_s=1.0),
}


@dataclass
class SeriesTiming:
    count: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    max_gap_ns: int | None = None
    max_gap_start_ns: int | None = None
    max_gap_end_ns: int | None = None
    regressions: int = 0

    def update(self, timestamp_ns: int) -> None:
        timestamp_ns = int(timestamp_ns)
        if self.first_ns is None:
            self.first_ns = timestamp_ns
        if self.last_ns is not None:
            gap_ns = timestamp_ns - self.last_ns
            if gap_ns < 0:
                self.regressions += 1
            elif self.max_gap_ns is None or gap_ns > self.max_gap_ns:
                self.max_gap_ns = gap_ns
                self.max_gap_start_ns = self.last_ns
                self.max_gap_end_ns = timestamp_ns
        self.last_ns = timestamp_ns
        self.count += 1

    def as_dict(self) -> dict[str, Any]:
        duration_s = (
            None
            if self.first_ns is None or self.last_ns is None
            else max(0.0, (self.last_ns - self.first_ns) / 1e9)
        )
        rate_hz = (
            None
            if duration_s is None or duration_s <= 0.0 or self.count < 2
            else (self.count - 1) / duration_s
        )
        return {
            "count": self.count,
            "first_timestamp_ns": self.first_ns,
            "last_timestamp_ns": self.last_ns,
            "duration_s": duration_s,
            "rate_hz": rate_hz,
            "max_gap_s": None if self.max_gap_ns is None else self.max_gap_ns / 1e9,
            "max_gap_start_ns": self.max_gap_start_ns,
            "max_gap_end_ns": self.max_gap_end_ns,
            "timestamp_regressions": self.regressions,
        }


@dataclass
class TopicTiming:
    record: SeriesTiming
    source: SeriesTiming
    source_missing_count: int = 0

    @classmethod
    def empty(cls) -> "TopicTiming":
        return cls(record=SeriesTiming(), source=SeriesTiming())

    def as_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.as_dict(),
            "source": self.source.as_dict(),
            "source_missing_count": self.source_missing_count,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write summary.json with audits but exit 0 even on failing gates.",
    )
    parser.add_argument(
        "--quality-profile",
        choices=("smoke", "pilot", "training", "raw_pilot", "raw_training"),
        default="smoke",
        help=(
            "Data-quality policy to enforce. smoke fails only contract-critical "
            "loss; pilot/training also fail critical stream gaps and rate loss. "
            "raw_* profiles omit RGB/camera-info requirements for GPU render replay."
        ),
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
    topic_timing: dict[str, TopicTiming] = {}
    canonical_counts: Counter[str] = Counter()
    canonical_topic_timing: dict[str, TopicTiming] = {}

    with messages_path.open("w", encoding="utf-8") as stream:
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            msg_type = message_types.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            canonical_topic = _canonical_topic(topic)
            env_index = _env_index(topic)
            if canonical_topic == COMMAND_BLOCK_TOPIC:
                command_block_sequence_ids.append(int(getattr(msg, "sequence_id", -1)))
            elif canonical_topic == EXECUTED_COMMAND_BLOCK_TOPIC:
                executed_command_sequence_ids.append(int(getattr(msg, "sequence_id", -1)))
            elif canonical_topic == RESET_EVENT_TOPIC:
                reset_counts.append(int(getattr(msg, "reset_count", -1)))
            timing = topic_timing.setdefault(topic, TopicTiming.empty())
            timing.record.update(int(timestamp_ns))
            canonical_timing = canonical_topic_timing.setdefault(
                canonical_topic, TopicTiming.empty()
            )
            canonical_timing.record.update(int(timestamp_ns))
            source_timestamp_ns = _source_timestamp_ns(canonical_topic, msg)
            if source_timestamp_ns is None:
                timing.source_missing_count += 1
                canonical_timing.source_missing_count += 1
            else:
                timing.source.update(source_timestamp_ns)
                canonical_timing.source.update(source_timestamp_ns)
            record = {
                "topic": topic,
                "canonical_topic": canonical_topic,
                "type": topic_types[topic],
                "timestamp_ns": int(timestamp_ns),
                "payload": _compact_message(canonical_topic, msg),
            }
            if env_index is not None:
                record["env_index"] = env_index
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            counts[topic] += 1
            canonical_counts[canonical_topic] += 1
            if first_timestamp_ns is None:
                first_timestamp_ns = int(timestamp_ns)
            last_timestamp_ns = int(timestamp_ns)

    contract_audit = _audit_contract_topics(
        command_block_sequence_ids=command_block_sequence_ids,
        executed_command_sequence_ids=executed_command_sequence_ids,
        reset_counts=reset_counts,
    )
    topic_audit = {
        "topics": {
            topic: timing.as_dict() for topic, timing in sorted(topic_timing.items())
        },
        "canonical_topics": {
            topic: timing.as_dict()
            for topic, timing in sorted(canonical_topic_timing.items())
        },
        "rules": {
            topic: {
                "min_count": rule.min_count,
                "min_rate_hz": rule.min_rate_hz,
                "max_gap_s": rule.max_gap_s,
                "time_basis": rule.time_basis,
            }
            for topic, rule in sorted(_topic_rules_for_profile(args.quality_profile).items())
        },
    }
    data_quality_audit = _audit_data_quality(
        profile=args.quality_profile,
        contract_audit=contract_audit,
        topic_audit=topic_audit,
    )

    summary = {
        "schema": "lewm_raw_rollout_smoke_v0",
        "source_bag": str(bag_dir),
        "storage_id": storage_id,
        "message_count": sum(counts.values()),
        "topic_counts": dict(sorted(counts.items())),
        "canonical_topic_counts": dict(sorted(canonical_counts.items())),
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
        "topic_audit": topic_audit,
        "data_quality_audit": data_quality_audit,
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
    print(
        "data_quality_audit:"
        f" profile={data_quality_audit['profile']}"
        f" pass={data_quality_audit['pass']}"
        f" issues={len(data_quality_audit['issues'])}"
    )
    if not contract_audit["pass"]:
        for issue in contract_audit["issues"]:
            print(f"  contract issue: {issue}")
    if not data_quality_audit["pass"]:
        for issue in data_quality_audit["issues"]:
            print(f"  data-quality issue: {issue}")
    if not contract_audit["pass"] or not data_quality_audit["pass"]:
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


def _audit_data_quality(
    *,
    profile: str,
    contract_audit: dict[str, Any],
    topic_audit: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    topics = topic_audit.get("canonical_topics", topic_audit["topics"])

    if not contract_audit["pass"]:
        issues.append("contract_audit failed; see contract_audit.issues")

    if profile == "smoke":
        warnings.append(
            "smoke profile does not fail non-contract topic gaps; use "
            "--quality-profile pilot or raw_pilot before pilot data capture"
        )
        return {
            "profile": profile,
            "pass": not issues,
            "policy": "fail only contract-critical command/reset loss",
            "issues": issues,
            "warnings": warnings,
            "critical_topics": [
                COMMAND_BLOCK_TOPIC,
                EXECUTED_COMMAND_BLOCK_TOPIC,
                RESET_EVENT_TOPIC,
            ],
        }

    if contract_audit["command_block_count"] <= 0:
        issues.append(f"{COMMAND_BLOCK_TOPIC} has no requested command blocks")
    if contract_audit["executed_command_block_count"] <= 0:
        issues.append(f"{EXECUTED_COMMAND_BLOCK_TOPIC} has no executed command blocks")
    if contract_audit["reset_event_count"] <= 0:
        issues.append(f"{RESET_EVENT_TOPIC} has no reset events")

    topic_rules = _topic_rules_for_profile(profile)
    for topic, rule in sorted(topic_rules.items()):
        stats = topics.get(topic)
        if stats is None or stats["record"]["count"] < rule.min_count:
            count = 0 if stats is None else stats["record"]["count"]
            issues.append(f"{topic} count {count} below minimum {rule.min_count}")
            continue

        basis_name = rule.time_basis
        basis = stats[basis_name]
        if basis["count"] <= 0:
            issues.append(f"{topic} has no {basis_name} timestamps for quality audit")
            continue

        if basis["timestamp_regressions"] > 0:
            issues.append(
                f"{topic} has {basis['timestamp_regressions']} "
                f"{basis_name} timestamp regressions"
            )

        max_gap_s = basis["max_gap_s"]
        if (
            rule.max_gap_s is not None
            and max_gap_s is not None
            and max_gap_s > rule.max_gap_s
        ):
            issues.append(
                f"{topic} max {basis_name} gap {max_gap_s:.6f}s exceeds "
                f"{rule.max_gap_s:.6f}s"
            )

        rate_hz = basis["rate_hz"]
        if (
            rule.min_rate_hz is not None
            and (rate_hz is None or rate_hz < rule.min_rate_hz)
        ):
            rate_text = "unavailable" if rate_hz is None else f"{rate_hz:.3f}Hz"
            issues.append(
                f"{topic} {basis_name} rate {rate_text} below "
                f"{rule.min_rate_hz:.3f}Hz"
            )

    return {
        "profile": profile,
        "pass": not issues,
        "policy": (
            "fail contract loss, missing critical streams, timestamp regressions, "
            "critical stream gaps, and low critical stream rates"
        ),
        "issues": issues,
        "warnings": warnings,
        "critical_topics": sorted(topic_rules),
    }


def _topic_rules_for_profile(profile: str) -> dict[str, TopicQualityRule]:
    if profile.startswith("raw_"):
        return GENESIS_RAW_TOPIC_RULES
    if profile == "smoke":
        return VISION_TOPIC_RULES
    return VISION_TOPIC_RULES


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


def _canonical_topic(topic: str) -> str:
    match = ENV_TOPIC_RE.match(topic)
    if match:
        topic = match.group(2)
    if topic == "/camera_info":
        return "/lewm/go2/camera_info"
    return topic


def _env_index(topic: str) -> int | None:
    match = ENV_TOPIC_RE.match(topic)
    if match:
        return int(match.group(1))
    return None


def _source_timestamp_ns(topic: str, msg: Any) -> int | None:
    if topic == "/clock":
        clock = getattr(msg, "clock", None)
        return _stamp_to_ns(clock)

    header = getattr(msg, "header", None)
    timestamp_ns = _stamp_to_ns(getattr(header, "stamp", None))
    if timestamp_ns is not None:
        return timestamp_ns

    transforms = getattr(msg, "transforms", None)
    if transforms:
        first_transform = transforms[0]
        transform_header = getattr(first_transform, "header", None)
        return _stamp_to_ns(getattr(transform_header, "stamp", None))

    return None


def _stamp_to_ns(stamp: Any) -> int | None:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


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
