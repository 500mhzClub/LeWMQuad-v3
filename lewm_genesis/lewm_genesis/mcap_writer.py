"""Per-scene rosbag2 MCAP writer for the Genesis bulk rollout.

One ``MCAPSceneWriter`` instance manages one ``raw_rollout`` + ``rendered_vision``
bundle for a single scene. All ``n_envs`` parallel streams are written into
one MCAP under per-env topic namespaces (``/env_NN/lewm/go2/...``) so each
stream is independently replayable as if it were one robot.

Requires the ROS 2 Jazzy overlay (and the workspace ``install/setup.bash``)
to be sourced for ``rosbag2_py`` + ``rclpy.serialization`` + the
``lewm_go2_control`` message types.

The sidecar ``summary.json`` carries:

- scene identity (``scene_id``, ``family``, ``split``, ``manifest_sha256``)
- env / step counts
- audit-relevant rates and per-topic counts (filled in by ``audit.py`` later)

The MCAP plus sidecar together are one ``raw_rollout`` directory in the data
spec's section 4 sense.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lewm_genesis.scene_loader import ScenePack


# Topic name templates. ``{env}`` is the 2-digit env index.
TOPIC_TEMPLATES: dict[str, tuple[str, str]] = {
    # local key                     →  (topic name template, ROS type string)
    "command_block":         ("/env_{env}/lewm/go2/command_block",         "lewm_go2_control/msg/CommandBlock"),
    "executed_command_block":("/env_{env}/lewm/go2/executed_command_block","lewm_go2_control/msg/ExecutedCommandBlock"),
    "base_state":            ("/env_{env}/lewm/go2/base_state",            "lewm_go2_control/msg/BaseState"),
    "foot_contacts":         ("/env_{env}/lewm/go2/foot_contacts",         "lewm_go2_control/msg/FootContacts"),
    "reset_event":           ("/env_{env}/lewm/go2/reset_event",           "lewm_go2_control/msg/ResetEvent"),
    "episode_info":          ("/env_{env}/lewm/episode_info",              "lewm_go2_control/msg/EpisodeInfo"),
    "mode":                  ("/env_{env}/lewm/go2/mode",                  "lewm_go2_control/msg/Go2ModeState"),
    "rgb_image":             ("/env_{env}/rgb_image",                      "sensor_msgs/msg/Image"),
    "camera_info":           ("/env_{env}/camera_info",                    "sensor_msgs/msg/CameraInfo"),
    "imu":                   ("/env_{env}/imu/data",                       "sensor_msgs/msg/Imu"),
    "joint_states":          ("/env_{env}/joint_states",                   "sensor_msgs/msg/JointState"),
    "odom":                  ("/env_{env}/odom",                           "nav_msgs/msg/Odometry"),
}

GLOBAL_CLOCK_TOPIC = ("/clock", "rosgraph_msgs/msg/Clock")


def _require_rosbag2() -> None:
    try:
        import rosbag2_py  # noqa: F401
        import rclpy.serialization  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when overlay missing
        raise RuntimeError(
            "ROS 2 Jazzy overlay (workspace install/setup.bash) must be sourced "
            "before instantiating MCAPSceneWriter."
        ) from exc


def topic_name(key: str, env_index: int) -> str:
    """Resolve a logical topic key + env index to its full topic name."""

    template, _ = TOPIC_TEMPLATES[key]
    return template.format(env=f"{env_index:02d}")


def topic_type(key: str) -> str:
    """ROS 2 type string for a logical topic key."""

    return TOPIC_TEMPLATES[key][1]


@dataclass
class WriterStats:
    """Per-scene counters filled in as messages are written."""

    env_count: int = 0
    total_messages: int = 0
    per_topic_counts: dict[str, int] = field(default_factory=dict)
    first_stamp_ns: int | None = None
    last_stamp_ns: int | None = None

    def record(self, topic: str, stamp_ns: int) -> None:
        self.total_messages += 1
        self.per_topic_counts[topic] = self.per_topic_counts.get(topic, 0) + 1
        if self.first_stamp_ns is None or stamp_ns < self.first_stamp_ns:
            self.first_stamp_ns = int(stamp_ns)
        if self.last_stamp_ns is None or stamp_ns > self.last_stamp_ns:
            self.last_stamp_ns = int(stamp_ns)


class MCAPSceneWriter:
    """Writes one per-scene rosbag2 MCAP under ``out_dir/<scene_id>/``.

    The directory layout matches rosbag2's default for an ``mcap`` storage
    backend: ``<out_dir>/<scene_id>/`` contains ``metadata.yaml`` plus the
    ``*.mcap`` file. A ``summary.json`` sidecar is written by :meth:`close`.
    """

    def __init__(
        self,
        pack: ScenePack,
        out_dir: str | Path,
        *,
        n_envs: int,
        storage_id: str = "mcap",
        serialization_format: str = "cdr",
    ) -> None:
        _require_rosbag2()
        import rosbag2_py

        from rclpy.serialization import serialize_message  # noqa: F401  (sanity)

        self.pack = pack
        self.n_envs = int(n_envs)
        self.bag_dir = Path(out_dir) / pack.scene_id
        if self.bag_dir.exists():
            raise FileExistsError(f"output directory already exists: {self.bag_dir}")
        self.bag_dir.parent.mkdir(parents=True, exist_ok=True)

        self._writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_dir), storage_id=storage_id
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )
        self._writer.open(storage_options, converter_options)

        # Register one global clock topic plus all per-env topics.
        # Jazzy rosbag2_py.TopicMetadata requires a monotonic id.
        clock_topic, clock_type = GLOBAL_CLOCK_TOPIC
        next_id = 0
        self._writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=next_id,
                name=clock_topic,
                type=clock_type,
                serialization_format=serialization_format,
            )
        )
        next_id += 1
        self._registered: set[str] = {clock_topic}
        for env_idx in range(self.n_envs):
            for key in TOPIC_TEMPLATES:
                name = topic_name(key, env_idx)
                self._writer.create_topic(
                    rosbag2_py.TopicMetadata(
                        id=next_id,
                        name=name,
                        type=topic_type(key),
                        serialization_format=serialization_format,
                    )
                )
                next_id += 1
                self._registered.add(name)

        self._serialize = __import__("rclpy.serialization", fromlist=["serialize_message"]).serialize_message
        self.stats = WriterStats(env_count=self.n_envs)

    def write(self, topic: str, message: Any, stamp_ns: int) -> None:
        """Serialize and write a single message to ``topic`` at ``stamp_ns``."""

        if topic not in self._registered:
            raise KeyError(f"topic {topic!r} was not registered with this writer")
        serialized = self._serialize(message)
        self._writer.write(topic, serialized, int(stamp_ns))
        self.stats.record(topic, int(stamp_ns))

    def write_env(self, env_index: int, key: str, message: Any, stamp_ns: int) -> None:
        """Convenience: resolve ``(key, env_index)`` and write."""

        self.write(topic_name(key, env_index), message, stamp_ns)

    def write_clock(self, stamp_ns: int) -> None:
        """Write a ``/clock`` message at ``stamp_ns``."""

        # Import lazily so the module is safe to import without ROS.
        from rosgraph_msgs.msg import Clock  # type: ignore[import-not-found]

        sec = int(stamp_ns // 1_000_000_000)
        nanosec = int(stamp_ns % 1_000_000_000)
        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self.write(GLOBAL_CLOCK_TOPIC[0], msg, stamp_ns)

    def close(self, *, extra_summary: dict[str, Any] | None = None) -> Path:
        """Close the bag and write ``summary.json``. Returns the summary path."""

        # rosbag2_py's SequentialWriter is closed implicitly on garbage collection;
        # we drop the reference to flush.
        del self._writer

        summary = {
            "schema": "lewm_raw_rollout_v0",
            "scene_id": self.pack.scene_id,
            "family": self.pack.family,
            "split": self.pack.split,
            "difficulty_tier": self.pack.difficulty_tier,
            "manifest_sha256": self.pack.manifest_sha256,
            "physics_seed": self.pack.physics_seed,
            "topology_seed": self.pack.topology_seed,
            "visual_seed": self.pack.visual_seed,
            "n_envs": int(self.n_envs),
            "bag_dir": str(self.bag_dir),
            "stats": {
                "env_count": self.stats.env_count,
                "total_messages": self.stats.total_messages,
                "per_topic_counts": dict(self.stats.per_topic_counts),
                "first_stamp_ns": self.stats.first_stamp_ns,
                "last_stamp_ns": self.stats.last_stamp_ns,
                "duration_s": (
                    None
                    if self.stats.first_stamp_ns is None or self.stats.last_stamp_ns is None
                    else (self.stats.last_stamp_ns - self.stats.first_stamp_ns) / 1e9
                ),
            },
        }
        if extra_summary:
            summary["extra"] = dict(extra_summary)

        summary_path = self.bag_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary_path

    def __enter__(self) -> "MCAPSceneWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Drop the writer without writing a summary on error.
            try:
                del self._writer
            except Exception:
                pass
