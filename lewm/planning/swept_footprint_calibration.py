"""Reproducible Go2 collision-envelope calibration from URDF and gait logs."""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

__all__ = [
    "CollisionKinematicModel",
    "Envelope",
    "RolloutDataset",
    "SourceArtifact",
    "StateSample",
    "build_calibration_report",
    "load_open_field_rollout",
    "load_policy_nominal_stance",
    "sha256_file",
]

CARDINAL_DIRECTIONS = ("forward", "left", "rear", "right", "radius")
DEFAULT_REQUIRED_PRIMITIVES = (
    "hold",
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "backward",
    "yaw_left",
    "yaw_right",
    "arc_left",
    "arc_right",
)


@dataclass(frozen=True)
class SourceArtifact:
    role: str
    path: Path


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class CollisionPrimitive:
    link_name: str
    kind: str
    origin: np.ndarray
    parameters: tuple[float, ...]

    def support(
        self,
        link_transform: np.ndarray,
        directions_xyz: np.ndarray,
    ) -> np.ndarray:
        transform = link_transform @ self.origin
        center = transform[:3, 3]
        rotation = transform[:3, :3]
        local_directions = directions_xyz @ rotation
        support = directions_xyz @ center
        if self.kind == "box":
            half_size = 0.5 * np.asarray(self.parameters, dtype=np.float64)
            return support + np.abs(local_directions) @ half_size
        if self.kind == "sphere":
            return support + self.parameters[0]
        if self.kind == "cylinder":
            radius, length = self.parameters
            radial = radius * np.linalg.norm(local_directions[:, :2], axis=1)
            axial = 0.5 * length * np.abs(local_directions[:, 2])
            return support + radial + axial
        raise RuntimeError(f"unsupported collision primitive {self.kind!r}")


@dataclass(frozen=True)
class Envelope:
    forward_m: float
    rear_m: float
    left_m: float
    right_m: float
    radius_m: float
    directional_support_m: Mapping[int, float]

    def cardinal_dict(self) -> dict[str, float]:
        return {
            "forward": self.forward_m,
            "rear": self.rear_m,
            "left": self.left_m,
            "right": self.right_m,
            "radius": self.radius_m,
        }


@dataclass(frozen=True)
class StateSample:
    source_path: Path
    env_index: int
    timestamp_ns: int
    primitive_name: str
    command_source: str
    sequence_id: int
    block_start_ns: int
    block_phase_s: float
    is_initial_block: bool
    joint_positions: Mapping[str, float]
    base_roll_rad: float
    base_pitch_rad: float
    base_z_m: float

    @property
    def block_key(self) -> tuple[str, int, int, int]:
        return (
            str(self.source_path.resolve()),
            self.env_index,
            self.sequence_id,
            self.block_start_ns,
        )


@dataclass(frozen=True)
class RolloutDataset:
    messages_path: Path
    samples: tuple[StateSample, ...]
    artifacts: tuple[SourceArtifact, ...]
    metadata: Mapping[str, Any]
    skipped_counts: Mapping[str, int]


@dataclass(frozen=True)
class _CommandContext:
    primitive_name: str
    command_source: str
    sequence_id: int
    start_ns: int
    duration_ns: int
    is_initial_block: bool


class CollisionKinematicModel:
    """Primitive-collision URDF with deterministic forward kinematics."""

    def __init__(
        self,
        *,
        root_link: str,
        links: Iterable[str],
        joints: Sequence[JointSpec],
        collisions: Sequence[CollisionPrimitive],
        urdf_path: Path,
    ) -> None:
        self.root_link = root_link
        self.links = frozenset(links)
        self.joints = tuple(joints)
        self.collisions = tuple(collisions)
        self.urdf_path = Path(urdf_path)
        self._children: dict[str, list[JointSpec]] = {}
        for joint in self.joints:
            self._children.setdefault(joint.parent, []).append(joint)
        for children in self._children.values():
            children.sort(key=lambda joint: joint.name)
        self.actuated_joint_names = tuple(
            joint.name
            for joint in self.joints
            if joint.joint_type in {"revolute", "continuous"}
        )

    @classmethod
    def from_urdf(cls, path: str | Path) -> CollisionKinematicModel:
        urdf_path = Path(path).resolve()
        root = ET.parse(urdf_path).getroot()
        if root.tag != "robot":
            raise ValueError("URDF root element must be <robot>")
        links = {
            element.attrib["name"]
            for element in root.findall("link")
            if element.attrib.get("name")
        }
        joints: list[JointSpec] = []
        children: set[str] = set()
        for element in root.findall("joint"):
            name = _required_attribute(element, "name", "joint")
            joint_type = _required_attribute(element, "type", f"joint {name}")
            if joint_type not in {"fixed", "revolute", "continuous"}:
                raise ValueError(
                    f"joint {name!r} has unsupported type {joint_type!r}"
                )
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise ValueError(f"joint {name!r} is missing parent or child")
            parent = _required_attribute(parent_element, "link", f"joint {name}")
            child = _required_attribute(child_element, "link", f"joint {name}")
            axis_element = element.find("axis")
            axis = (
                _parse_vector(axis_element.attrib.get("xyz", "1 0 0"), 3, "axis")
                if axis_element is not None
                else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            )
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 0.0 and joint_type != "fixed":
                raise ValueError(f"joint {name!r} has a zero rotation axis")
            if axis_norm > 0.0:
                axis = axis / axis_norm
            joints.append(
                JointSpec(
                    name=name,
                    joint_type=joint_type,
                    parent=parent,
                    child=child,
                    origin=_parse_origin(element.find("origin")),
                    axis=axis,
                )
            )
            children.add(child)
        roots = sorted(links - children)
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, found {roots}")

        collisions: list[CollisionPrimitive] = []
        for link_element in root.findall("link"):
            link_name = _required_attribute(link_element, "name", "link")
            for collision in link_element.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None:
                    raise ValueError(f"collision on {link_name!r} has no geometry")
                shape_children = list(geometry)
                if len(shape_children) != 1:
                    raise ValueError(
                        f"collision on {link_name!r} must contain one shape"
                    )
                shape = shape_children[0]
                if shape.tag == "box":
                    parameters = tuple(
                        _parse_vector(
                            _required_attribute(shape, "size", "box"),
                            3,
                            "box size",
                        ).tolist()
                    )
                elif shape.tag == "sphere":
                    parameters = (
                        _positive_float(
                            _required_attribute(shape, "radius", "sphere"),
                            "sphere radius",
                        ),
                    )
                elif shape.tag == "cylinder":
                    parameters = (
                        _positive_float(
                            _required_attribute(shape, "radius", "cylinder"),
                            "cylinder radius",
                        ),
                        _positive_float(
                            _required_attribute(shape, "length", "cylinder"),
                            "cylinder length",
                        ),
                    )
                else:
                    raise ValueError(
                        f"collision on {link_name!r} uses unsupported "
                        f"geometry <{shape.tag}>"
                    )
                if any(value <= 0.0 for value in parameters):
                    raise ValueError(
                        f"collision on {link_name!r} has non-positive dimensions"
                    )
                collisions.append(
                    CollisionPrimitive(
                        link_name=link_name,
                        kind=shape.tag,
                        origin=_parse_origin(collision.find("origin")),
                        parameters=parameters,
                    )
                )
        if not collisions:
            raise ValueError("URDF contains no collision primitives")
        return cls(
            root_link=roots[0],
            links=links,
            joints=joints,
            collisions=collisions,
            urdf_path=urdf_path,
        )

    def link_transforms(
        self,
        joint_positions: Mapping[str, float],
    ) -> dict[str, np.ndarray]:
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        pending = [self.root_link]
        while pending:
            parent = pending.pop()
            for joint in self._children.get(parent, ()):
                transform = transforms[parent] @ joint.origin
                if joint.joint_type in {"revolute", "continuous"}:
                    if joint.name not in joint_positions:
                        raise ValueError(
                            f"joint state is missing actuated joint {joint.name!r}"
                        )
                    angle = _finite_float(joint_positions[joint.name], joint.name)
                    transform = transform @ _axis_angle_transform(joint.axis, angle)
                transforms[joint.child] = transform
                pending.append(joint.child)
        if set(transforms) != set(self.links):
            missing = sorted(set(self.links) - set(transforms))
            raise ValueError(f"URDF kinematic tree did not reach links: {missing}")
        return transforms

    def envelope(
        self,
        joint_positions: Mapping[str, float],
        *,
        radial_step_deg: float = 1.0,
        report_step_deg: int = 15,
    ) -> Envelope:
        radial_step = _positive_float(radial_step_deg, "radial_step_deg")
        if radial_step > 15.0:
            raise ValueError("radial_step_deg must be <= 15 degrees")
        if report_step_deg <= 0 or 360 % report_step_deg:
            raise ValueError("report_step_deg must divide 360")
        radial_angles = np.arange(0.0, 360.0, radial_step, dtype=np.float64)
        report_angles = np.arange(0.0, 360.0, report_step_deg, dtype=np.float64)
        angles = np.unique(np.concatenate((radial_angles, report_angles)))
        radians = np.deg2rad(angles)
        directions = np.stack(
            (np.cos(radians), np.sin(radians), np.zeros_like(radians)),
            axis=1,
        )
        transforms = self.link_transforms(joint_positions)
        supports = np.full(len(angles), -np.inf, dtype=np.float64)
        for collision in self.collisions:
            supports = np.maximum(
                supports,
                collision.support(transforms[collision.link_name], directions),
            )
        support_by_angle = {
            _angle_key(angle): float(value)
            for angle, value in zip(angles, supports)
        }
        directional = {
            int(angle): support_by_angle[_angle_key(angle)]
            for angle in report_angles
        }
        return Envelope(
            forward_m=support_by_angle[_angle_key(0.0)],
            left_m=support_by_angle[_angle_key(90.0)],
            rear_m=support_by_angle[_angle_key(180.0)],
            right_m=support_by_angle[_angle_key(270.0)],
            radius_m=float(np.max(supports)),
            directional_support_m=directional,
        )

    def description(self) -> dict[str, Any]:
        shape_counts: dict[str, int] = {}
        for collision in self.collisions:
            shape_counts[collision.kind] = shape_counts.get(collision.kind, 0) + 1
        return {
            "root_link": self.root_link,
            "link_count": len(self.links),
            "joint_count": len(self.joints),
            "actuated_joint_count": len(self.actuated_joint_names),
            "actuated_joint_names": list(self.actuated_joint_names),
            "collision_primitive_count": len(self.collisions),
            "collision_shape_counts": shape_counts,
            "collision_links": sorted({item.link_name for item in self.collisions}),
        }


def load_policy_nominal_stance(
    cfg_path: str | Path,
    *,
    required_joint_names: Sequence[str],
) -> dict[str, float]:
    """Load the trusted local PPO config and return its default joint angles."""

    path = Path(cfg_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted, hashed local artifact
    if not isinstance(payload, (list, tuple)) or not payload:
        raise ValueError("policy cfg must contain the expected config tuple")
    env_cfg = payload[0]
    if not isinstance(env_cfg, Mapping):
        raise ValueError("policy cfg env config must be a mapping")
    angles = env_cfg.get("default_joint_angles")
    if not isinstance(angles, Mapping):
        raise ValueError("policy cfg is missing default_joint_angles")
    result: dict[str, float] = {}
    for name in required_joint_names:
        if name not in angles:
            raise ValueError(f"policy cfg is missing default angle for {name!r}")
        result[name] = _finite_float(angles[name], name)
    return result


def load_open_field_rollout(
    messages_path: str | Path,
    *,
    required_joint_names: Sequence[str],
    workspace_root: str | Path,
    maximum_abs_roll_pitch_rad: float = math.radians(25.0),
    minimum_base_z_m: float = 0.20,
) -> RolloutDataset:
    """Load a train/open-field JSONL trace and reject benchmark-like sources."""

    path = Path(messages_path).resolve()
    root = Path(workspace_root).resolve()
    maximum_tip = _positive_float(
        maximum_abs_roll_pitch_rad,
        "maximum_abs_roll_pitch_rad",
    )
    minimum_z = _finite_float(minimum_base_z_m, "minimum_base_z_m")
    artifacts = [SourceArtifact("rollout_messages", path)]
    raw_summary_path = path.parent / "summary.json"
    raw_summary = _load_json_object(raw_summary_path, "raw rollout summary")
    artifacts.append(SourceArtifact("raw_rollout_summary", raw_summary_path))
    source_bag_value = raw_summary.get("source_bag")
    if not isinstance(source_bag_value, str) or not source_bag_value:
        raise ValueError("raw rollout summary is missing source_bag")
    source_bag = Path(source_bag_value)
    if not source_bag.is_absolute():
        source_bag = root / source_bag
    bag_summary_path = source_bag / "summary.json"
    bag_summary = _load_json_object(bag_summary_path, "source rollout summary")
    artifacts.append(SourceArtifact("source_rollout_summary", bag_summary_path))
    if bag_summary.get("split") != "train":
        raise ValueError("footprint calibration accepts only train split rollouts")
    if bag_summary.get("family") != "open_obstacle_field":
        raise ValueError(
            "footprint calibration accepts only open_obstacle_field rollouts"
        )
    policy_artifact = bag_summary.get("extra", {}).get("policy_artifact", {})
    if isinstance(policy_artifact, Mapping):
        for path_key, hash_key, role in (
            ("path", "sha256", "locomotion_policy_checkpoint"),
            ("cfg_path", "cfg_sha256", "locomotion_policy_config"),
        ):
            value = policy_artifact.get(path_key)
            if not isinstance(value, str) or not value:
                continue
            dependency = Path(value)
            if not dependency.is_absolute():
                dependency = root / dependency
            dependency = dependency.resolve()
            declared_hash = policy_artifact.get(hash_key)
            actual_hash = sha256_file(dependency)
            if declared_hash is not None and declared_hash != actual_hash:
                raise ValueError(
                    f"declared hash for {dependency} does not match its contents"
                )
            artifacts.append(SourceArtifact(role, dependency))

    current_commands: dict[int, _CommandContext] = {}
    command_counts_since_reset: dict[int, int] = {}
    pending_joint_rows: list[tuple[dict[str, Any], _CommandContext]] = []
    base_states: dict[tuple[int, int], tuple[float, float, float]] = {}
    observed_splits: set[str] = set()
    observed_families: set[str] = set()
    required = tuple(required_joint_names)
    required_set = set(required)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            topic = record.get("canonical_topic")
            env_index = int(record.get("env_index", 0))
            timestamp_ns = int(record.get("timestamp_ns", 0))
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if topic == "/lewm/go2/reset_event":
                current_commands.pop(env_index, None)
                command_counts_since_reset[env_index] = 0
            elif topic == "/lewm/go2/command_block":
                block_size = int(payload.get("block_size", 0))
                command_dt_s = _finite_float(
                    payload.get("command_dt_s"),
                    "command_dt_s",
                )
                if block_size <= 0 or command_dt_s <= 0.0:
                    raise ValueError("command block has invalid duration")
                command_count = command_counts_since_reset.get(env_index, 0)
                current_commands[env_index] = _CommandContext(
                    primitive_name=str(payload.get("primitive_name", "")),
                    command_source=str(payload.get("command_source", "unknown")),
                    sequence_id=int(payload.get("sequence_id", -1)),
                    start_ns=timestamp_ns,
                    duration_ns=int(round(block_size * command_dt_s * 1e9)),
                    is_initial_block=command_count == 0,
                )
                command_counts_since_reset[env_index] = command_count + 1
            elif topic == "/joint_states":
                context = current_commands.get(env_index)
                if context is not None:
                    pending_joint_rows.append((dict(record), context))
            elif topic == "/lewm/go2/base_state":
                pose = payload.get("pose_world", {})
                position = pose.get("position", {}) if isinstance(pose, Mapping) else {}
                base_states[(env_index, timestamp_ns)] = (
                    _finite_float(payload.get("roll_rad"), "roll_rad"),
                    _finite_float(payload.get("pitch_rad"), "pitch_rad"),
                    _finite_float(position.get("z"), "base z"),
                )
            elif topic == "/lewm/episode_info":
                split = payload.get("split")
                family = payload.get("scene_family")
                if isinstance(split, str):
                    observed_splits.add(split)
                if isinstance(family, str):
                    observed_families.add(family)
    if observed_splits != {"train"}:
        raise ValueError(
            f"rollout stream split metadata is not train: {observed_splits}"
        )
    if observed_families != {"open_obstacle_field"}:
        raise ValueError(
            "rollout stream family metadata is not open_obstacle_field: "
            f"{observed_families}"
        )

    skipped = {
        "missing_base_state": 0,
        "tipped_or_low_base": 0,
        "outside_command_window": 0,
    }
    samples: list[StateSample] = []
    for record, context in pending_joint_rows:
        env_index = int(record.get("env_index", 0))
        timestamp_ns = int(record.get("timestamp_ns", 0))
        if timestamp_ns < context.start_ns or (
            timestamp_ns > context.start_ns + context.duration_ns
        ):
            skipped["outside_command_window"] += 1
            continue
        base = base_states.get((env_index, timestamp_ns))
        if base is None:
            skipped["missing_base_state"] += 1
            continue
        roll, pitch, base_z = base
        if abs(roll) > maximum_tip or abs(pitch) > maximum_tip or base_z < minimum_z:
            skipped["tipped_or_low_base"] += 1
            continue
        payload = record["payload"]
        names = payload.get("name")
        positions = payload.get("position")
        if not isinstance(names, list) or not isinstance(positions, list):
            raise ValueError("joint state is missing name or position arrays")
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("joint state names and positions are inconsistent")
        position_map = {
            str(name): _finite_float(value, str(name))
            for name, value in zip(names, positions)
        }
        missing = sorted(required_set - set(position_map))
        if missing:
            raise ValueError(f"joint state is missing required joints: {missing}")
        samples.append(
            StateSample(
                source_path=path,
                env_index=env_index,
                timestamp_ns=timestamp_ns,
                primitive_name=context.primitive_name,
                command_source=context.command_source,
                sequence_id=context.sequence_id,
                block_start_ns=context.start_ns,
                block_phase_s=(timestamp_ns - context.start_ns) / 1e9,
                is_initial_block=context.is_initial_block,
                joint_positions={name: position_map[name] for name in required},
                base_roll_rad=roll,
                base_pitch_rad=pitch,
                base_z_m=base_z,
            )
        )
    if not samples:
        raise ValueError(f"no valid gait states found in {path}")
    metadata = {
        "split": bag_summary.get("split"),
        "family": bag_summary.get("family"),
        "scene_id": bag_summary.get("scene_id"),
        "n_envs": bag_summary.get("n_envs"),
        "sample_count": len(samples),
        "filter": {
            "maximum_abs_roll_pitch_rad": maximum_tip,
            "minimum_base_z_m": minimum_z,
        },
        "observed_primitive_names": sorted(
            {sample.primitive_name for sample in samples}
        ),
    }
    return RolloutDataset(
        messages_path=path,
        samples=tuple(samples),
        artifacts=tuple(artifacts),
        metadata=metadata,
        skipped_counts=skipped,
    )


def build_calibration_report(
    model: CollisionKinematicModel,
    *,
    nominal_joint_positions: Mapping[str, float],
    datasets: Sequence[RolloutDataset],
    source_artifacts: Sequence[SourceArtifact] = (),
    required_primitives: Sequence[str] = DEFAULT_REQUIRED_PRIMITIVES,
    minimum_safety_margin_m: float = 0.03,
    output_rounding_m: float = 0.01,
    radial_step_deg: float = 1.0,
    minimum_blocks_per_primitive: int = 10,
    minimum_samples_per_primitive: int = 40,
    minimum_noninitial_blocks_per_primitive: int = 3,
) -> dict[str, Any]:
    """Build a deterministic, JSON-safe footprint calibration report."""

    if not datasets:
        raise ValueError("at least one rollout dataset is required")
    margin = _positive_float(minimum_safety_margin_m, "minimum_safety_margin_m")
    rounding = _positive_float(output_rounding_m, "output_rounding_m")
    samples = tuple(sample for dataset in datasets for sample in dataset.samples)
    if not samples:
        raise ValueError("rollout datasets contain no states")
    nominal = model.envelope(
        nominal_joint_positions,
        radial_step_deg=radial_step_deg,
    )
    sample_envelopes = [
        model.envelope(sample.joint_positions, radial_step_deg=radial_step_deg)
        for sample in samples
    ]
    all_stats = _aggregate_envelopes(sample_envelopes)
    per_primitive: dict[str, Any] = {}
    for primitive in sorted({sample.primitive_name for sample in samples}):
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample.primitive_name == primitive
        ]
        primitive_samples = [samples[index] for index in indices]
        block_keys = {sample.block_key for sample in primitive_samples}
        noninitial_keys = {
            sample.block_key
            for sample in primitive_samples
            if not sample.is_initial_block
        }
        per_primitive[primitive] = {
            "sample_count": len(indices),
            "block_count": len(block_keys),
            "noninitial_block_count": len(noninitial_keys),
            "block_phase_s": {
                "minimum": min(sample.block_phase_s for sample in primitive_samples),
                "maximum": max(sample.block_phase_s for sample in primitive_samples),
            },
            "envelope": _aggregate_envelopes(
                [sample_envelopes[index] for index in indices]
            ),
        }

    coverage: dict[str, Any] = {}
    missing_coverage: list[str] = []
    for primitive in required_primitives:
        stats = per_primitive.get(primitive)
        reasons: list[str] = []
        if stats is None:
            reasons.append("no_samples")
            block_count = sample_count = noninitial_count = 0
        else:
            block_count = int(stats["block_count"])
            sample_count = int(stats["sample_count"])
            noninitial_count = int(stats["noninitial_block_count"])
            if block_count < minimum_blocks_per_primitive:
                reasons.append("insufficient_blocks")
            if sample_count < minimum_samples_per_primitive:
                reasons.append("insufficient_samples")
            if noninitial_count < minimum_noninitial_blocks_per_primitive:
                reasons.append("insufficient_steady_state_blocks")
        coverage[primitive] = {
            "pass": not reasons,
            "reasons": reasons,
            "block_count": block_count,
            "sample_count": sample_count,
            "noninitial_block_count": noninitial_count,
        }
        if reasons:
            missing_coverage.append(primitive)

    observed_maxima = {
        direction: max(
            nominal.cardinal_dict()[direction],
            float(all_stats[direction]["maximum"]),
        )
        for direction in CARDINAL_DIRECTIONS
    }
    maxima_provenance: dict[str, Any] = {}
    for direction in CARDINAL_DIRECTIONS:
        executed_values = [
            envelope.cardinal_dict()[direction] for envelope in sample_envelopes
        ]
        maximum_index = int(np.argmax(np.asarray(executed_values)))
        maximum_sample = samples[maximum_index]
        nominal_value = nominal.cardinal_dict()[direction]
        executed_value = executed_values[maximum_index]
        if nominal_value >= executed_value:
            maxima_provenance[direction] = {
                "source": "nominal_stance",
                "value_m": nominal_value,
            }
        else:
            maxima_provenance[direction] = {
                "source": "executed_state",
                "value_m": executed_value,
                "messages_path": str(maximum_sample.source_path),
                "env_index": maximum_sample.env_index,
                "timestamp_ns": maximum_sample.timestamp_ns,
                "primitive_name": maximum_sample.primitive_name,
                "command_source": maximum_sample.command_source,
                "sequence_id": maximum_sample.sequence_id,
                "block_phase_s": maximum_sample.block_phase_s,
                "joint_positions_rad": dict(maximum_sample.joint_positions),
                "base_roll_rad": maximum_sample.base_roll_rad,
                "base_pitch_rad": maximum_sample.base_pitch_rad,
                "base_z_m": maximum_sample.base_z_m,
            }
    tail_excursion = {
        direction: max(
            0.0,
            float(all_stats[direction]["maximum"])
            - float(all_stats[direction]["q99"]),
        )
        for direction in CARDINAL_DIRECTIONS
    }
    recommendation = {
        "static_configuration_space_radius_m": _round_up(
            observed_maxima["radius"] + margin,
            rounding,
        ),
        "action_probe": {
            "forward_m": _round_up(observed_maxima["forward"] + margin, rounding),
            "rear_m": _round_up(observed_maxima["rear"] + margin, rounding),
            "half_width_m": _round_up(
                max(observed_maxima["left"], observed_maxima["right"]) + margin,
                rounding,
            ),
            "probe_margin_m": margin,
        },
        "basis": (
            "maximum projected Genesis collision geometry across nominal and "
            "accepted executed gait states, plus explicit unmodeled margin"
        ),
    }

    artifacts = [
        SourceArtifact("genesis_collision_urdf", model.urdf_path),
        *source_artifacts,
    ]
    for dataset in datasets:
        artifacts.extend(dataset.artifacts)
    artifact_records = _artifact_records(artifacts)
    needs_rollout = bool(missing_coverage)
    return {
        "schema": "lewm_go2_swept_footprint_calibration_v1",
        "method": {
            "reference_frame": model.root_link,
            "collision_geometry": "URDF primitive support functions",
            "radial_maximum": "directional support sweep",
            "radial_step_deg": radial_step_deg,
            "report_direction_step_deg": 15,
            "quantiles": [0.5, 0.9, 0.95, 0.99],
            "includes_all_collision_links": True,
            "numeric_geometry_authority": "Genesis URDF collision primitives",
            "reference_xacros": (
                "hashed for provenance but not numerically substituted into "
                "the Genesis kinematic model"
            ),
        },
        "source_artifacts": artifact_records,
        "urdf_model": model.description(),
        "nominal_stance": {
            "joint_positions_rad": {
                name: float(nominal_joint_positions[name])
                for name in model.actuated_joint_names
            },
            "envelope_m": nominal.cardinal_dict(),
            "directional_support_m": {
                str(degree): value
                for degree, value in nominal.directional_support_m.items()
            },
        },
        "executed_states": {
            "sample_count": len(samples),
            "dataset_count": len(datasets),
            "datasets": [
                {
                    "messages_path": str(dataset.messages_path),
                    "metadata": dict(dataset.metadata),
                    "skipped_counts": dict(dataset.skipped_counts),
                }
                for dataset in datasets
            ],
            "all_primitives": all_stats,
            "per_primitive": per_primitive,
        },
        "coverage_gate": {
            "required_primitives": list(required_primitives),
            "minimum_blocks_per_primitive": minimum_blocks_per_primitive,
            "minimum_samples_per_primitive": minimum_samples_per_primitive,
            "minimum_noninitial_blocks_per_primitive": (
                minimum_noninitial_blocks_per_primitive
            ),
            "per_primitive": coverage,
            "pass": not needs_rollout,
        },
        "safety_margin": {
            "minimum_unmodeled_margin_m": margin,
            "observed_max_minus_q99_m": tail_excursion,
            "applied_margin_beyond_observed_max_m": margin,
            "physical_measurement_status": "not_available",
            "note": (
                "The applied margin is an explicit engineering floor, not a "
                "measured physical confidence bound. Physical promotion still "
                "requires hardware dimensional/controller validation."
            ),
        },
        "observed_maxima_m": observed_maxima,
        "observed_maxima_provenance": maxima_provenance,
        "recommendation": recommendation,
        "additional_genesis_rollout": {
            "required": needs_rollout,
            "primitives_missing_coverage": missing_coverage,
            "recommended_protocol": (
                "Open plane, no landmarks or walls; one warmed environment per "
                "primitive; discard two warmup blocks, then record at least ten "
                "0.5 s blocks and policy-rate joint states for each primitive."
                if needs_rollout
                else None
            ),
        },
        "freeze_status": (
            "pending_additional_genesis_and_physical_validation"
            if needs_rollout
            else "pending_physical_validation"
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_envelopes(envelopes: Sequence[Envelope]) -> dict[str, Any]:
    if not envelopes:
        raise ValueError("cannot aggregate an empty envelope sequence")
    result = {
        direction: _quantile_summary(
            [envelope.cardinal_dict()[direction] for envelope in envelopes]
        )
        for direction in CARDINAL_DIRECTIONS
    }
    degrees = sorted(envelopes[0].directional_support_m)
    result["directional_support_m"] = {
        str(degree): _quantile_summary(
            [envelope.directional_support_m[degree] for envelope in envelopes]
        )
        for degree in degrees
    }
    return result


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _artifact_records(artifacts: Sequence[SourceArtifact]) -> list[dict[str, str]]:
    unique: dict[Path, set[str]] = {}
    for artifact in artifacts:
        path = artifact.path.resolve()
        unique.setdefault(path, set()).add(artifact.role)
    return [
        {
            "roles": ",".join(sorted(unique[path])),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in sorted(unique, key=str)
    ]


def _round_up(value: float, increment: float) -> float:
    return float(math.ceil((value - 1e-12) / increment) * increment)


def _parse_origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    xyz = _parse_vector(element.attrib.get("xyz", "0 0 0"), 3, "origin xyz")
    rpy = _parse_vector(element.attrib.get("rpy", "0 0 0"), 3, "origin rpy")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_rotation(*rpy)
    transform[:3, 3] = xyz
    return transform


def _rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    rotation = np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _parse_vector(value: str, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _required_attribute(element: ET.Element, key: str, context: str) -> str:
    value = element.attrib.get(key)
    if not value:
        raise ValueError(f"{context} is missing required attribute {key!r}")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _angle_key(angle: float) -> float:
    return round(float(angle) % 360.0, 9)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload
