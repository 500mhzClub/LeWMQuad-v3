"""Built-in kinematic Genesis stack for the first Shared V5 maze smoke.

This module is development-only.  Its calibrated raster fuser provides a
practical closed-loop integration path, but neither the fuser nor its receipts
grant G2/G4 qualification, production promotion, or held-out authority.
Genesis is imported only when :func:`build_kinematic_development_stack` is
called; importing this source does not start a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    OUTPUT_CELL_SIZE_M,
    OUTPUT_FORWARD_MIN_EDGE_M,
    OUTPUT_LEFT_MIN_EDGE_M,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    soft_rasterize_observable_camera_ray_evidence_v4,
)
from lewm.models.two_resolution_frontier_value_head_v1 import (
    FrozenCandidateFeatureBatchV1,
)
from lewm.navigation.shared_v5_dev_runtime import (
    DevelopmentPhysicalFuseReceipt,
    G4CandidateBatch,
    MotionCommand,
    Pose2D,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    FusionMode,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    PHYSICAL_CELL_SIZE_M,
    PROFILE_SHA256,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
)


_DEVELOPMENT_FUSER_PRODUCER_SHA256 = hashlib.sha256(
    b"lewm_shared_v5_development_raster_fuser_v1_not_qualification"
).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0,1]")
    return result


class DevelopmentRasterPhysicalFuser:
    """Threshold the calibrated Shared V5 raster into development memory."""

    def __init__(self, calibration: Mapping[str, object]) -> None:
        if not isinstance(calibration, Mapping) or calibration.get("qualified") is not True:
            raise ValueError("physical calibration must be explicitly qualified")
        self.free_threshold = _probability(
            calibration.get("free_probability_threshold"),
            name="free_probability_threshold",
        )
        self.occupied_threshold = _probability(
            calibration.get("occupied_probability_threshold"),
            name="occupied_probability_threshold",
        )
        self.camera_transform_sha256 = _require_sha256(
            calibration.get("camera_transform_sha256"),
            name="camera_transform_sha256",
        )

    @staticmethod
    def _world_xy(pose: Pose2D, forward_m: float, left_m: float) -> tuple[float, float]:
        cosine = math.cos(pose.yaw_rad)
        sine = math.sin(pose.yaw_rad)
        return (
            pose.x_m + cosine * forward_m - sine * left_m,
            pose.y_m + sine * forward_m + cosine * left_m,
        )

    def fuse(
        self,
        *,
        evidence: object,
        pose: Pose2D,
        tick_index: int,
        memory: object,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> DevelopmentPhysicalFuseReceipt:
        del ground_plane_z_body_m
        if type(memory) is not RevisionedPhysicalMemory:
            raise TypeError("development fuser requires RevisionedPhysicalMemory")
        revision_before = memory.revision
        with torch.inference_mode():
            raster = soft_rasterize_observable_camera_ray_evidence_v4(
                evidence,
                camera_origin_body_m=camera_origin_body_m,
                camera_basis_body_fru=camera_basis_body_fru,
            )
        if tuple(raster.class_probabilities.shape[:2]) != (1, 3):
            raise ValueError("development fuser requires one raster batch")
        free = raster.class_probabilities[0, 1].detach().cpu()
        occupied = raster.class_probabilities[0, 2].detach().cpu()
        labels: dict[tuple[int, int], PhysicalLabel] = {}
        quarter_offsets = (0.25 * OUTPUT_CELL_SIZE_M, 0.75 * OUTPUT_CELL_SIZE_M)
        for row in range(int(free.shape[0])):
            for column in range(int(free.shape[1])):
                if float(occupied[row, column]) >= self.occupied_threshold:
                    label = PhysicalLabel.OCCUPIED
                elif float(free[row, column]) >= self.free_threshold:
                    label = PhysicalLabel.FREE
                else:
                    continue
                for forward_offset in quarter_offsets:
                    for left_offset in quarter_offsets:
                        forward = (
                            OUTPUT_FORWARD_MIN_EDGE_M
                            + row * OUTPUT_CELL_SIZE_M
                            + forward_offset
                        )
                        left = (
                            OUTPUT_LEFT_MIN_EDGE_M
                            + column * OUTPUT_CELL_SIZE_M
                            + left_offset
                        )
                        cell = memory.map_frame.world_to_cell(
                            self._world_xy(pose, forward, left)
                        )
                        previous = labels.get(cell)
                        if previous is not PhysicalLabel.OCCUPIED:
                            labels[cell] = label
        evidence_rows = tuple(
            PhysicalCellEvidence(cell=cell, label=label)
            for cell, label in sorted(labels.items())
        )
        payload = _canonical_sha256(
            {
                "schema": "lewm_shared_v5_development_raster_labels_v1",
                "tick_index": tick_index,
                "labels": [row.to_dict() for row in evidence_rows],
            }
        )
        observation = ObservationIdentity(
            observation_id=f"shared-v5-development-frame-{tick_index}",
            payload_sha256=payload,
            producer_sha256=_DEVELOPMENT_FUSER_PRODUCER_SHA256,
            authority=EvidenceAuthority.LEARNED_PHYSICAL,
        )
        transaction = PhysicalEvidenceTransaction(
            observation=observation,
            map_frame=memory.map_frame,
            pose=PoseProvenance(
                source=PoseSource.DEPLOYMENT_ODOMETRY,
                frame_id=memory.map_frame.frame_id,
                mean_xy_yaw=(pose.x_m, pose.y_m, pose.yaw_rad),
                covariance_xy_yaw=((0.0, 0.0, 0.0),) * 3,
                timestamp_ns=tick_index,
                synchronization_id=f"shared-v5-development-sync-{tick_index}",
                camera_transform_sha256=self.camera_transform_sha256,
            ),
            physical_evidence=evidence_rows,
            projection_contract_sha256=PROFILE_SHA256,
        )
        memory.apply_transaction(transaction)
        return DevelopmentPhysicalFuseReceipt(
            memory=memory,
            physical_map_frame_sha256=memory.map_frame.content_sha256,
            revision_before=revision_before,
            revision_after=memory.revision,
            physical_content_sha256=memory.physical_content_sha256,
        )


class GenesisKinematicDevBackend:
    """Single-environment RGB renderer with conservative kinematic motion."""

    def __init__(self, build: object, *, device: torch.device) -> None:
        self.build = build
        self.pack = build.pack
        self.device = device
        self._spawn_xyz = tuple(float(value) for value in self.pack.robot.spawn_xyz_m)
        qw, qx, qy, qz = (float(value) for value in self.pack.robot.spawn_quat_wxyz)
        self._spawn_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self._pose = Pose2D(self._spawn_xyz[0], self._spawn_xyz[1], self._spawn_yaw)
        self._last_camera_body: tuple[np.ndarray, np.ndarray] | None = None
        self._stopped = False

    @staticmethod
    def _quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))

    def _write_robot_pose(self) -> None:
        self.build.robot.set_pos(
            (self._pose.x_m, self._pose.y_m, self._spawn_xyz[2]),
            zero_velocity=True,
        )
        self.build.robot.set_quat(self._quat_wxyz(self._pose.yaw_rad), zero_velocity=False)

    def reset(self) -> None:
        self._stopped = False
        self._pose = Pose2D(self._spawn_xyz[0], self._spawn_xyz[1], self._spawn_yaw)
        self._write_robot_pose()

    def render_rgb(self) -> np.ndarray:
        if self._stopped:
            raise RuntimeError("kinematic backend is stopped")
        from lewm_genesis.camera_safety import (
            camera_safety_config_from_pack,
            safe_camera_pose_from_base,
        )
        from lewm_genesis.rollout import RolloutRunner
        from lewm_genesis.scene_loader import effective_camera_mount_xyz_rpy

        yaw = self._pose.yaw_rad
        quat_xyzw = np.array(
            [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
            dtype=np.float64,
        )
        base = np.array(
            [self._pose.x_m, self._pose.y_m, self._spawn_xyz[2]],
            dtype=np.float64,
        )
        mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(self.pack)
        camera_pose, _safety = safe_camera_pose_from_base(
            base,
            quat_xyzw,
            mount_xyz_body=mount_xyz,
            mount_rpy_body=mount_rpy,
            objects=self.pack.static_objects,
            config=camera_safety_config_from_pack(self.pack),
        )
        self.build.camera.set_pose(
            pos=camera_pose.position,
            lookat=camera_pose.lookat,
            up=camera_pose.up,
        )
        rgb = RolloutRunner._extract_rgb(self.build.camera.render())
        if rgb is None:
            raise RuntimeError("Genesis camera returned no RGB frame")
        if rgb.ndim == 4:
            rgb = rgb[0]
        if tuple(rgb.shape[-1:]) != (3,):
            raise RuntimeError("Genesis camera RGB shape changed")
        cosine = math.cos(yaw)
        sine = math.sin(yaw)

        def yaw_body(vector_world: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    cosine * vector_world[0] + sine * vector_world[1],
                    -sine * vector_world[0] + cosine * vector_world[1],
                    vector_world[2],
                ],
                dtype=np.float32,
            )

        origin = yaw_body(np.asarray(camera_pose.position) - base)
        forward = yaw_body(np.asarray(camera_pose.lookat) - np.asarray(camera_pose.position))
        forward /= np.linalg.norm(forward)
        up_hint = yaw_body(np.asarray(camera_pose.up))
        right = np.cross(forward, up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        self._last_camera_body = (origin, np.stack((forward, right, up)))
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def preprocess_rgb(self, frame: np.ndarray) -> torch.Tensor:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("RGB frame must have HxWx3 shape")
        image = torch.from_numpy(np.ascontiguousarray(frame)).to(
            device=self.device,
            dtype=torch.float32,
        )
        image = image.permute(2, 0, 1)[None] / 255.0
        image = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return ((image - mean) / std).contiguous()

    def camera_calibration_tensors(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._last_camera_body is None:
            raise RuntimeError("render_rgb must precede camera calibration")
        origin, basis = self._last_camera_body
        return (
            torch.as_tensor(origin, dtype=image.dtype, device=image.device)[None],
            torch.as_tensor(basis, dtype=image.dtype, device=image.device)[None],
            torch.tensor(
                [-self._spawn_xyz[2]],
                dtype=image.dtype,
                device=image.device,
            ),
        )

    def pose_xy_yaw(self) -> Pose2D:
        return self._pose

    def apply_command(self, command: MotionCommand) -> None:
        if self._stopped:
            raise RuntimeError("kinematic backend is stopped")
        dt = float(self.pack.timing.command_dt_s)
        mid_yaw = self._pose.yaw_rad + 0.5 * command.yaw_rate_radps * dt
        world_dx = (
            math.cos(mid_yaw) * command.vx_body_mps
            - math.sin(mid_yaw) * command.vy_body_mps
        ) * dt
        world_dy = (
            math.sin(mid_yaw) * command.vx_body_mps
            + math.cos(mid_yaw) * command.vy_body_mps
        ) * dt
        yaw = (self._pose.yaw_rad + command.yaw_rate_radps * dt + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        self._pose = Pose2D(self._pose.x_m + world_dx, self._pose.y_m + world_dy, yaw)
        self._write_robot_pose()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

    def observer_snapshot(self) -> dict[str, object]:
        return {
            "pose_xy_yaw": as_pose_tuple(self._pose),
            "scene_id": self.pack.scene_id,
            "backend_stopped": self._stopped,
        }


def as_pose_tuple(pose: Pose2D) -> tuple[float, float, float]:
    return (pose.x_m, pose.y_m, pose.yaw_rad)


class DevelopmentG4CandidateBuilder:
    def __init__(self, head: object, calibration: Mapping[str, object]) -> None:
        if calibration.get("candidate_feature_schema") != "relative_dx_dy_distance_zero_pad_v1":
            raise ValueError("G4 calibration candidate feature schema is unsupported")
        self.head = head

    def __call__(self, **values: object) -> G4CandidateBatch:
        cells = tuple(values["nearest_cells"])
        start = tuple(values["start_cell"])
        dimension = int(self.head.config.candidate_feature_dim)
        if dimension < 3:
            raise ValueError("G4 candidate feature dimension must be at least three")
        rows = []
        row_sha256s = []
        for cell in cells:
            dx = float(cell[0] - start[0])
            dy = float(cell[1] - start[1])
            row = [dx, dy, math.hypot(dx, dy)] + [0.0] * (dimension - 3)
            rows.append(row)
            row_sha256s.append(
                _canonical_sha256(
                    {"schema": "lewm_development_g4_candidate_row_v1", "cell": list(cell)}
                )
            )
        device = next(self.head.parameters()).device
        features = torch.tensor([rows], dtype=torch.float32, device=device)
        candidate_set_sha = _canonical_sha256(
            {"schema": "lewm_development_g4_candidate_set_v1", "rows": row_sha256s}
        )
        return G4CandidateBatch(
            cells=cells,
            head_batch=FrozenCandidateFeatureBatchV1(
                candidate_set_sha256=candidate_set_sha,
                candidate_row_sha256s=tuple(row_sha256s),
                features=features,
            ),
        )


def _even_physical_shape(
    bounds: Sequence[Sequence[float]],
) -> tuple[int, int]:
    spans = (
        float(bounds[1][0]) - float(bounds[0][0]),
        float(bounds[1][1]) - float(bounds[0][1]),
    )
    values = []
    for span in spans:
        count = max(2, int(math.ceil(span / PHYSICAL_CELL_SIZE_M)))
        values.append(count if count % 2 == 0 else count + 1)
    return (values[0], values[1])


def build_kinematic_development_stack(
    *,
    scene_pack: object,
    genesis_backend: str,
    device: torch.device,
    physical_calibration: Mapping[str, object],
    g4_head: object | None,
    g4_calibration: Mapping[str, object] | None,
    **_unused: object,
) -> dict[str, object]:
    """Build one real-render, kinematic-motion development stack.

    No PPO or locomotion artifact is opened.  A future physical-motion mode
    must be a separate explicit path with a regular trained artifact binding.
    """

    from lewm_genesis.scene_builder import build_scene_from_pack

    build = build_scene_from_pack(
        scene_pack,
        n_envs=1,
        backend=genesis_backend,
        show_viewer=False,
        render_robot=False,
        apply_textures=True,
    )
    camera_transform_sha256 = _require_sha256(
        physical_calibration.get("camera_transform_sha256"),
        name="camera_transform_sha256",
    )
    physical_shape = _even_physical_shape(scene_pack.world_bounds_xy_m)
    physical_frame = MapFrameIdentity(
        session_id=f"{scene_pack.scene_id}:shared-v5-development-physical",
        origin_xy_m=tuple(scene_pack.world_bounds_xy_m[0]),
        cell_size_m=PHYSICAL_CELL_SIZE_M,
        frame_id="shared_v5_development_physical_0p05m",
    )
    configuration_frame = MapFrameIdentity(
        session_id=f"{scene_pack.scene_id}:shared-v5-development-configuration",
        origin_xy_m=tuple(scene_pack.world_bounds_xy_m[0]),
        cell_size_m=CONFIGURATION_CELL_SIZE_M,
        frame_id="shared_v5_development_configuration_0p10m",
    )
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=physical_frame,
            fusion_mode=FusionMode.CURRENT_FRAME_ONLY,
            planning_connectivity=4,
            allow_diagonal_corner_cutting=False,
            require_registered_lattice=False,
            physical_projection_contract_sha256=PROFILE_SHA256,
            expected_camera_transform_sha256=camera_transform_sha256,
            promoted_runtime=False,
        )
    )
    projection = TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=configuration_frame,
        physical_shape=physical_shape,
        configuration_shape=(physical_shape[0] // 2, physical_shape[1] // 2),
    )
    planner = TwoResolutionConfigurationPlannerV2(projection)
    candidate_builder = None
    if g4_head is not None:
        if not isinstance(g4_calibration, Mapping):
            raise ValueError("trained G4 requires explicit calibration")
        candidate_builder = DevelopmentG4CandidateBuilder(g4_head, g4_calibration)
    return {
        "backend": GenesisKinematicDevBackend(build, device=device),
        "physical_fuser": DevelopmentRasterPhysicalFuser(physical_calibration),
        "physical_memory": memory,
        "projection": projection,
        "planner": planner,
        "g4_candidate_builder": candidate_builder,
    }


__all__ = [
    "DevelopmentG4CandidateBuilder",
    "DevelopmentRasterPhysicalFuser",
    "GenesisKinematicDevBackend",
    "build_kinematic_development_stack",
]
