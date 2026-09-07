"""New physical initialization differing only in the explicit scene builder."""
from __future__ import annotations
import copy
import hashlib
import math
from typing import Any, Mapping

from scripts.run_physical_graph_edge_handoff_qualification_v1 import (
    _GenesisPhysicalSession, REPO_ROOT, ExperimentError, canonical_bytes,
    _collect_solver_fields_compat,
)


class BoundedFloorPhysicalInit(_GenesisPhysicalSession):
    def __init__(self, spec: Mapping[str, Any], *, backend: str) -> None:
        import numpy as np
        from lewm_genesis.go2_adapter import resolve_go2_urdf
        from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
        from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner
        from lewm_genesis.bounded_scene_builder_development import build_scene_from_pack
        from lewm_genesis.scene_loader import (
            DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
            LightingSpec,
            MaterialOverride,
            PhysicsRandomization,
            RobotSpec,
            ScenePack,
            StaticObject,
            VisualRandomization,
            camera_mount_from_platform,
            load_platform_manifest,
            physics_timing_from_platform,
        )
        from scripts.run_go2_oracle_branch_pilot_v1 import BranchContext

        self.spec = copy.deepcopy(dict(spec))
        self.geometry = copy.deepcopy(dict(spec["geometry"]))
        self.backend = str(backend)
        if self.backend != "cpu":
            raise ExperimentError(
                "physical contact qualification requires the reviewed CPU manifold path"
            )
        platform_path = REPO_ROOT / "config/go2_platform_manifest.yaml"
        registry_path = REPO_ROOT / "config/go2_primitive_registry.yaml"
        platform = load_platform_manifest(platform_path)
        spawn = self.geometry["spawn_se2_world"]
        yaw = float(spawn[2])
        static_objects = tuple(
            StaticObject(
                object_id=str(row["wall_id"]),
                kind="wall",
                center_xyz_m=tuple(float(value) for value in row["centre_xyz"]),
                size_xyz_m=tuple(float(value) for value in row["size_xyz"]),
                yaw_rad=float(row["yaw_rad"]),
                material_id=str(row["material_id"]),
            )
            for row in self.geometry["wall_boxes"]
        )
        pack = ScenePack(
            scene_id=str(spec["scene_id"]),
            family=str(spec["family"]),
            split="DEVELOPMENT_UNASSIGNED",
            difficulty_tier="PHYSICAL_GRAPH_EDGE_HANDOFF",
            manifest_sha256=hashlib.sha256(canonical_bytes(self.geometry)[:-1]).hexdigest(),
            physics_seed=int(spec["procedural_seed"]),
            topology_seed=int(spec["procedural_seed"]),
            visual_seed=int(spec["procedural_seed"]),
            world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
            static_objects=static_objects,
            robot=RobotSpec(
                urdf_path=resolve_go2_urdf(platform, REPO_ROOT),
                spawn_xyz_m=(float(spawn[0]), float(spawn[1]), 0.375),
                spawn_quat_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                foot_links_in_lewm_order=DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
            ),
            camera=camera_mount_from_platform(platform),
            timing=physics_timing_from_platform(platform),
            camera_constraints={"min_camera_clearance_m": 0.10, "min_wall_thickness_m": 0.08},
            source_dir=REPO_ROOT,
            visual_randomization=VisualRandomization(
                material_overrides=(
                    MaterialOverride("NEUTRAL_FLOOR", (0.50, 0.50, 0.50, 1.0)),
                    MaterialOverride("NEUTRAL_WALL", (0.35, 0.35, 0.35, 1.0)),
                ),
                lighting=LightingSpec(
                    direction=(0.0, 0.0, -1.0), diffuse_rgb=(0.8, 0.8, 0.8),
                    specular_rgb=(0.2, 0.2, 0.2), ambient_rgb=(0.25, 0.25, 0.25),
                ),
            ),
            physics_randomization=PhysicsRandomization(1.0, 0.0, 1.0, 0.0),
        )
        build = build_scene_from_pack(
            pack, n_envs=1, backend=self.backend, show_viewer=False,
            render_robot=False, apply_textures=False,
        )
        registry = PrimitiveRegistry.from_yaml(registry_path)
        safety = SafetyLimits.from_manifest(platform)
        policy = GenesisGo2PPOPolicy.from_platform_manifest(
            platform, REPO_ROOT, device="cpu"
        )
        runner = RolloutRunner(
            build, policy, registry, safety,
            config=RolloutConfig(
                n_blocks=1, fall_z_threshold_m=0.15, rgb_capture_per_block=False,
                seed=int(spec["procedural_seed"]), log_progress_every_blocks=0,
                foot_contact_source="zero", randomize_spawn_pose=False,
            ),
        )
        self.ctx = BranchContext(
            runner=runner, policy=policy, build=build, pack=pack,
            scene_graph=None, manifest=None, grid=None,
            solver_fields=_collect_solver_fields_compat(build.scene),
        )
        # These are controller inputs/state, not reconstructed report fields.
        # Keep a rolling copy at the same command/policy boundaries used by
        # the production loop so that the frozen ranker can consume the exact
        # persisted predecessor state and exact restore can reinstate it.
        self._command_history = np.zeros((15, 3), dtype=np.float64)
        self._control_history = np.zeros((15, 2), dtype=np.float64)
        self._last_controller_observation = np.zeros((45,), dtype=np.float64)
        self._previous_policy_action = np.zeros((12,), dtype=np.float64)
        self._low_level_policy_state = np.zeros((12,), dtype=np.float64)
        self._contact_topology = self._build_contact_topology()
        self._runtime = {
            "backend": self.backend,
            "n_envs": 1,
            "physics_dt_s": float(pack.timing.physics_dt_s),
            "policy_dt_s": float(pack.timing.policy_dt_s),
            "command_dt_s": float(pack.timing.command_dt_s),
            "policy_evaluation_mode": True,
            "policy_device": "cpu",
            "simulate_action_latency": bool(getattr(policy, "simulate_action_latency", False)),
            "contact_api": "robot.get_contacts(exclude_self_contact=False)",
            "forbidden_net_force_api_used": False,
            "snapshot_restore_source": "scripts/run_go2_oracle_branch_pilot_v1.py",
            "solver_field_collector": (
                "runner-owned exact reviewed walk port for quadrants 0.6.2"
            ),
            "ppo_observation_contact_inputs": "none_in_frozen_45_element_contract",
            "foot_contact_source_zero_scope": (
                "rollout telemetry placeholder only; physical contact labels use "
                "robot.get_contacts at every 2 ms physics step"
            ),
        }

