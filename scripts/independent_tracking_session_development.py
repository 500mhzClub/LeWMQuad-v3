"""New challenge constructor; unchanged core ordered RGBD capture and recorders.

No live/frozen inventory adapter is patched or impersonated. Importing this
module does not allocate Genesis, read a checkpoint, collect data or launch work.
"""
from copy import deepcopy
import numpy as np

from lewm.independent_tracking_challenge_development import pack, validate_specification
from lewm.support_friction_challenge_development import native_friction
from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
from scripts.independent_pulse_context_session_development import PulseContextSession
from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession
from scripts.independent_tracking_native_contact_guard_development import from_scene
from scripts.run_physical_graph_edge_handoff_qualification_v1 import (
    REPO_ROOT, ExperimentError, _collect_solver_fields_compat)


class IndependentTrackingPhysicalInit(PulseContextPhysicalInit):
    def __init__(self, spec, *, backend):
        validate_specification(spec)
        if backend != 'cpu':
            raise ExperimentError('one fixed CPU independent tracking scene required')
        from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
        from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner
        from lewm_genesis.union_wall_rgbd_scene_development import build_scene_from_pack
        from lewm_genesis.ordered_union_raster_development import install_order
        from lewm_genesis.scene_loader import load_platform_manifest
        from scripts.run_go2_oracle_branch_pilot_v1 import BranchContext
        self.spec = deepcopy(spec)
        self.geometry = deepcopy(spec['geometry'])
        self.backend = backend
        definition = pack(spec)
        platform = load_platform_manifest(REPO_ROOT / 'config/go2_platform_manifest.yaml')
        visual = self.output / 'visual_meshes'
        visual.mkdir()
        build = build_scene_from_pack(definition, output=visual, appearance_arm=spec['appearance_arm'],
            appearance_seed=spec['appearance_seed'], n_envs=1, backend='cpu', show_viewer=False, render_robot=False)
        try:
            self.raster_order = install_order(build.camera._rasterizer._context._scene, 'floor_first')
            self.initial_friction = native_friction(build, spec['friction_mu'], install=True)
            registry = PrimitiveRegistry.from_yaml(REPO_ROOT / 'config/go2_primitive_registry.yaml')
            safety = SafetyLimits.from_manifest(platform)
            policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, REPO_ROOT, device='cpu')
            runner = RolloutRunner(build, policy, registry, safety, config=RolloutConfig(
                n_blocks=1, fall_z_threshold_m=.15, rgb_capture_per_block=False,
                seed=definition.physics_seed, log_progress_every_blocks=0,
                foot_contact_source='zero', randomize_spawn_pose=False))
            self.ctx = BranchContext(runner=runner, policy=policy, build=build, pack=definition,
                scene_graph=None, manifest=None, grid=None, solver_fields=_collect_solver_fields_compat(build.scene))
            self._command_history = np.zeros((15, 3))
            self._control_history = np.zeros((15, 2))
            self._last_controller_observation = np.zeros(45)
            self._previous_policy_action = np.zeros(12)
            self._low_level_policy_state = np.zeros(12)
            self._contact_topology = self._build_contact_topology()
            self._runtime = dict(backend='cpu', n_envs=1, physics_dt_s=.002, policy_dt_s=.02,
                command_dt_s=.1, policy_evaluation_mode=True, policy_device='cpu',
                simulate_action_latency=bool(getattr(policy, 'simulate_action_latency', False)),
                contact_api='robot.get_contacts(exclude_self_contact=False)', forbidden_net_force_api_used=False,
                ppo_observation_contact_inputs='none_in_frozen45elementcontract',
                high_level='fixed independent observation challenge; sensor-valid commands and native stop-only supervision',
                physics_paused_during_compute=True, real_time_qualified=False, navigation_qualified=False)
        except BaseException:
            build.scene.destroy()
            self._independent_scene_destroyed = True
            raise


class IndependentTrackingSession(PulseContextSession, IndependentTrackingPhysicalInit):
    capture_fixed_rgb = CoreOrderedDynamicSession.capture_fixed_rgb

    def __init__(self, spec, output):
        # Validate before any superclass/native allocation or directory creation.
        validate_specification(spec)
        try:
            super().__init__(spec, output)
            self.contact_integrity=from_scene(self.ctx.build.scene)
        except BaseException:
            # A later recorder/joint-order initializer can fail after the
            # physical initializer has returned. It still owns a native scene.
            if hasattr(self, 'ctx') and not getattr(self, '_independent_scene_destroyed', False):
                self.ctx.build.scene.destroy()
                self._independent_scene_destroyed = True
            raise

    def _sample(self,requested,applied,timestamp_s):
        # The physics step already occurred; reject errno/capacity failure before
        # any wrapper accepts that step as a complete recorded measurement.
        self.contact_integrity.before_sample(len(self.samples))
        return super()._sample(requested,applied,timestamp_s)
