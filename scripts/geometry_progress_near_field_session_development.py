"""Distinct ordered near-field geometry-progress construction; inherited gait remains frozen."""
import copy
import numpy as np
from lewm.geometry_progress_near_field_development import specification,pack
from lewm.support_friction_challenge_development import native_friction
from scripts.run_physical_graph_edge_handoff_qualification_v1 import (
    _GenesisPhysicalSession,REPO_ROOT,ExperimentError,_collect_solver_fields_compat)


class GeometryProgressNearFieldPhysicalInit(_GenesisPhysicalSession):
    def __init__(self,spec,*,backend):
        from lewm_genesis.lewm_contract import PrimitiveRegistry,SafetyLimits
        from lewm_genesis.rollout import GenesisGo2PPOPolicy,RolloutConfig,RolloutRunner
        from lewm_genesis.union_wall_rgbd_scene_development import build_scene_from_pack
        from lewm_genesis.scene_loader import load_platform_manifest
        from scripts.run_go2_oracle_branch_pilot_v1 import BranchContext
        self.spec=copy.deepcopy(dict(spec));self.geometry=copy.deepcopy(spec['geometry']);self.backend=str(backend)
        if self.backend!='cpu' or spec!=specification(spec['trial']):raise ExperimentError('exact CPU geometry-progress experiment required')
        definition=pack(spec);platform=load_platform_manifest(REPO_ROOT/'config/go2_platform_manifest.yaml')
        visual=self.output/'visual_meshes';visual.mkdir()
        build=build_scene_from_pack(definition,output=visual,appearance_arm=spec['appearance_arm'],
            appearance_seed=spec['appearance_seed'],n_envs=1,backend='cpu',show_viewer=False,render_robot=False)
        from lewm_genesis.ordered_union_raster_development import install_order
        self.raster_order=install_order(build.camera._rasterizer._context._scene,'floor_first')
        self.initial_friction=native_friction(build,spec['friction_mu'],install=True)
        registry=PrimitiveRegistry.from_yaml(REPO_ROOT/'config/go2_primitive_registry.yaml');safety=SafetyLimits.from_manifest(platform)
        policy=GenesisGo2PPOPolicy.from_platform_manifest(platform,REPO_ROOT,device='cpu')
        runner=RolloutRunner(build,policy,registry,safety,config=RolloutConfig(n_blocks=1,fall_z_threshold_m=.15,
            rgb_capture_per_block=False,seed=definition.physics_seed,log_progress_every_blocks=0,
            foot_contact_source='zero',randomize_spawn_pose=False))
        self.ctx=BranchContext(runner=runner,policy=policy,build=build,pack=definition,scene_graph=None,manifest=None,grid=None,
            solver_fields=_collect_solver_fields_compat(build.scene))
        self._command_history=np.zeros((15,3));self._control_history=np.zeros((15,2))
        self._last_controller_observation=np.zeros(45);self._previous_policy_action=np.zeros(12);self._low_level_policy_state=np.zeros(12)
        self._contact_topology=self._build_contact_topology()
        self._runtime=dict(backend='cpu',n_envs=1,physics_dt_s=.002,policy_dt_s=.02,command_dt_s=.1,
            policy_evaluation_mode=True,policy_device='cpu',simulate_action_latency=bool(getattr(policy,'simulate_action_latency',False)),
            contact_api='robot.get_contacts(exclude_self_contact=False)',forbidden_net_force_api_used=False,
            ppo_observation_contact_inputs='none_in_frozen45elementcontract',
            high_level='fixed training excitation,causal packet validity,external native stops;shadow tracking only',
            physics_paused_during_compute=True,real_time_qualified=False)


from lewm.command_pulse_response_development import validate_command
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.rgbd_session_development import RGBDSession
from scripts.independent_pulse_context_session_development import PulseContextSession
from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession


class GeometryProgressNearFieldSession(RGBDSession,GeometryProgressNearFieldPhysicalInit):
    _build_contact_topology=PulseContextSession._build_contact_topology
    install_contact_identity=PulseContextSession.install_contact_identity
    capture_fixed_rgb=CoreOrderedDynamicSession.capture_fixed_rgb
    sensor_packets=PulseContextSession.sensor_packets

    def __init__(self,*args,**kwargs):
        self.guard=None;self.guard_rows=[]
        super().__init__(*args,**kwargs)

    def command_tick(self,requested):return super().command_tick(validate_command(requested))

    def _sample(self,requested,applied,timestamp_s):
        row=super()._sample(requested,applied,timestamp_s)
        if self.guard is not None:
            packet={k:np.asarray(v)[0] for k,v in self.packets[-1].items()}
            indices=nonfoot_ground_contact_indices(packet,**self.guard)
            speed=float(np.linalg.norm(row['base_twist_world'][:3]));inside=bool((np.abs(row['base_pose_world'][:2])<8).all())
            self.guard_rows.append(dict(sample_index=len(self.samples)-1,nonfoot_ground_contact_indices=indices,
                base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
            if indices or speed>.3 or not inside:raise PhysicalStop('CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP')
        return row

