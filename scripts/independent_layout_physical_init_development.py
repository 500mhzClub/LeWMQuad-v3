"""Inventory-backed CPU scene construction; inherited gait/recorders remain frozen."""
import copy
import numpy as np
from lewm.independent_layout_collection_development import CollectionInventory
from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
from lewm.support_friction_challenge_development import native_friction
from scripts.run_physical_graph_edge_handoff_qualification_v1 import (
    _GenesisPhysicalSession,REPO_ROOT,ExperimentError,_collect_solver_fields_compat)


class InventoryPhysicalInit(PulseContextPhysicalInit):
    def __init__(self,spec,*,backend):
        from lewm_genesis.lewm_contract import PrimitiveRegistry,SafetyLimits
        from lewm_genesis.rollout import GenesisGo2PPOPolicy,RolloutConfig,RolloutRunner
        from lewm_genesis.near_field_rgbd_scene_development import build_scene_from_pack
        from lewm_genesis.scene_loader import load_platform_manifest
        from scripts.run_go2_oracle_branch_pilot_v1 import BranchContext
        self.spec=copy.deepcopy(dict(spec));self.geometry=copy.deepcopy(spec['geometry']);self.backend=str(backend)
        if self.backend!='cpu' or not isinstance(self.inventory,CollectionInventory):raise ExperimentError('exact CPU frozen inventory required')
        definition=self.inventory.pack(spec);platform=load_platform_manifest(REPO_ROOT/'config/go2_platform_manifest.yaml')
        visual=self.output/'visual_meshes';visual.mkdir()
        build=build_scene_from_pack(definition,output=visual,appearance_arm=spec['appearance_arm'],
            appearance_seed=spec['appearance_seed'],n_envs=1,backend='cpu',show_viewer=False,render_robot=False)
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
