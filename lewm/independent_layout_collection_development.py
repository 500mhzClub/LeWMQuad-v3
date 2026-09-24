"""Exact frozen inventory -> native scene specification and sensor-valid schedule.

Construction metadata never enters the selector: only action index, history kind,
tick and causal RGB/body packets. Actual prefix/outcome audits remain required.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from lewm.independent_layout_inventory_development import validate_inventory,HISTORIES
from lewm.longer_motion_collection_development import specification as old_spec,pack as old_pack
from lewm.coupled_pulse_rollout_development import COMMANDS
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm_genesis.scene_loader import StaticObject
from lewm_genesis.near_field_rgbd_scene_development import RENDER_NEAR_M


class CollectionInventory:
    def __init__(self,inventory):
        validate_inventory(inventory);self.data=deepcopy(inventory)
        self.layouts={l['layout_id']:l for l in self.data['layouts']}
        self.episodes={e['episode_id']:e for e in self.data['episodes']}
        self.batches={f"l{l['layout_index']:02d}":tuple(e['episode_id'] for e in self.data['episodes'] if e['layout_id']==l['layout_id'])
            for l in self.data['layouts']}
        self.identity=hashlib.sha256(json.dumps(self.data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def episode_ids(self,batch):
        if batch not in self.batches:raise ValueError('one explicit frozen layout batch required')
        return self.batches[batch]

    def specification(self,episode_id):
        if episode_id not in self.episodes:raise ValueError('explicit frozen inventory episode required')
        e=self.episodes[episode_id];layout=self.layouts[e['layout_id']]
        spec=old_spec('fit')|deepcopy(e)|dict(trial=episode_id,scene_id='independent-layout-collection-v1-'+episode_id,
            family='PROSPECTIVE_CONNECTED_LAYOUT_PULSE_COLLECTION',data_role=e['role'],condition=e['support'],
            procedural_seed=e['physics_seed'],appearance_arm='distinctive',inventory_identity=self.identity,
            render_near_m=RENDER_NEAR_M,public_depth_range_m=[.2,5.],
            geometry=dict(spawn_se2_world=list(e['spawn_se2_world']),wall_boxes=deepcopy(layout['wall_boxes'])),
            evaluation_layout=dict(cells=deepcopy(layout['cells']),edges=deepcopy(layout['edges']),pitch_m=layout['pitch_m'],role=e['role']))
        return spec

    def pack(self,spec):
        expected=self.specification(spec['trial'])
        if json.dumps(spec,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
            raise ValueError('exact frozen inventory scene specification required')
        base=old_pack(old_spec('fit'));x,y,yaw=spec['geometry']['spawn_se2_world']
        objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
            size_xyz_m=tuple(b['size_xyz']),yaw_rad=b['yaw_rad'],material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
        return replace(base,scene_id=spec['scene_id'],family=spec['family'],static_objects=objects,
            camera=replace(base.camera,near_m=spec['render_near_m']),
            physics_seed=spec['physics_seed'],topology_seed=spec['physics_seed'],visual_seed=spec['appearance_seed'],
            manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
            robot=replace(base.robot,spawn_xyz_m=(x,y,spec['spawn_z_m']),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
            physics_randomization=replace(base.physics_randomization,floor_friction_mu=spec['friction_mu']))


def schedule(action_index,history_kind):
    if type(action_index) is not int or not 0<=action_index<6 or history_kind not in HISTORIES:
        raise ValueError('six action-duration cells and explicit causal history required')
    common=[0.,0.,0.] if history_kind=='quiet' else [.12,0.,0.]
    duration=(2,5)[action_index%2];command=COMMANDS[action_index//2]
    return [dict(phase=1,role='common_'+history_kind+'_context',requested_command=list(common)) for _ in range(8)]+[
        dict(phase=2 if i<duration else 3,role='candidate_pulse' if i<duration else 'candidate_brake',
            requested_command=list(command) if i<duration else [0.,0.,0.]) for i in range(duration+20)]


def decision(action_index,history_kind,tick,policy):
    if type(tick) is not int or tick<0:raise ValueError('nonnegative actual command tick required')
    validate_policy_packet(policy);now=1_500_000_000+tick*100_000_000
    if (policy['sensor_state']['decision_ns']!=now or policy['image']['measured_ns']!=now
            or tuple(policy['sensor_state']['identity'])!=(0,0,0)):
        raise ValueError('same-episode actual decision clock required')
    rows=schedule(action_index,history_kind)
    if tick>len(rows):raise ValueError('fixed collection budget exhausted')
    row=rows[tick] if tick<len(rows) else dict(phase=9,role='terminal',requested_command=[0.,0.,0.])
    return row|dict(tick=tick,decision_ns=now,terminal=tick==len(rows),tracker_required=False,
        native_state_used=False,navigation_qualified=False)
