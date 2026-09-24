"""New training-only connected-layout constructor and fixed sensor-valid excitation."""
from dataclasses import replace
import hashlib
import json
import math
from lewm.coupled_pulse_rollout_development import COMMANDS
from lewm.longer_motion_collection_development import specification as old_spec,pack as old_pack
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm_genesis.scene_loader import StaticObject

PITCH=1.2
WARMUP_TICKS=8
WARMUP_COMMAND=(.12,0.,0.)
EDGES=(((0,0),(-1,0)),((-1,0),(-1,1)),((-1,1),(0,1)),((0,1),(0,0)),
       ((-1,0),(-1,-1)),((0,1),(1,1)),((1,1),(1,0)),((1,0),(1,-1)))
SUPPORTS=('nominal','lower_friction')
TRIALS=tuple(f'{support}_action_{a}' for support in SUPPORTS for a in range(6))


def geometry():
    cells={p for e in EDGES for p in e};edges={frozenset(e) for e in EDGES};walls={}
    for x,y in sorted(cells):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if frozenset(((x,y),(x+dx,y+dy))) in edges:continue
            mid=(2*x+dx,2*y+dy);key=(mid,abs(dx))
            walls[key]=dict(wall_id=f'context_wall_{mid[0]}_{mid[1]}_{abs(dx)}',
                centre_xyz=[mid[0]*PITCH/2,mid[1]*PITCH/2,.7],
                size_xyz=[.08,PITCH+.08,1.4] if dx else [PITCH+.08,.08,1.4],
                yaw_rad=0.,material_id='NEUTRAL_WALL')
    return dict(spawn_se2_world=[.08,0.,0.],wall_boxes=[walls[k] for k in sorted(walls)])


def layout_identity():
    # This pilot has one immutable topology and one role. This identity does not
    # claim graph-isomorphism rejection for a future multi-layout inventory.
    source=dict(pitch_m=PITCH,edges=EDGES,walls=geometry()['wall_boxes'])
    return 'pulse-context-layout-v1-'+hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()


def specification(trial):
    if trial not in TRIALS:raise ValueError('exact twelve training-context trials required')
    support,action=trial.rsplit('_action_',1);action=int(action)
    return old_spec('fit')|dict(scene_id='independent-pulse-context-v1-'+trial,
        family='CONNECTED_LAYOUT_PULSE_CONTEXT_PILOT',trial=trial,condition=support,
        layout_id=layout_identity(),data_role='train',context_id=support+'_recent_forward_near_wall',
        action_index=action,command=list(COMMANDS[action//2]),pulse_ticks=(2,5)[action%2],
        warmup_ticks=WARMUP_TICKS,warmup_command=list(WARMUP_COMMAND),
        procedural_seed=2026090801,appearance_seed=2026090802,
        friction_mu=1. if support=='nominal' else .15,geometry=geometry(),
        evaluation_layout=dict(cells=[list(p) for p in sorted({p for e in EDGES for p in e})],
            edges=[[list(a),list(b)] for a,b in EDGES],pitch_m=PITCH,role='train'))


def pack(spec):
    if spec!=specification(spec['trial']):raise ValueError('exact frozen context specification required')
    base=old_pack(old_spec('fit'));x,y,yaw=spec['geometry']['spawn_se2_world']
    objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
        size_xyz_m=tuple(b['size_xyz']),yaw_rad=b['yaw_rad'],material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
    return replace(base,scene_id=spec['scene_id'],family=spec['family'],static_objects=objects,
        physics_seed=spec['procedural_seed'],topology_seed=spec['procedural_seed'],visual_seed=spec['appearance_seed'],
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        robot=replace(base.robot,spawn_xyz_m=(x,y,.375),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
        physics_randomization=replace(base.physics_randomization,floor_friction_mu=spec['friction_mu']))


def schedule(action_index):
    if type(action_index) is not int or not 0<=action_index<6:raise ValueError('six pulse-duration cells required')
    ticks=(2,5)[action_index%2];command=COMMANDS[action_index//2]
    return [dict(phase=1,role='common_recent_motion_context',requested_command=list(WARMUP_COMMAND)) for _ in range(WARMUP_TICKS)]+[
        dict(phase=2 if i<ticks else 3,role='candidate_pulse' if i<ticks else 'candidate_brake',
            requested_command=list(command) if i<ticks else [0.,0.,0.]) for i in range(ticks+20)]


def decision(action_index,tick,policy):
    """No depth, visual-pose, native-state, friction or outcome input exists."""
    if type(tick) is not int or tick<0:raise ValueError('nonnegative command tick required')
    validate_policy_packet(policy);now=1_500_000_000+tick*100_000_000
    if (policy['sensor_state']['decision_ns']!=now or policy['image']['measured_ns']!=now
            or tuple(policy['sensor_state']['identity'])!=(0,0,0)):
        raise ValueError('same-episode actual decision clock required')
    rows=schedule(action_index)
    if tick>len(rows):raise ValueError('fixed collection budget exhausted')
    row=rows[tick] if tick<len(rows) else dict(phase=9,role='terminal',requested_command=[0.,0.,0.])
    return row|dict(tick=tick,decision_ns=now,terminal=tick==len(rows),
        tracker_required=False,native_state_used=False,navigation_qualified=False)
