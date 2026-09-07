"""Three new nominal starts and one matched-start friction challenge."""
from dataclasses import replace
import hashlib
import json
import math
from lewm.longer_motion_collection_development import specification as old_spec,pack as old_pack

TRIALS=('nominal_a','nominal_b','nominal_c','lower_friction_a')
STARTS={'a':[-.65,-.35,0.],'b':[-.55,.10,.05],'c':[-.75,.20,-.06]}


def specification(trial):
    if trial not in TRIALS:raise ValueError('fixed pulse-feedback trial required')
    start=trial[-1];i='abc'.index(start);previous=old_spec('fit');condition=trial[:-2]
    return previous|dict(trial=trial,condition=condition,start=start,scene_id='pulse-feedback-v1-'+trial,
        family='CONTROLLED_FLOOR_PULSE_FEEDBACK',procedural_seed=2026090661+i,appearance_seed=2026090665+i,
        friction_mu=1. if condition=='nominal' else .15,
        geometry=previous['geometry']|dict(spawn_se2_world=STARTS[start].copy()),
        controlled_continuous_level_floor=True,hidden_robot_ideal_camera=True)


def pack(spec):
    if spec!=specification(spec['trial']):raise ValueError('exact frozen pulse-feedback specification required')
    old=old_pack(old_spec('fit'));x,y,yaw=spec['geometry']['spawn_se2_world']
    return replace(old,scene_id=spec['scene_id'],family=spec['family'],physics_seed=spec['procedural_seed'],
        topology_seed=spec['procedural_seed'],visual_seed=spec['appearance_seed'],
        robot=replace(old.robot,spawn_xyz_m=(x,y,.375),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
        physics_randomization=replace(old.physics_randomization,floor_friction_mu=spec['friction_mu']),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
