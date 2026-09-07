"""Explicit fresh starting pose; controlled floor is an experiment condition."""
from dataclasses import replace
import hashlib
import json
import math

from lewm.longer_motion_collection_development import specification as old_spec,pack as old_pack
from lewm.support_friction_challenge_development import CONDITIONS,FRICTION


def specification(condition):
    if condition not in CONDITIONS: raise ValueError('fixed paired friction condition required')
    previous=old_spec('fit')
    return previous|dict(trial=condition,scene_id='course-aware-visual-servo-v1-'+condition,
        family='CONTROLLED_FLOOR_COURSE_AWARE_VISUAL_SERVO',procedural_seed=2026090649,
        appearance_seed=2026090651,friction_mu=FRICTION[condition],
        geometry=previous['geometry']|dict(spawn_se2_world=[-.55,-.20,-.08]),
        controlled_continuous_level_floor=True,hidden_robot_ideal_camera=True)


def pack(spec):
    if spec!=specification(spec['trial']): raise ValueError('exact frozen servo specification required')
    old=old_pack(old_spec('fit')); x,y,yaw=spec['geometry']['spawn_se2_world']
    return replace(old,scene_id=spec['scene_id'],family=spec['family'],physics_seed=spec['procedural_seed'],
        topology_seed=spec['procedural_seed'],visual_seed=spec['appearance_seed'],
        robot=replace(old.robot,spawn_xyz_m=(x,y,.375),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
        physics_randomization=replace(old.physics_randomization,floor_friction_mu=spec['friction_mu']),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
