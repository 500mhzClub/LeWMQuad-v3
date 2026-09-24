"""Four-wall room for both signed turn sequences; explicitly not a maze."""
from dataclasses import replace
import hashlib
import json
import math
from lewm.longer_motion_collection_development import specification as previous_spec,pack as previous_pack
from lewm_genesis.scene_loader import StaticObject

TRIALS=('nominal_left','nominal_right','lower_friction_left')


def specification(trial):
    if trial not in TRIALS:raise ValueError('fixed room-return trial required')
    right=trial=='nominal_right';base=previous_spec('fit');condition='lower_friction' if trial.startswith('lower') else 'nominal'
    walls=[dict(wall_id=name,centre_xyz=centre,size_xyz=size,yaw_rad=0.,material_id='NEUTRAL_WALL')
           for name,centre,size in [('east',[2.5,0.,.3],[.08,5.,.6]),('west',[-2.5,0.,.3],[.08,5.,.6]),
                                   ('north',[0.,2.5,.3],[5.,.08,.6]),('south',[0.,-2.5,.3],[5.,.08,.6])]]
    return base|dict(scene_id='continuous-room-return-v1-'+trial,family='CONTROLLED_ROOM_RETURN',trial=trial,
                     condition=condition,turn_sign=-1 if right else 1,friction_mu=.15 if condition=='lower_friction' else 1.,
                     procedural_seed=2026090681+int(right),appearance_seed=2026090685+int(right),
                     geometry=base['geometry']|dict(spawn_se2_world=[-.25,.20,-.025] if right else [-.30,-.20,.025],wall_boxes=walls))


def pack(spec):
    if spec!=specification(spec['trial']):raise ValueError('exact fixed room specification required')
    old=previous_pack(previous_spec('fit'));x,y,yaw=spec['geometry']['spawn_se2_world']
    objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
                              size_xyz_m=tuple(b['size_xyz']),yaw_rad=0.,material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
    return replace(old,scene_id=spec['scene_id'],family=spec['family'],physics_seed=spec['procedural_seed'],
                   topology_seed=spec['procedural_seed'],visual_seed=spec['appearance_seed'],static_objects=objects,
                   robot=replace(old.robot,spawn_xyz_m=(x,y,.375),spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))),
                   physics_randomization=replace(old.physics_randomization,floor_friction_mu=spec['friction_mu']),
                   manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
