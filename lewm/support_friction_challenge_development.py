"""Matched fresh friction intervention; fixed supervised commands, not a policy."""
from dataclasses import replace
import hashlib
import json
import numpy as np

from lewm.longer_motion_collection_development import specification as old_spec,pack as old_pack

CONDITIONS=('nominal','lower_friction')
FRICTION={'nominal':1.,'lower_friction':.15}
SEGMENTS=(('initial_quiet',10,(0.,0.,0.)),('forward',120,(.12,0.,0.)),
    ('forward_brake',10,(0.,0.,0.)),('left',30,(0.,0.,.25)),('left_brake',10,(0.,0.,0.)),
    ('right',30,(0.,0.,-.25)),('right_brake',10,(0.,0.,0.)),('zero_tail',5,(0.,0.,0.)))


def specification(condition):
    if condition not in CONDITIONS: raise ValueError('fixed friction condition required')
    return old_spec('fit')|dict(trial=condition,scene_id='support-friction-challenge-v1-'+condition,
        family='SUPERVISED_MATCHED_FRICTION_CHALLENGE',procedural_seed=2026090641,appearance_seed=2026090643,
        friction_mu=FRICTION[condition],ideal_foot_sensor_identity='support-friction-v1-'+condition)


def pack(spec):
    if spec!=specification(spec['trial']): raise ValueError('exact declared friction spec required')
    old=old_pack(old_spec('fit')); seed=spec['procedural_seed']
    return replace(old,scene_id=spec['scene_id'],family=spec['family'],physics_seed=seed,topology_seed=seed,
        visual_seed=spec['appearance_seed'],physics_randomization=replace(old.physics_randomization,floor_friction_mu=spec['friction_mu']),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


def schedule():
    return [dict(segment=name,phase=phase,requested_command=list(command))
        for phase,(name,count,command) in enumerate(SEGMENTS,1) for _ in range(count)]


def native_friction(build,mu,*,install=False):
    """Read solver coefficients AND ratios, not only cached Python material data."""
    if mu not in FRICTION.values(): raise ValueError('predeclared intervention required')
    if install:
        if int(build.scene.t)!=0: raise ValueError('install only before any physics')
        build.robot.set_friction(mu); build.collision_floor.set_friction(mu)
    robot=[int(g.idx) for g in build.robot.geoms]; floor=[int(g.idx) for g in build.collision_floor.geoms]
    if len(robot)!=27 or len(floor)!=1 or set(robot)&set(floor): raise ValueError('exact27robot plusonefloor geoms required')
    ids=robot+floor; solver=build.robot._solver
    def array(v): return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
    friction=array(solver.get_geoms_friction(ids)); ratio=array(solver.get_geoms_friction_ratio(ids))
    if friction.shape!=(28,) or ratio.shape!=(1,28): raise ValueError('explicit solver geometry axes required')
    np.testing.assert_allclose(friction,mu,atol=1e-7,rtol=0); np.testing.assert_array_equal(ratio,np.ones((1,28)))
    cached=np.array([g.friction for g in list(build.robot.geoms)+list(build.collision_floor.geoms)])
    np.testing.assert_allclose(cached,mu,atol=1e-7,rtol=0)
    return dict(robot_geom_ids=robot,ground_geom_ids=floor,solver_friction=friction.tolist(),solver_ratio=ratio.tolist(),
        cached_friction=cached.tolist(),pair_combination='maximum',requested_pair_coefficient=mu,
        all_robot_geometries_changed_including_nonfoot=True,physics_steps=int(build.scene.t))
