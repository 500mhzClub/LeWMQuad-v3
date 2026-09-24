from types import SimpleNamespace
import numpy as np
import pytest

from lewm.support_friction_challenge_development import CONDITIONS,specification,pack,schedule,native_friction


def test_fixed_matched_conditions_and_schedule():
    a,b=[specification(c) for c in CONDITIONS]
    assert a['procedural_seed']==b['procedural_seed'] and a['appearance_seed']==b['appearance_seed']
    assert a['geometry']==b['geometry'] and a['friction_mu']==1 and b['friction_mu']==.15
    assert len(schedule())==225 and sum(r['segment']=='forward' for r in schedule())==120
    assert all(0<=r['requested_command'][0]<=.12 and r['requested_command'][1]==0 and abs(r['requested_command'][2])<=.25 for r in schedule())
    assert schedule()[-1]['requested_command']==[0,0,0]
    assert pack(a).physics_randomization.floor_friction_mu==1
    assert pack(b).physics_randomization.floor_friction_mu==.15
    with pytest.raises(ValueError): pack(a|{'procedural_seed':99})
    with pytest.raises(ValueError): specification('adaptive')


def build():
    solver=SimpleNamespace(values=np.ones(28),ratio=np.ones((1,28)))
    solver.get_geoms_friction=lambda ids:solver.values[ids]
    solver.get_geoms_friction_ratio=lambda ids:solver.ratio[:,ids]
    def entity(ids):
        e=SimpleNamespace(geoms=[SimpleNamespace(idx=i,friction=1.) for i in ids],_solver=solver)
        def setter(value):
            for g in e.geoms:g.friction=value;solver.values[g.idx]=value
        e.set_friction=setter;return e
    return SimpleNamespace(robot=entity(range(27)),collision_floor=entity([27]),scene=SimpleNamespace(t=0))


def test_both_sides_installed_and_native_readback():
    b=build(); r=native_friction(b,.15,install=True)
    assert r['solver_friction']==[.15]*28 and r['requested_pair_coefficient']==.15
    assert r['all_robot_geometries_changed_including_nonfoot']


@pytest.mark.parametrize('fault',['floor_only','cached_only','ratio','time'])
def test_rejects_ineffective_intervention(fault):
    b=build()
    if fault=='floor_only':b.collision_floor.set_friction(.15)
    elif fault=='cached_only':
        for g in b.robot.geoms+b.collision_floor.geoms:g.friction=.15
    elif fault=='ratio':
        native_friction(b,.15,install=True);b.robot._solver.ratio[0,1]=2
    else:b.scene.t=1
    with pytest.raises((ValueError,AssertionError)):native_friction(b,.15,install=fault=='time')
