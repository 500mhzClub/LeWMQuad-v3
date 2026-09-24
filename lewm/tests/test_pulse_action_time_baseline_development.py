import numpy as np
import pytest
from lewm.pulse_action_time_baseline_development import ActionTimeMean


def fixture():
    actions = np.array([0, 0, 1])
    offsets = np.zeros((3, 8), dtype=np.int64)
    offsets[:, :2] = [500_000_000, 2_200_000_000]
    offsets[2, 1] = 2_500_000_000
    active = offsets > 0
    motion = np.full((3, 8, 3), np.nan)
    motion[active] = 0
    motion[0, :2, 0] = 1
    motion[1, :2, 0] = 4
    motion[2, :2, 0] = 9
    contact = np.full((3, 8), np.nan); contact[active] = 0
    return actions, offsets, active, dict(motion=motion, contact=contact,
        motion_valid=active.copy(), contact_valid=active.copy())


def fit(args):
    return ActionTimeMean.fit(*args, roles=['train']*len(args[0]))


def test_repeated_draws_preserve_weight_and_partial_time():
    a, t, v, y = fixture(); ids = [0, 0, 1, 2]
    model = fit((a[ids], t[ids], v[ids], {k:x[ids] for k,x in y.items()}))
    p, missing = model.predict(a, t, v)
    assert not missing
    assert p[0, 1, 0] == 2
    assert p[2, 1, 0] == 9
    assert model.cells[(0, 2_200_000_000)]['training_motion_count'] == 3
    assert model.record()['training_draws'] == 4


def test_circular_angles_and_no_target_aliasing():
    a,t,v,y = fixture(); y['motion'][0,:2,2] = np.pi-.01
    y['motion'][1,:2,2] = -np.pi+.01
    model = fit((a,t,v,y)); before,_ = model.predict(a,t,v)
    assert abs(abs(np.arctan2(before[0,0,2], before[0,0,3]))-np.pi) < 1e-12
    y['motion'][:] = 123
    after,_ = model.predict(a,t,v)
    np.testing.assert_array_equal(before,after)


def test_no_interpolation_or_unseen_action_fallback():
    a,t,v,y = fixture(); model = fit((a,t,v,y))
    t[0,1] = 2_500_000_000; a[1] = 5
    p,missing = model.predict(a,t,v)
    assert len(missing) == 3
    assert np.isnan(p[0,1]).all()
    assert np.isnan(p[1,:2]).all()
    assert (p[~v] == 0).all()


def test_contact_only_cell_not_fabricated_motion():
    a,t,v,y = fixture(); y['motion_valid'][:2,1] = False
    y['motion'][:2,1] = np.nan; y['contact'][:2,1] = [0,1]
    model = fit((a,t,v,y)); p,missing = model.predict(a,t,v)
    assert len(missing) == 2
    assert p[0,1,4] == 0
    assert model.cells[(0,2_200_000_000)]['contact_frequency'] == .5
    assert all(m['motion_missing'] and not m['contact_missing'] for m in missing)


@pytest.mark.parametrize('role',['selection','development_eval','sealed',''])
def test_reject_nontraining_fit(role):
    args = fixture()
    with pytest.raises(ValueError, match='training-role'):
        ActionTimeMean.fit(*args, roles=['train',role,'train'])


@pytest.mark.parametrize('problem',['float_time','bool_action','unknown_time','nonprefix','nan_motion','nonbinary_contact'])
def test_contract_rejection(problem):
    a,t,v,y = fixture()
    if problem == 'float_time': t=t.astype(float)
    if problem == 'bool_action': a=a.astype(bool)
    if problem == 'unknown_time': t[0,7]=99
    if problem == 'nonprefix': v[0,0]=False; t[0,0]=0
    if problem == 'nan_motion': y['motion'][0,0,0]=np.nan
    if problem == 'nonbinary_contact': y['contact'][0,0]=.5
    with pytest.raises(ValueError): fit((a,t,v,y))
