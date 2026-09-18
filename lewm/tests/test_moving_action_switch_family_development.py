from collections import Counter
import numpy as np
from lewm.moving_action_switch_family_development import (
    ACTIONS,TRIALS,assignments,specification,branch_specification,schedule,native_horizons)


def test_complete_balanced_roster_and_identical_sibling_scene_prefixes():
    cells=assignments()
    assert len(TRIALS)==144 and len(set(TRIALS))==144
    assert Counter(r['data_role'] for r in cells.values())=={'train':72,'geometry_transfer':72}
    assert sum(r['prefix_action']==r['suffix_action'] for r in cells.values())==24
    for cluster in {r['cluster'] for r in cells.values()}:
        for action in ACTIONS:
            siblings=[t for t,r in cells.items() if r['cluster']==cluster and r['prefix_action']==action]
            assert {cells[t]['suffix_action'] for t in siblings}==set(ACTIONS)
            first=siblings[0]
            for t in siblings:
                assert specification(t)==specification(first)
                assert schedule(t)[:13]==schedule(first)[:13]
                assert branch_specification(t)['history_observation_indices']==[10,11,12,13]


def test_exact_branch_and_drain_boundaries():
    from lewm.geometry_progress_pilot_development import candidate_commands
    for t,r in assignments().items():
        plan=schedule(t)
        assert len(plan)==63
        assert [p['requested_command'] for p in plan[3:13]]==[list(candidate_commands(r['prefix_action'])[0])]*10
        assert [p['requested_command'] for p in plan[13:53]]==[list(c) for c in candidate_commands(r['suffix_action'])]
        assert all(p['requested_command']==[0.,0.,0.] for p in plan[53:])


def raw(n):
    pose=np.zeros((n,7));pose[:,0]=np.arange(n)*.001;pose[:,6]=1.
    return dict(timestamp_s=(np.arange(n)+1)*.002,base_pose_world=pose,physics_contact=np.zeros(n,bool))


def frames(n):
    return [dict(physical_sample_index=i) for i in range(749,n,50)]


def test_labels_use_branch_clock_and_complete_four_second_suffix():
    r=native_horizons(raw(3900),frames(3900))
    assert r['departure_ns']==2_800_000_000 and r['departure_tick']==13
    assert [t['future_observation_index'] for t in r['targets']]==list(range(18,54,5))
    assert all(t['motion_valid'] and t['future_image_valid'] and t['contact']==0. for t in r['targets'])
    assert np.allclose(r['targets'][0]['motion'],[.25,0.,0.])


def test_contact_and_missing_suffix_are_never_zero_motion_targets():
    data=raw(1700);data['physics_contact'][1699]=True
    r=native_horizons(data,frames(1699))
    assert r['targets'][0]['motion_valid']
    assert all(t['contact']==1. and t['motion'] is None and not t['future_image_valid'] for t in r['targets'][1:])
    empty=native_horizons(raw(1400),frames(1400))
    assert all(not t['motion_valid'] and not t['contact_valid'] and t['motion'] is None for t in empty['targets'])
    assert native_horizons(raw(1399),frames(1399)) is None
