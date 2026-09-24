"""Reject role leakage and changed physical commands while retaining all phases."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.all_phase_training_targets_development import expand_trial
from lewm.observation_horizon_targets_development import derive
from lewm.geometry_progress_pilot_development import candidate_commands


def fixture(source):
    start = 3 if source == 'family' else 13
    n = 750+50*(start+40)
    pose = np.zeros((n,7)); pose[:,6] = 1.; pose[:,0] = np.arange(n)*.0001
    raw = dict(timestamp_s=np.arange(1,n+1)*.002, base_pose_world=pose,
        physics_contact=np.zeros(n), requested_command=np.zeros((n,3)))
    commands = candidate_commands('left_turn')
    for offset,command in enumerate(commands):
        a = 750+50*(start+offset); raw['requested_command'][a:a+50] = command
    cameras = [dict(physical_sample_index=749+50*i) for i in range(start+41)]
    rows = []
    for offset in (range(0,40,5) if source == 'family' else (0,)):
        labels = derive(raw,cameras,frame=start+offset,commands=commands[offset:offset+8])
        rows.append(dict(source=source,trial='fixture',action='left_turn',data_role='train',
            offset_ticks=offset,sample_id=f'old/{source}/{offset}',
            available=labels['available'],targets=labels['targets']))
    return rows,raw,cameras


@pytest.mark.parametrize('source',['family','switch'])
def test_all_phases_complete_original_overlap_and_exact_command_suffixes(source):
    original,raw,cameras=fixture(source); before=deepcopy(original)
    rows=expand_trial(original,raw,cameras)
    assert len(rows)==40 and len({r['sample_id'] for r in rows})==40
    assert original==before and all(r['data_role']=='train' for r in rows)
    assert all(sum(r['control_phase_modulo_five']==p for r in rows)==8 for p in range(5))
    assert sum(r['original_sample_id'] is not None for r in rows)==len(original)
    assert all(r['available'] and not r['future_rgb_materialized'] for r in rows)
    commands=candidate_commands('left_turn')
    for i,r in enumerate(rows):
        frame=(3 if source=='family' else 13)+i
        assert r['known_commands']==commands[i:i+8]
        assert r['history_observation_indices']==list(range(frame-3,frame+1))
        assert [t['future_observation_index'] for t in r['targets'] if t['future_image_valid']]==list(range(frame+1,frame+1+min(8,40-i)))
    assert sum(all(t['motion_valid'] for t in r['targets']) for r in rows)==33


@pytest.mark.parametrize('fault',['transfer','mixed_role','mixed_trial','mixed_action',
    'duplicate','missing','target','actual_command'])
def test_rejects_incompatible_original_role_population_or_targets(fault):
    original,raw,cameras=fixture('family')
    if fault=='transfer':
        for r in original:r['data_role']='geometry_transfer'
    elif fault=='mixed_role':original[-1]['data_role']='geometry_transfer'
    elif fault=='mixed_trial':original[-1]['trial']='another'
    elif fault=='mixed_action':original[-1]['action']='hold'
    elif fault=='duplicate':original.append(deepcopy(original[0]))
    elif fault=='missing':original.pop()
    elif fault=='target':original[0]['targets'][0]['motion'][0]+=.01
    elif fault=='actual_command':raw['requested_command'][900,0]=.1
    with pytest.raises((ValueError,AssertionError)):expand_trial(original,raw,cameras)


def test_missing_and_post_contact_contexts_are_retained():
    original,raw,cameras=fixture('family')
    raw['physics_contact'][1000]=1
    for r in original:
        labels=derive(raw,cameras,frame=3+r['offset_ticks'],
            commands=candidate_commands('left_turn')[r['offset_ticks']:r['offset_ticks']+8])
        r.update(available=labels['available'],targets=labels['targets'])
    rows=expand_trial(original,raw,cameras)
    assert len(rows)==40 and rows[0]['available']
    assert rows[0]['targets'][2]['contact']==1.
    assert rows[3]['reason']=='POST_CONTACT_CONTEXT' and rows[3]['targets'] is None
