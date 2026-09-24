"""Synthetic corruption checks; no recorded or protected benchmark inputs."""
import numpy as np
import pytest

from scripts.audit_go2_fresh_fused_maze_development_v1 import commands
from scripts.fresh_maze_turn_conflict_audit_development import window
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.uncertain_ray_memory_development import depth_evidence, query_envelopes


def records():
    n = 11900
    raw = dict(timestamp_s=np.arange(1, n+1)*.002, requested_command=np.zeros((n, 3)),
               applied_command=np.zeros((n, 3)), phase=np.zeros(n), base_twist_world=np.zeros((n, 6)),
               base_pose_world=np.zeros((n, 7)))
    tape, decisions, cameras = [], [], []
    for i in range(223):
        pre = 749+50*i
        command = [.2, 0., 0.] if i < 218 else [0., 0., 0.]
        phase = 1 if i < 218 else 2
        item = dict(completed=True, pre_sample_index=pre, post_sample_index=pre+50,
                    requested_command=command, phase=phase)
        item['decision_tick' if i < 218 else 'tail_tick'] = i if i < 218 else i-218
        tape.append(item)
        for k in ('requested_command', 'applied_command'):
            raw[k][pre+1:pre+51] = np.asarray(command, np.float32)
        # Requested commands retain their original double values.
        raw['requested_command'][pre+1:pre+51] = command
        raw['phase'][pre+1:pre+51] = phase
    raw['post_slew_applied_command'] = raw['applied_command'].copy()
    for i in range(219):
        decisions.append(dict(tick=i, observation_index=i, decision_ns=1_500_000_000+i*100_000_000,
            pre_sample_index=749+50*i, failure=None, executed=i < 218,
            controller=dict(terminal=i == 218, requested_command=[.2, 0., 0.] if i < 218 else [0., 0., 0.]),
            start_perf_counter_ns=i*200_000_000, end_perf_counter_ns=i*200_000_000+150_000_000, outer_wall_ms=150.))
        cameras.append(dict(physical_sample_index=749+50*i))
    return raw, decisions, tape, cameras


def test_full_sample_and_command_accounting():
    report = commands(*records())
    assert report['every_physics_sample_accounted']
    assert report['full_cycle_wall_ms']['exceeding100ms'] == 218
    assert report['zero_tail_ticks'] == 5


@pytest.mark.parametrize('corruption', ['requested', 'applied', 'phase', 'partial', 'tail', 'decision', 'camera', 'time'])
def test_accounting_rejects_corruption(corruption):
    raw, decisions, tape, cameras = records()
    if corruption == 'requested': raw['requested_command'][900, 0] = .1
    if corruption == 'applied': raw['applied_command'][900, 0] = .1
    if corruption == 'phase': raw['phase'][900] = 2
    if corruption == 'partial': tape[3]['post_sample_index'] -= 1
    if corruption == 'tail': tape[-1]['requested_command'][0] = .1
    if corruption == 'decision': decisions[4]['controller']['requested_command'][0] = .1
    if corruption == 'camera': cameras[4]['physical_sample_index'] -= 1
    if corruption == 'time': decisions[4]['end_perf_counter_ns'] += 1
    with pytest.raises(AssertionError):
        commands(raw, decisions, tape, cameras)


@pytest.mark.parametrize('optical,radius', [([0., 0., 1.], 0.), ([0., 0., 1.], .04),
    ([0., 0., -.5], 0.), ([0., 0., .01], .3), ([2., 0., 1.], .03), ([0., 1., 1.], .1)])
def test_diagnostic_window_matches_reference_pixel_population(optical, radius):
    T = np.asarray(BODY_FROM_OPTICAL)
    point = np.asarray(optical)@T[:3, :3].T+T[:3, 3]
    evidence = depth_evidence(np.full((480, 640), 1., np.float32), np.ones((480, 640), bool), np.array([0., 0., 1.]))
    reference = query_envelopes(evidence, point[None], np.array([radius]), np.array([False]), backend='reference')
    bounds = window(point, radius)
    expected = 0 if bounds is None else (bounds[1]-bounds[0]+1)*(bounds[3]-bounds[2]+1)
    assert reference['projected_window_pixels'][0] == expected
