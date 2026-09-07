"""Evaluation-only command/clock accounting, not sensor or navigation validation.

The eventual authenticated cohort evaluator must admit the completed paired
sensor phase before loading native arrays. This helper has no filesystem access.
"""
import numpy as np
from lewm.independent_tracking_challenge_development import MAX_TICKS, MAX_FRAMES, schedule


def require(value, message):
    if not value: raise ValueError(message)


def audit_commands(raw, tape, rows, result, direction):
    require(result['status'] == 'TRACKING_TAPE_REQUIRES_RAW_AUDIT'
            and result['infrastructure_failure'] is None and not result['secondary_failures'],
            'infrastructure-failed tape cannot enter a completed raw audit')
    planned = schedule(direction)
    n = result['physics_samples']
    require(type(n) is int and n >= 0, 'explicit physics count required')
    if n == 0:
        require(not raw and not rows and not tape and not result['schedule_complete']
                and (result['physical_stop'] is not None or result['acquisition_stop'] is not None),
                'zero-sample stopped trial only')
        return dict(command_accounting_verified=True, physics_samples=0, requested_intervals=0,
                    completed_intervals=0, navigation_qualified=False)
    require(all(k in raw for k in ('timestamp_s', 'requested_command', 'applied_command', 'phase')),
            'native command fields required')
    for k in ('requested_command', 'applied_command'):
        require(raw[k].shape == (n, 3) and np.isfinite(raw[k]).all(), 'complete finite native commands')
    require(raw['requested_command'].dtype == np.float64, 'exact float64 requests required')
    require(raw['timestamp_s'].shape == (n,) and raw['phase'].shape == (n,), 'complete native clocks/phases')
    np.testing.assert_allclose(raw['timestamp_s'], np.arange(1, n+1)*.002, atol=1e-9, rtol=0)
    require(len(tape) == result['command_ticks'] <= MAX_TICKS and len(rows) == result['decisions'] <= MAX_FRAMES,
            'declared command and decision counts required')
    require(result['completed_ticks'] == sum(t['completed'] for t in tape), 'completed command count differs')
    require(all(type(t['completed']) is bool for t in tape) and all(t['completed'] for t in tape[:-1]),
            'only final command can be interrupted')
    require(len(rows) == len(tape) + int(result['schedule_complete']), 'decision/dispatch population differs')
    require(result['rgbd_frames'] in (len(rows), len(rows)+1), 'capture prefix differs from decision population')
    for tick, row in enumerate(rows):
        terminal = tick == MAX_TICKS
        expected = planned[tick] if not terminal else dict(phase=10, role='terminal', requested_command=[0.,0.,0.])
        d = row['decision']
        require(row['tick'] == row['observation_index'] == d['tick'] == tick, 'decision frame identity mismatch')
        require(row['pre_sample_index'] == 749+50*tick < n, 'decision pre-sample mismatch')
        require(d['decision_ns'] == 1_500_000_000+100_000_000*tick and d['terminal'] is terminal,
                'decision clock/terminal mismatch')
        for key in ('phase', 'role', 'requested_command'):
            require(d[key] == expected[key], 'command differs from fixed tape: ' + key)
        require(d['tracker_required'] is False and d['native_state_used'] is False and d['navigation_qualified'] is False,
                'no tracking/native command authority')
        a, b = row['acquisition_wall_ms'], row['selection_wall_ms']
        require(all(type(v) in (int,float) and np.isfinite(v) and v >= 0 for v in (a,b)), 'finite measured stage timings')
        require(row['acquisition_selection_deadline_missed'] is bool(a+b>100.)
                and row['observer_computation_included'] is False and row['real_time_qualified'] is False,
                'partial-path timing must not claim full-loop qualification')
    for tick, item in enumerate(tape):
        require(item['tick'] == tick, 'ordered command intervals required')
        for key in ('phase', 'role', 'requested_command'):
            require(item[key] == planned[tick][key], 'dispatch differs from fixed tape')
        a, b = item['pre_sample_index'], item['post_sample_index']
        require(type(a) is int and type(b) is int and a == 749+50*tick and a <= b <= a+50 and b < n,
                'bounded actual command sample span required')
        require(not item['completed'] or b == a+50, 'completed interval missing samples')
        require(item['completed'] or result['physical_stop'] is not None, 'partial command requires native stop')
        request = np.asarray(item['requested_command'], np.float64)
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1], np.tile(request,(b-a,1)))
        request32 = request.astype(np.float32)
        applied = raw['applied_command'][a] + np.clip(request32-raw['applied_command'][a], [-.25,0.,-.35],[.25,0.,.35])
        np.testing.assert_allclose(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)),atol=1e-7,rtol=0)
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(b-a,item['phase']))
        ms = item['dispatch_and_physics_wall_ms']
        require(type(ms) in (int,float) and np.isfinite(ms) and ms >= 0, 'finite actual dispatch/physics timing required')
    require(n == min(n,750)+sum(t['post_sample_index']-t['pre_sample_index'] for t in tape),
            'unaccounted physics or fabricated command completion')
    np.testing.assert_array_equal(raw['requested_command'][:min(n,750)],np.zeros((min(n,750),3)))
    require(result['tracker_required_for_commands'] is False and result['native_state_used_for_commands'] is False
            and result['observer_executed'] is False and result['native_evaluation_executed'] is False,
            'collection must remain sensor-only fixed excitation')
    if result['schedule_complete']:
        require(len(tape)==MAX_TICKS and all(t['completed'] for t in tape)
                and result['physical_stop'] is None and result['acquisition_stop'] is None,
                'full tape completion contradicts stops or interval accounting')
    else:
        require(result['physical_stop'] is not None or result['acquisition_stop'] is not None,
                'incomplete tape missing stop reason')
    if not result['setup_admitted']:
        require(not rows and not tape and n <= 750, 'commands before initial setup admission')
    return dict(command_accounting_verified=True, physics_samples=n, requested_intervals=len(tape),
                completed_intervals=result['completed_ticks'], navigation_qualified=False)
