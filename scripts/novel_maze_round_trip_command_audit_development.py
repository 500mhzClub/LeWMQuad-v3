"""Original raw command/slew audit with explicit prospective maze bounds."""
import numpy as np
from lewm.novel_maze_round_trip_contract_development import WARMUP_TICKS, NAVIGATION_TICKS, DRAIN_TICKS


def audit_commands(raw, tape, rows, result):
    n = len(raw['timestamp_s'])
    assert len(tape) == result['command_ticks'] <= WARMUP_TICKS + NAVIGATION_TICKS + DRAIN_TICKS
    assert len(rows) == result['decisions'] and len(tape) <= len(rows) <= len(tape)+1
    assert result['completed_ticks'] == sum(t['completed'] for t in tape)
    assert all(t['completed'] for t in tape[:-1])
    if tape and not tape[-1]['completed']:
        assert result['physical_stop'] is not None
    drain = 0
    for i, item in enumerate(tape):
        decision = rows[i]['decision']
        assert item['tick'] == rows[i]['tick'] == i
        assert item['requested_command'] == decision['requested_command']
        phase, role = ((3, 'terminal_zero_drain') if decision['terminal'] is not None else
            (1, 'causal_history_warmup') if i < WARMUP_TICKS else (2, 'online_learned_round_trip_command'))
        assert (item['phase'], item['role']) == (phase, role)
        if phase == 3:
            assert item['requested_command'] == [0., 0., 0.]
            drain += int(item['completed'])
        a, b = item['pre_sample_index'], item['post_sample_index']
        assert a == 749+50*i and type(b) is int and a <= b <= a+50 and b < n
        if item['completed']:
            assert b == a+50
        for key in ('requested_command', 'applied_command', 'post_slew_applied_command'):
            assert raw[key].dtype == np.float64
        request = np.asarray(item['requested_command'], np.float64)
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1], np.tile(request, (b-a, 1)))
        prior = raw['applied_command'][a].astype(np.float32)
        delta = np.array([.25, 0., .35], dtype=np.float32)
        applied = np.clip(request.astype(np.float32), prior-delta, prior+delta).astype(np.float64)
        for key in ('applied_command', 'post_slew_applied_command'):
            np.testing.assert_array_equal(raw[key][a+1:b+1], np.tile(applied, (b-a, 1)))
        np.testing.assert_array_equal(raw['phase'][a+1:b+1], np.full(b-a, phase))
    assert result['terminal_zero_ticks'] == drain <= DRAIN_TICKS
    assert n == min(n, 750) + sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    np.testing.assert_array_equal(raw['requested_command'][:min(n, 750)], np.zeros((min(n, 750), 3)))
    if result['physical_stop'] is None and result['acquisition_stop'] is None:
        assert result['schedule_terminal'] is not None and drain == DRAIN_TICKS
        assert rows[-1]['decision']['terminal'] == result['schedule_terminal']
        assert len(rows) == len(tape)+1


