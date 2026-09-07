from copy import deepcopy

import numpy as np
import pytest

from lewm.sustained_motion_collection_development import schedule
from scripts.audit_go2_sustained_observed_floor_motion_development_v1 import audit_tape


def fixture():
    commands = schedule()+[dict(segment='terminal_zero_tail',phase=10,requested_command=[0.,0.,0.]) for _ in range(5)]
    requested = np.zeros((17500,3)); phase = np.zeros(17500); tape = []
    for i,row in enumerate(commands):
        lo,hi = 750+50*i,800+50*i
        requested[lo:hi] = row['requested_command']; phase[lo:hi] = row['phase']
        tape.append(deepcopy(row)|dict(tick=i,completed=True,pre_sample_index=lo-1,post_sample_index=hi-1,
            start_perf_counter_ns=i*1000,command_finished_perf_counter_ns=i*1000+500,
            end_perf_counter_ns=i*1000+1000,outer_wall_ms=.001))
    applied = requested.astype(np.float32).astype(float)
    return dict(timestamp_s=np.arange(1,17501)*.002,requested_command=requested,
        applied_command=applied,post_slew_applied_command=applied.copy(),phase=phase),tape


def test_all_fixed_commands_and_timing_reconstruct():
    raw,tape = fixture()
    assert audit_tape(raw,tape)['all_commands_and_clocks_reconstructed']


@pytest.mark.parametrize('mutation',['missing_tail','command','applied','phase','clock','sample','duration'])
def test_acquisition_mutations_are_not_certified(mutation):
    raw,tape = fixture()
    if mutation=='missing_tail': tape.pop()
    elif mutation=='command': raw['requested_command'][999,0]=.1
    elif mutation=='applied': raw['applied_command'][999,0]=.1
    elif mutation=='phase': raw['phase'][999]=4
    elif mutation=='clock': raw['timestamp_s'][999]+=.002
    elif mutation=='sample': tape[2]['post_sample_index']-=1
    elif mutation=='duration': tape[2]['outer_wall_ms']+=.1
    with pytest.raises((ValueError,AssertionError)): audit_tape(raw,tape)
