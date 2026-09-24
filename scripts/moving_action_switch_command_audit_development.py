"""Exact float64 request/float32 actuator audit for the new 63-command schedule."""
import numpy as np
from lewm.moving_action_switch_family_development import schedule,BRANCH_TICK


def audit_commands(raw,tape,rows,result,trial):
    n=len(raw['timestamp_s']);planned=schedule(trial)
    assert len(tape)==result['command_ticks'] and len(rows)==result['decisions']
    assert result['completed_ticks']==sum(t['completed'] for t in tape)
    assert len(tape)<=len(planned) and len(rows)<=len(planned)+1
    assert len(tape)==sum(not r['decision']['terminal'] for r in rows)
    assert result['departure_present']==any(r['tick']==BRANCH_TICK for r in rows)
    assert all(t['completed'] for t in tape[:-1])
    if tape and not tape[-1]['completed']:assert result['physical_stop'] is not None
    for i,item in enumerate(tape):
        assert item['tick']==i and type(item['completed']) is bool
        for k in ('phase','role','requested_command'):assert item[k]==planned[i][k]==rows[i]['decision'][k]
        a,b=item['pre_sample_index'],item['post_sample_index']
        assert a==749+50*i and type(b) is int and a<=b<=a+50 and b<n
        if item['completed']:assert b==a+50
        requested=np.asarray(item['requested_command'],np.float64)
        assert raw['requested_command'].dtype==raw['applied_command'].dtype==np.float64
        assert raw['post_slew_applied_command'].dtype==np.float64
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(requested,(b-a,1)))
        previous=raw['applied_command'][a].astype(np.float32)
        delta=np.array([.25,0.,.35],dtype=np.float32)
        applied=np.clip(requested.astype(np.float32),previous-delta,previous+delta).astype(np.float64)
        np.testing.assert_array_equal(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)))
        np.testing.assert_array_equal(raw['post_slew_applied_command'][a+1:b+1],np.tile(applied,(b-a,1)))
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(b-a,item['phase']))
    assert n==min(n,750)+sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    np.testing.assert_array_equal(raw['requested_command'][:min(n,750)],np.zeros((min(n,750),3)))
    if not result['setup_admitted']:assert not rows and not tape and result['physical_stop'] is not None
    if result['schedule_terminal'] is not None:
        assert result['schedule_terminal']=='FIXED_MOVING_ACTION_SWITCH_COMPLETE'
        assert result['physical_stop'] is None and result['acquisition_stop'] is None
        assert len(tape)==len(planned) and rows[-1]['decision']['terminal'] and all(t['completed'] for t in tape)
    else:assert result['physical_stop'] is not None or result['acquisition_stop'] is not None
    if result['acquisition_stop'] is not None:
        assert result['physical_stop'] is None and result['setup_admitted']
        assert result['acquisition_stop']=='STORAGE_RESERVE_STOP' or result['acquisition_stop'].startswith('PACKET_CONTRACT_STOP: ')
