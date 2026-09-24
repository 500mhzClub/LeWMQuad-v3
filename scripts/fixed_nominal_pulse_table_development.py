"""Only fixed older nominal fitting endpoints enter the online model boundary."""
import numpy as np
from lewm.coupled_pulse_rollout_development import PulseEffect,PulseTable,COMMANDS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest

FITTING=ROOT/'.generated/go2_command_pulse_response_v1_attempt_001'
AUDIT_SHA256='af809e5cb6101bf67169e1b2a3f5e66be424b53c316d7f4e40854ee9fd5d1d46'


def load_fixed_table():
    if digest(FITTING/'raw_pulse_audit.json')!=AUDIT_SHA256:raise ValueError('fixed nominal fitting audit required')
    audit=read_json(FITTING,'raw_pulse_audit.json');cells={}
    for condition in ('nominal_a','nominal_b'):
        name=condition+'_pulse_evaluation.json'
        if digest(FITTING/name)!=audit['evaluation_sha256'][condition]:raise ValueError('fixed fitting evaluation required')
        for e in read_json(FITTING,name)['events']:
            key=(tuple(e['requested_command']),e['pulse_ticks'])
            if key[0] not in COMMANDS or key[1] not in (2,5):continue
            if not e['pulse_complete']:raise ValueError('complete fitting pulse required')
            v=e['endpoints']['brake_20']['visual']
            cells.setdefault(key,[]).append([*v['displacement_body_m'][:2],v['yaw_change_rad']])
    if len(cells)!=6 or any(len(v)!=2 for v in cells.values()):raise ValueError('two nominal samples per cell required')
    return PulseTable(tuple(PulseEffect(c,t,tuple(np.mean(cells[(c,t)],axis=0)),2)
                            for c in COMMANDS for t in (2,5)),AUDIT_SHA256)
