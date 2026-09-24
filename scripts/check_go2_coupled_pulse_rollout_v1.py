"""Exclusive offline development diagnostic; no sensors, training or physics run.

Existing audited development pulse endpoints fit the six-cell table. Completed
room responses are only evaluated, never fitted. Model-generated trajectories
test planning implementation, not counterfactual physical outcomes or JEPA.
"""
import math
from lewm.coupled_pulse_rollout_development import PulseEffect,PulseTable,plan
from scripts.diagnose_go2_room_pulse_transfer_v1 import report
from scripts.run_go2_room_return_pulse_v1 import OUTPUT as ROOM
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings

OUTPUT=ROOT/'.generated/go2_coupled_pulse_rollout_diagnostic_v1_attempt_001'


def main():
    if OUTPUT.exists():raise ValueError('exclusive offline diagnostic output')
    old=ROOT/'.generated/go2_command_pulse_response_v1_attempt_001'
    paths=[old/n for n in ('raw_pulse_audit.json','nominal_a_pulse_evaluation.json','nominal_b_pulse_evaluation.json')]
    paths += [ROOM/n for n in ('launch.json','result.json','raw_return_audit.json')]
    room=read_json(ROOM,'result.json')
    assert read_json(ROOM,'raw_return_audit.json')['status']=='RAW_RETURN_AUDIT_PASS'
    inputs={str(p.relative_to(ROOT)):digest(p) for p in paths}
    inputs|={str((ROOM/n).relative_to(ROOT)):h for n,h in room['artifact_sha256'].items()}
    sources=discover_sources(('scripts/check_go2_coupled_pulse_rollout_v1.py',),read_json(ROOM,'launch.json')['source_sha256'])
    sources['lewm/tests/test_coupled_pulse_rollout_development.py']=digest(ROOT/'lewm/tests/test_coupled_pulse_rollout_development.py')
    verify_bindings(inputs|sources)
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json',dict(scope='post-hoc offline development only',source_sha256=sources,input_sha256=inputs))
    try:
        transfer=report()
        table=PulseTable(tuple(PulseEffect(tuple(e['command']),e['pulse_ticks'],
                         tuple(e['mean_sensor_delta_xyz_yaw'][i] for i in (0,1,3)),2)
                         for e in transfer['table']),transfer['source_audit_sha256'])
        tasks={'forward':(.4,0,0),'left_quarter':(0,0,math.pi/2),'right_quarter':(0,0,-math.pi/2),
               'left_half':(0,0,math.pi),'right_half':(0,0,-math.pi)}
        plans={name:plan(table,(0,0,0),target,yaw_mode='winding',horizon=35) for name,target in tasks.items()}
        write_json(OUTPUT/'result.json',dict(status='OFFLINE_DIAGNOSTIC_COMPLETE',transfer=transfer,plans=plans,
                   fitting_uses_room_responses=False,physical_execution=False,learned_jepa_used=False,goal_achieved=False))
        verify_bindings(inputs|sources)
        print('COUPLED_OFFLINE', {k:(v['status'],len(v['action_indices']),v['predicted_position_error_m'],v['predicted_yaw_error_rad']) for k,v in plans.items()},flush=True)
        print('RESULT_SHA256',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_OFFLINE_DIAGNOSTIC_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
