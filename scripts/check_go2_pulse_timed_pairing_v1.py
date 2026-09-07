"""One recorded metadata diagnostic; no fitting, physics or target leakage."""
import json
from collections import Counter
from lewm.pulse_timed_observation_pairing_development import pulse_window
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings

INPUT=ROOT/'.generated/go2_coupled_room_return_v1_attempt_001'
OUTPUT=ROOT/'.generated/go2_pulse_timed_pairing_diagnostic_v1_attempt_001'
PROTOCOL='docs/go2_pulse_timed_pairing_diagnostic_v1_2026-09-06.md'
TRIALS=('nominal_left','nominal_right','lower_friction_left')
IDENTITIES={'launch.json':'be4267ab90f122e18ca8ef8f260cacc2150cdb6aed57f568b6170863c41e0fbf',
    'result.json':'b5ac72bf6ad35df56d99f0dc9be00ae19208d091e805aeaea4501e8d0152cf98',
    'raw_return_audit.json':'2d25a7ce77d4c7c5cf3688f50c6a5f17d872243b10d4bf613554ebca6c97c099'}


def preflight():
    inputs={str((INPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(inputs)
    old=read_json(INPUT,'launch.json');result=read_json(INPUT,'result.json')
    for c in TRIALS:
        for n in ('policy_observations.json','command_tape.json','servo_decisions.json'):
            name=c+'/'+n;inputs[str((INPUT/name).relative_to(ROOT))]=result['artifact_sha256'][name]
    sources=discover_sources((PROTOCOL,'scripts/check_go2_pulse_timed_pairing_v1.py',
        'lewm/tests/test_pulse_timed_observation_pairing_development.py'),old['source_sha256'])
    verify_bindings(sources|inputs)
    return dict(source_sha256=sources,input_sha256=inputs,scope='exact audited metadata-only pulse target index',
                no_training=True,no_new_physics=True,no_split_or_generalization_claim=True,goal_achieved=False)


def condition(c):
    frames=read_json(INPUT/c,'policy_observations.json')['frames']
    tape=read_json(INPUT/c,'command_tape.json');rows=read_json(INPUT/c,'servo_decisions.json');windows=[]
    for row in rows:
        local=row['decision']['execution']['local_decision']
        event=None if local is None else local['diagnostic'].get('new_pulse')
        if event is None:continue
        if row['tick']!=row['observation_index']:raise ValueError('departure frame/tick identity mismatch')
        w=pulse_window(frames,tape,departure_tick=row['tick'],departure_ns=row['decision']['decision_ns'],
                       command=tuple(local['requested_command']),pulse_ticks=event['ticks'])
        windows.append(w|dict(condition=c,action_index=event['action_index']))
    reasons=Counter(t['reason'] for w in windows for t in w['targets'])
    return windows,dict(declared_pulses=len(windows),history_ready=sum(w['history_ready'] for w in windows),
        target_slots=len(windows)*8,target_status_counts=dict(reasons),
        exact_minimum_brake_endpoints=sum(w['targets'][4]['future_valid'] for w in windows),
        action_duration_counts=dict(Counter(str((w['command'],w['pulse_ticks'])) for w in windows)),
        latent_target_observation_gate='actual raw RGB plus verified requested-command prefix, not visual tracking success',
        native_motion_contact_labels_added=False,model_trained=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive recorded pairing diagnostic')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);all_windows=[];reports={}
    try:
        for c in TRIALS:
            windows,reports[c]=condition(c);all_windows.extend(windows)
            print('PULSE_PAIRING',c,reports[c],flush=True)
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'windows.json',all_windows)
        write_json(OUTPUT/'result.json',dict(status='RECORDED_PULSE_PAIRING_COMPLETE',conditions=reports,
            windows_sha256=digest(OUTPUT/'windows.json'),window_count=len(all_windows),goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_PAIRING_DIAGNOSTIC_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
