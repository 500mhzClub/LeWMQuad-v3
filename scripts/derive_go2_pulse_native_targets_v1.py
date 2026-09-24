"""Bound target-only native labels for the already indexed old pulse windows."""
from collections import Counter
from lewm.pulse_native_targets_development import PulseNativeTargets
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,TRIALS,OUTPUT as PAIRING
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings

OUTPUT=ROOT/'.generated/go2_pulse_native_targets_v1_attempt_001'
PROTOCOL='docs/go2_pulse_native_targets_v1_2026-09-06.md'
IDENTITIES={'launch.json':'78e212aadea9b66cbad6048f26d70cd4e54d9a6659a229f9eda448feff1a9370',
 'result.json':'e8786cbda4b821d79bf8554311423224562e04ddf0bcf78fff960fbd9c581e84',
 'windows.json':'7798f2f8b60491475828d900c9df75656bf5ad08eacc0d51850615b53c61b900'}


def preflight():
    ids={str((PAIRING/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(ids)
    old=read_json(PAIRING,'launch.json');inputs=old['input_sha256']|ids
    verify_bindings(inputs|old['source_sha256'])
    collection=read_json(INPUT,'result.json')
    for c in TRIALS:
        name=c+'/physics_trace.npz';inputs[str((INPUT/name).relative_to(ROOT))]=collection['artifact_sha256'][name]
    sources=discover_sources((PROTOCOL,'scripts/derive_go2_pulse_native_targets_v1.py',
        'lewm/tests/test_pulse_native_targets_development.py'),old['source_sha256'])
    verify_bindings(sources|inputs)
    return dict(source_sha256=sources,input_sha256=inputs,target_only=True,model_trained=False,new_physics=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive target derivation')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);all_rows=[];reports={}
    try:
        windows=read_json(PAIRING,'windows.json')
        for c in TRIALS:
            model=PulseNativeTargets(read_npz(INPUT/c,'physics_trace.npz'));counts=Counter();rows=[]
            for w in windows:
                if w['condition']!=c:continue
                result=model.labels(w);targets=[]
                for i,t in enumerate(w['targets']):
                    mv=bool(result['motion_valid'][i]);cv=bool(result['contact_valid'][i])
                    contact=float(result['contact'][i]) if cv else None
                    counts.update(dict(motion_valid=int(mv),contact_valid=int(cv),contact_positive=int(contact==1),
                        image_without_motion=int(t['future_valid'] and not mv),motion_without_image=int(mv and not t['future_valid'])))
                    targets.append(dict(offset_ns=t['offset_ns'],motion_valid=mv,contact_valid=cv,
                        motion=result['motion'][i].tolist() if mv else None,contact=contact,
                        image_target_valid=t['future_valid'],status=result['accounting']['target_status'][i]))
                rows.append(dict(condition=c,departure_tick=w['departure_tick'],decision_ns=w['decision_ns'],
                    action_index=w['action_index'],targets=targets,accounting=result['accounting'],
                    label_definition=result['label_definition'],target_only=True))
            reports[c]=dict(windows=len(rows),**counts);all_rows.extend(rows);print('NATIVE_TARGETS',c,reports[c],flush=True)
        if len(all_rows)!=len(windows):raise ValueError('all original windows must be retained')
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'targets.json',all_rows)
        write_json(OUTPUT/'result.json',dict(status='PULSE_NATIVE_TARGET_DERIVATION_COMPLETE',conditions=reports,
            targets_sha256=digest(OUTPUT/'targets.json'),windows=len(all_rows),model_trained=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_NATIVE_TARGET_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
