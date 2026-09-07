"""Terminal-only eight-run accounting; repeats never become independent layouts."""
import json
import cv2
import torch
from scripts.run_go2_ordered_union_dynamic_sensor_pilot_v1 import OUTPUT,validate_launch
from scripts.ordered_dynamic_pilot_development import RUNS,TRIALS,PROTOCOL,artifacts,compare_streams
from scripts.ordered_dynamic_audit_development import audit_dynamic_condition
from scripts.independent_layout_batch_development import load_inventory
from scripts.audit_go2_independent_pulse_context_pilot_v1 import compare_prefixes
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json


def comparisons(evidence):
    missing=dict(status='MISSING_DEPARTURE_PREFIX',native_samples=0,frames=0,sha256={})
    pairs=[]
    for repeat in (0,1):
        for a,b in ((TRIALS[0],TRIALS[1]),(TRIALS[2],TRIALS[3])):
            left,right=f'repeat_{repeat}_{a}',f'repeat_{repeat}_{b}'
            row=compare_prefixes(evidence.get(left,{}).get('prefix',missing),evidence.get(right,{}).get('prefix',missing))
            pairs.append(dict(reference=left,candidate=right,**row))
    repeats=[]
    for c in TRIALS:
        row=compare_streams(evidence.get('repeat_0_'+c,{}).get('whole_stream'),evidence.get('repeat_1_'+c,{}).get('whole_stream'))
        repeats.append(dict(trial=c,**row))
    return pairs,repeats


def load_terminal(inv):
    launch=read_json(OUTPUT,'launch.json');verify_ordered_launch(launch);validate_launch(launch,inv)
    terminals=[n for n in ('result.json','failure.json') if (OUTPUT/n).is_file()]
    if len(terminals)!=1:raise ValueError('exactly one terminal record required')
    name=terminals[0];terminal=read_json(OUTPUT,name);count=len(terminal['commits']);checked=len(terminal['prechecks'])
    ids=[r for r,_,_ in RUNS]
    assert terminal['planned_runs']==ids and list(terminal['conditions'])==ids[:count] and 0<=count<=8
    assert list(terminal['commits'])==[f'episode_{i:03d}_commit.json' for i in range(count)]
    assert max(0,count-1)<=checked<=count
    assert list(terminal['prechecks'])==[f'episode_{i:03d}_raw_precheck.json' for i in range(checked)]
    if name=='result.json':
        assert terminal['status']=='ORDERED_DYNAMIC_SENSOR_COLLECTION_COMPLETE'
        assert count==checked==8 and terminal['uncommitted_run'] is None
    else:
        assert terminal['status']=='TERMINAL_ORDERED_DYNAMIC_COLLECTION_FAILURE'
        if terminal['uncommitted_run'] is not None:assert count<8 and terminal['uncommitted_run']==ids[count]
    bindings={n:digest(OUTPUT/n) for n in ('launch.json',name)}|terminal['commits']|terminal['prechecks']
    verify_artifacts(OUTPUT,bindings);commits={};used=(OUTPUT/'launch.json').stat().st_size
    for i,(run,trial,_) in enumerate(RUNS[:count]):
        leaf=f'episode_{i:03d}_commit.json';commit=read_json(OUTPUT,leaf)
        assert commit['run']==run and commit['trial']==trial and commit['result']==terminal['conditions'][run]
        expected={run+'/'+n for n in artifacts(inv.specification(trial),commit['result'])}
        present=set(commit['artifact_sha256']);missing=set(commit['absent_expected_artifacts'])
        assert not present&missing and present|missing==expected
        assert len(missing)==len(commit['absent_expected_artifacts'])
        verify_artifacts(OUTPUT,commit['artifact_sha256'])
        assert commit['artifact_bytes']==sum((OUTPUT/n).stat().st_size for n in present)
        used+=commit['artifact_bytes']+(OUTPUT/leaf).stat().st_size
        bindings.update(commit['artifact_sha256']);commits[run]=commit
        if i<checked:used+=(OUTPUT/f'episode_{i:03d}_raw_precheck.json').stat().st_size
    assert used==terminal['committed_bytes']
    return launch,terminal,commits,bindings


def main():
    if (OUTPUT/'dynamic_audit_launch.json').exists():raise ValueError('exclusive terminal audit; no retry/resume')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inv=load_inventory();launch,terminal,commits,bindings=load_terminal(inv)
    write_json(OUTPUT/'dynamic_audit_launch.json',dict(source_sha256=launch['source_sha256'],input_sha256=bindings,model_training=False))
    evidence={};statuses={};outputs={}
    try:
        for i,(run,trial,_) in enumerate(RUNS):
            if run not in commits:
                statuses[run]='ATTEMPTED_UNCOMMITTED' if terminal['uncommitted_run']==run else 'NOT_ATTEMPTED';continue
            c=commits[run]
            if c['absent_expected_artifacts']:statuses[run]='INCOMPLETE_ARTIFACTS';continue
            row=audit_dynamic_condition(OUTPUT/run,inv.specification(trial),c['result'],launch['source_sha256'][PROTOCOL])
            precheck=f'episode_{i:03d}_raw_precheck.json'
            if precheck in terminal['prechecks']:assert json.loads(json.dumps(row))==read_json(OUTPUT,precheck)
            evidence[run]=row;statuses[run]='RAW_SENSOR_DIAGNOSTIC_AUDITED'
            leaf=run+'_evaluation.json';write_json(OUTPUT/leaf,row);outputs[leaf]=digest(OUTPUT/leaf)
            print('DYNAMIC_AUDIT',i+1,run,row['report']['physical_visibility_pass'],flush=True)
        pairs,repeats=comparisons(evidence)
        verify_ordered_launch(launch);verify_artifacts(OUTPUT,bindings|outputs)
        result=dict(status='ORDERED_DYNAMIC_AVAILABLE_EVIDENCE_AUDITED',expected_runs=8,committed_runs=len(commits),
            audited_runs=len(evidence),run_statuses=statuses,output_sha256=outputs,prefix_comparisons=pairs,repeat_comparisons=repeats,
            exact_candidate_prefix_matches=sum(p['matched'] for p in pairs),
            exact_complete_repeats=sum(p['complete_repeatability'] for p in repeats),
            strict_visibility_failed_runs=[r for r,e in evidence.items() if e['report']['physical_visibility_pass'] is False],
            positive_contact_targets=sum(e['report']['target_contact_positive'] for e in evidence.values()),
            physics_samples=sum(e['report']['physics_samples'] for e in evidence.values()),
            rgbd_frames=sum(e['report']['paired_frames'] for e in evidence.values()),
            independent_layouts=1,training_eligibility_granted=False,boundary_pixels_certified=False,
            navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'dynamic_audit.json',result)
        print('DYNAMIC_AUDIT_COMPLETE',result['exact_candidate_prefix_matches'],result['exact_complete_repeats'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'dynamic_audit_failure.json',dict(status='TERMINAL_DYNAMIC_AUDIT_FAILURE',reason=repr(error),completed_runs=list(evidence)))
        raise


if __name__=='__main__':main()
