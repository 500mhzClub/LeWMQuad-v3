"""Audit complete or infrastructure-stopped batch without dropping attempt counts."""
import argparse
import json
import cv2
import torch
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from scripts.independent_layout_batch_development import (
    BATCHES,PROTOCOL,load_inventory,output_root,validate_launch,episode_artifacts)
from scripts.independent_layout_collection_audit_development import audit_condition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import compare_prefixes
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import verify_artifacts


def load_terminal_batch(output,inventory,batch):
    launch=read_json(output,'launch.json');verify(launch);validate_launch(launch,inventory,batch)
    terminal_names=[n for n in ('result.json','failure.json') if (output/n).is_file()]
    if len(terminal_names)!=1:raise ValueError('exactly one terminal batch record required; do not audit a live batch')
    terminal_name=terminal_names[0];terminal=read_json(output,terminal_name);ids=inventory.episode_ids(batch)
    assert terminal['batch']==batch and terminal['planned_trials']==list(ids)
    if terminal_name=='result.json':
        assert terminal['status']=='LAYOUT_COLLECTION_COMPLETE' and len(terminal['commits'])==len(ids)
        assert terminal['uncommitted_trial'] is None
    else:assert terminal['status']=='TERMINAL_LAYOUT_COLLECTION_FAILURE'
    count=len(terminal['commits']);assert 0<=count<=len(ids)
    assert list(terminal['conditions'])==list(ids[:count])
    assert list(terminal['commits'])==[f'episode_{i:03d}_commit.json' for i in range(count)]
    if terminal['uncommitted_trial'] is not None:
        assert count<len(ids) and terminal['uncommitted_trial']==ids[count]
    checked=len(terminal['prechecks']);assert max(0,count-1)<=checked<=count
    assert list(terminal['prechecks'])==[f'episode_{i:03d}_raw_precheck.json' for i in range(checked)]
    if terminal_name=='result.json':assert checked==count
    bindings={p:digest(output/p) for p in ('launch.json',terminal_name)}|terminal['commits']|terminal['prechecks']
    verify_artifacts(output,bindings);committed={};total=(output/'launch.json').stat().st_size
    for i,c in enumerate(ids[:count]):
        leaf=f'episode_{i:03d}_commit.json';commit=read_json(output,leaf)
        assert commit['trial']==c and commit['result']==terminal['conditions'][c]
        expected={c+'/'+p for p in episode_artifacts(inventory.specification(c),commit['result'])}
        present=set(commit['artifact_sha256']);missing=set(commit['absent_expected_artifacts'])
        assert not present&missing and present|missing==expected
        assert len(missing)==len(commit['absent_expected_artifacts'])
        verify_artifacts(output,commit['artifact_sha256'])
        assert commit['artifact_bytes']==sum((output/p).stat().st_size for p in present)
        total+=commit['artifact_bytes']+(output/leaf).stat().st_size
        if i<checked:
            precheck=f'episode_{i:03d}_raw_precheck.json';total+=(output/precheck).stat().st_size
            commit['raw_precheck']=read_json(output,precheck)
        bindings.update(commit['artifact_sha256']);committed[c]=commit
    assert terminal['committed_bytes']==total
    return launch,terminal,committed,bindings


def prefix_comparisons(inventory,batch,prefixes):
    ids=inventory.episode_ids(batch);groups={};pairs={}
    missing=dict(status='NO_COMMITTED_EPISODE',native_samples=0,frames=0,sha256={})
    for c in ids:
        e=inventory.episodes[c];key=tuple(e[k] for k in ('layout_id','context_kind','history_kind','support'))
        groups.setdefault(key,{})[e['action_index']]=c
    for group in groups.values():
        assert set(group)==set(range(6));reference=group[0]
        for a,c in group.items():
            pairs[c]=compare_prefixes(prefixes.get(reference,missing),prefixes.get(c,missing))|dict(reference=reference,is_reference=a==0)
    return pairs


def run_audit(batch):
    output=output_root(batch);inventory=load_inventory()
    if (output/'layout_audit_launch.json').exists():raise ValueError('exclusive layout audit; preserve prior result/failure')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch,terminal,committed,bindings=load_terminal_batch(output,inventory,batch)
    write_json(output/'layout_audit_launch.json',dict(batch=batch,source_sha256=launch['source_sha256'],
        artifact_sha256=bindings,complete_collection=terminal['status']=='LAYOUT_COLLECTION_COMPLETE',model_training=False))
    reports={};prefixes={};windows=[];targets=[];outputs=[]
    try:
        for c in inventory.episode_ids(batch):
            if c not in committed:
                reports[c]=dict(status='ATTEMPTED_UNCOMMITTED' if c==terminal['uncommitted_trial'] else 'NOT_ATTEMPTED',
                    recorded_sensor_reconstruction_pass=False,departure_present=False)
                continue
            commit=committed[c]
            if commit['absent_expected_artifacts']:
                reports[c]=dict(status='INCOMPLETE_ARTIFACTS_NOT_AUDITED',missing_artifacts=commit['absent_expected_artifacts'],
                    recorded_sensor_reconstruction_pass=False,departure_present=False)
                continue
            report,prefix,window,labels=audit_condition(output/c,inventory.specification(c),commit['result'],launch['source_sha256'][PROTOCOL])
            if 'raw_precheck' in commit:
                assert json.loads(json.dumps(dict(report=report,prefix=prefix,window=window,targets=labels)))==commit['raw_precheck']
            usable=report['physical_visibility_pass'] is True
            reports[c]=report|dict(status='RAW_SENSOR_VISIBILITY_FAILURE' if report['physical_visibility_pass'] is False else 'RAW_EPISODE_AUDITED');prefixes[c]=prefix
            if window is not None and usable:windows.append(window);targets.append(labels)
            leaf=c+'_layout_evaluation.json';write_json(output/leaf,reports[c]);outputs.append(leaf)
            print('LAYOUT_AUDIT',batch,c,{k:report[k] for k in ('setup_admitted','departure_present','target_motion_valid','target_contact_positive')},flush=True)
        pairs=prefix_comparisons(inventory,batch,prefixes)
        roles={w['condition']:dict(layout_id=inventory.episodes[w['condition']]['layout_id'],role=inventory.episodes[w['condition']]['role']) for w in windows}
        dataset=PulseTimedDataset(windows,targets,roles) if windows else None;integrated=[]
        if dataset is not None:
            for i,w in enumerate(windows):
                if not w['history_ready']:continue
                sample=dataset.sample(i,{w['condition']:IntentReturnRGBDReplay(output/w['condition'])})
                integrated.append(dict(condition=w['condition'],input_fields=sorted(sample['inputs']),motion_valid=int(sample['targets']['motion_valid'].sum())))
        for name,value in (('layout_windows.json',windows),('layout_targets.json',targets),('layout_prefix_witnesses.json',prefixes)):
            write_json(output/name,value);outputs.append(name)
        verify(launch);verify_artifacts(output,bindings)
        audited=[r for r in reports.values() if r['recorded_sensor_reconstruction_pass']]
        write_json(output/'layout_audit.json',dict(status='LAYOUT_AVAILABLE_RECORDED_EVIDENCE_AUDITED',batch=batch,
            collection_complete=terminal['status']=='LAYOUT_COLLECTION_COMPLETE',expected_trials=120,committed_trials=len(committed),
            audited_trials=len(audited),conditions=reports,prefix_comparisons=pairs,departures=len(windows),
            setup_admitted=sum(r['setup_admitted'] for r in audited),schedule_completions=sum(r['schedule_complete'] for r in audited),
            materialized_samples=integrated,role=launch['role'],action_coverage=dataset.coverage(launch['role']) if dataset else {},
            exact_nonreference_prefix_matches=sum(p['matched'] for p in pairs.values() if not p['is_reference']),
            contact_positive_targets=sum(r['target_contact_positive'] for r in audited),
            visibility_failed_trials=[c for c,r in reports.items() if r.get('physical_visibility_pass') is False],
            output_sha256={p:digest(output/p) for p in outputs},independent_layouts=1,model_trained=False,
            final_evaluation=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(output/'layout_audit_failure.json',dict(status='TERMINAL_LAYOUT_AUDIT_FAILURE',reason=repr(error),completed_reports=list(reports)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',choices=BATCHES,required=True)
    run_audit(parser.parse_args().batch)


if __name__=='__main__':main()
