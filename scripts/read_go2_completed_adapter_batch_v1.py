"""Reconstruct six final outcomes from bound audits and native contact arrays.

This is a bounded completed-batch readout, not a repeat of raw RGBD/model
auditing, all generated artifact hashing, or training/input admission.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from scripts.reached_frontier_native_inputs_development import batch, require_batch, BATCH_LAUNCH
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

SOURCE='scripts/read_go2_completed_adapter_batch_v1.py'
RESULT_SHA='9715a916b81d9a70edf2d26e4f3d7e952823d3e6b618670c46ab2aa0429ff416'
BOOT='1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
OWNER=dict(pid=2659758,created=1789030196.29,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',batch.SOURCE,
    '--correction-wait-result-sha256','4bf13d2e00fb318fa836d02bad93784fbb1a9c5ccef8792bd19dcfd503f657a0',
    '--native-result-sha256','330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'])
OUTPUT=ROOT/'docs/go2_completed_adapter_batch_readout_verification_2026-09-11.json'


def main():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=BOOT or owner_live(OWNER):
        raise ValueError('original batch owner must have ended on the recorded boot')
    if (batch.OUTPUT/'failure.json').exists() or (batch.OUTPUT/'failure.json').is_symlink():
        raise ValueError('original batch failure may not be replaced')
    selected={'result.json':RESULT_SHA,'launch.json':BATCH_LAUNCH}
    verify_artifacts(batch.OUTPUT,selected)
    result=read_json(batch.OUTPUT,'result.json');launch=read_json(batch.OUTPUT,'launch.json')
    if result['source_sha256']!=launch['source_sha256']:
        raise ValueError('unchanged original source closure required')
    sources=discover_sources((SOURCE,),launch['source_sha256']);verify(sources)
    if (launch['planned_cases']!=[list(c) for c in batch.CASES]
            or launch['direct_is_nonpredictive_baseline'] is not False
            or launch['model_rgb_ablation_keeps_controller_rgbd'] is not True
            or result['correction_wait_result_sha256']!=launch['input_admission']['correction_wait_result_sha256']
            or result['predecessor_native_result_sha256']!=launch['input_admission']['native_result_sha256']
            or result['adapter_prefix_result_sha256']!=launch['adapter_prefix_result_sha256']):
        raise ValueError('exact original treatment definitions and input links required')
    ids=result['artifact_sha256']
    for case in batch.CASES:
        for suffix in ('_worker_terminal.json','_audit.json','_startup_comparison.json',
                       '_readout.json','_parent_completion.json','_worker.log',
                       '/result.json','/physics_trace.npz'):
            name=case[0]+suffix;selected[name]=ids[name]
    selected['resource_monitor.jsonl']=ids['resource_monitor.jsonl']
    verify_artifacts(batch.OUTPUT,selected)
    records=[];audits=[];startups=[];readouts=[];compact=[]
    reconstructed_ids={'launch.json':BATCH_LAUNCH,'resource_monitor.jsonl':ids['resource_monitor.jsonl']}
    for case in batch.CASES:
        name=case[0]
        record=read_json(batch.OUTPUT,name+'_worker_terminal.json')
        audit=read_json(batch.OUTPUT,name+'_audit.json')
        startup=read_json(batch.OUTPUT,name+'_startup_comparison.json')
        readout=read_json(batch.OUTPUT,name+'_readout.json')
        collection=read_json(batch.OUTPUT,name+'/result.json')
        parent=read_json(batch.OUTPUT,name+'_parent_completion.json')
        if (collection!=record['collection'] or parent!=dict(case=name,
                worker_terminal_sha256=selected[name+'_worker_terminal.json'],
                verified_round_trip=record['verified_round_trip'],scientific_success_required=False)
                or record['worker_log_sha256']!=selected[name+'_worker.log']
                or record['head']!=('direct_outcomes' if case[3]=='direct' else 'rollout_outcomes')):
            raise ValueError('original completed collection, parent, worker log and head required')
        with np.load(batch.OUTPUT/name/'physics_trace.npz',allow_pickle=False) as saved:
            contact=saved['physics_contact']
            if len(contact)!=collection['physics_samples']:
                raise ValueError('complete native contact population required')
            rebuilt=batch.case_readout(audit,collection,contact)
        if fingerprint(rebuilt)!=fingerprint(readout):
            raise ValueError('full readout must reconstruct from audit, collection and native contacts')
        batch.require_case(case,record,audit)
        for n,h in record['artifact_sha256'].items():
            if n in reconstructed_ids and reconstructed_ids[n]!=h:
                raise ValueError('conflicting original artifact identity')
            reconstructed_ids[n]=h
        for suffix in ('_worker_terminal.json','_worker.log','_parent_completion.json'):
            reconstructed_ids[name+suffix]=selected[name+suffix]
        records.append(record);audits.append(audit);startups.append(startup);readouts.append(readout)
        compact.append(dict(case=name,model_name=case[4],variant=case[2],condition=case[3],
            model_state_sha256=record['model_state_sha256'],observations=collection['rgbd_frames'],
            command_ticks=collection['command_ticks'],**readout))
    if reconstructed_ids!=ids:
        raise ValueError('exact final artifact roster must equal the six completed worker rosters')
    completion=require_batch(result,launch,records,audits,startups,readouts)
    verify(sources);verify_artifacts(batch.OUTPUT,selected)
    if owner_live(OWNER):raise ValueError('original owner unexpectedly live')
    write_json(OUTPUT,dict(status='COMPLETED_ADAPTER_BATCH_BOUND_READOUT_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(),batch_result_sha256=RESULT_SHA,
        batch_launch_sha256=BATCH_LAUNCH,source_sha256=sources,source_count=len(sources),
        original_source_count=len(launch['source_sha256']),original_owner=OWNER,original_owner_ended=True,
        rehashed_artifact_sha256=selected,rehashed_artifact_count=len(selected),
        final_artifact_roster_count=len(ids),final_roster_reconstructed_from_workers=True,
        six_complete_readouts_reconstructed=True,completion=completion,cases=compact,
        raw_sensor_model_audits_reexecuted=False,full_generated_artifact_roster_rehashed=False,
        full_training_ancestry_or_runtime_input_admission_reexecuted=False,
        source_of_sensor_model_visibility_evidence='authenticated original completed raw audits',
        direct_head_is_predictive=True,model_rgb_ablation_preserves_rgbd_frontend=True,
        independent_layout_execution=False,new_native_execution=False,model_training=False,
        model_or_policy_selected=False,online_planning_advantage_established=False,
        persistent_memory_advantage_established=False,jepa_advantage_established=False,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
    print('COMPLETED_ADAPTER_BATCH_READOUT_VERIFIED',digest(OUTPUT),len(sources),len(selected),flush=True)
    for case in compact:
        print(case['model_name'],case['outbound']['distinct_open_edges'],
              len(case['native_arrival_windows']),case['verified_round_trip'],
              case['strict_physical_visibility_pass'],case['observation_and_control']['median_ms'],flush=True)


if __name__=='__main__':main()
