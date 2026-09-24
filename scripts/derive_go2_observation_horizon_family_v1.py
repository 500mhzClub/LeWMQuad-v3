"""Complete new target population from two authenticated recorded cohorts."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.observation_horizon_targets_development import derive,verify_half_second_overlap
from scripts.augmented_family_switch_fit_inputs_development import authenticate,stream,SWITCH_INPUT
from scripts.family_transition_fit_inputs_development import INPUT as FAMILY_INPUT
from scripts.read_go2_observation_replan_goal_probe_v1 import OUTPUT as DIAGNOSTIC
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_observation_horizon_family_targets_v1_attempt_001'
PROTOCOL='docs/go2_observation_horizon_family_targets_v1_2026-09-08.md'
SWITCH_CHECK_SHA='2a699ba2a37ce26565324c4e7dbf4f97b957f952c709c470fcf3fb4aaedf2083'
DIAGNOSTIC_SHA='be2c24798bca35111c62b3795c42c8eb14783405b7b1e6afe3b2f7a1c59f554d'


def summarize(rows):
    reports=[]
    for source in ('family','switch'):
        for role in ('train','geometry_transfer'):
            selected=[r for r in rows if r['source']==source and r['data_role']==role]
            targets=[t for r in selected if r['available'] for t in r['targets']]
            reports.append(dict(source=source,role=role,planned_contexts=len(selected),
                available_contexts=sum(r['available'] for r in selected),unavailable_contexts=sum(not r['available'] for r in selected),
                target_slots=len(targets),in_plan=sum(t['in_plan'] for t in targets),
                motion_targets=sum(t['motion_valid'] for t in targets),
                contact_targets=sum(t['contact_valid'] for t in targets),
                contact_positives=sum(t['contact_valid'] and t['contact']==1. for t in targets),
                future_observations_available=sum(t['future_image_valid'] for t in targets),
                first_100ms_motion_targets=sum(r['targets'][0]['motion_valid'] for r in selected if r['available'])))
    return reports


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive short-horizon target derivation')
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('derivation resource allowance unavailable')
    definition,checked,schedules=authenticate(SWITCH_CHECK_SHA);data=stream(SWITCH_CHECK_SHA)
    view=data.view
    assert len(view.rows)==912 and len(view.indices('train'))==408 and len(view.indices('geometry_transfer'))==420
    verify_artifacts(DIAGNOSTIC,{'result.json':DIAGNOSTIC_SHA});diagnostic=read_json(DIAGNOSTIC,'result.json')
    assert diagnostic['status']=='OBSERVATION_REPLAN_GOAL_READOUT_COMPLETE'
    diagnostic_ids={'result.json':DIAGNOSTIC_SHA,'launch.json':diagnostic['launch_sha256']}
    verify_artifacts(DIAGNOSTIC,diagnostic_ids);verify(read_json(DIAGNOSTIC,'launch.json'))
    sources=discover_sources((PROTOCOL,'scripts/derive_go2_observation_horizon_family_v1.py',
        'lewm/observation_horizon_plan_development.py','lewm/tests/test_observation_horizon_targets_development.py',
        'docs/go2_observation_horizon_family_design_v1_2026-09-08.md',
        'docs/go2_observation_replan_goal_probe_result_2026-09-08.md',
        *diagnostic['source_sha256']),definition['source_sha256'])
    for name,h in diagnostic['source_sha256'].items():assert sources[name]==h,name
    root_bindings={str(FAMILY_INPUT):data.family.bindings,str(SWITCH_INPUT):data.switch.bindings}
    launch=definition|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        source_collection_artifact_sha256=root_bindings,switch_input_check_sha256=SWITCH_CHECK_SHA,
        cadence_probe_readout_sha256=diagnostic_ids,hardware=resources,workers=1,threads=1,
        concurrency_reason='240 recorded episodes; one trace resident at a time; no rendering, model forward or optimization',
        native_execution=False,model_training=False,native_labels_are_target_only=True,
        future_rgb_materialization=False,target_cadence_ns=100_000_000,maximum_horizon_ns=800_000_000)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('OBSERVATION_HORIZON_TARGETS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        rows=[None]*len(view.rows);episodes=0
        for source,root in (('family',FAMILY_INPUT),('switch',SWITCH_INPUT)):
            trials=sorted({r['trial'] for r in view.rows if r['source']==source})
            for trial in trials:
                names=[trial+'/'+n for n in ('physics_trace.npz','camera_audit.json')]
                ids={n:root_bindings[str(root)][n] for n in names};verify_artifacts(root,ids)
                with np.load(root/trial/'physics_trace.npz',allow_pickle=False) as z:
                    raw={k:z[k] for k in ('timestamp_s','base_pose_world','physics_contact','requested_command')}
                cameras=read_json(root/trial,'camera_audit.json')
                for i,row in enumerate(view.rows):
                    if row['source']!=source or row['trial']!=trial:continue
                    offset=row['offset_ticks'] if source=='family' else 0
                    frame=3+offset if source=='family' else 13
                    commands=candidate_commands(row['action'])[offset:offset+8]
                    labels=derive(raw,cameras,frame=frame,commands=commands)
                    if labels['available']!=row['available']:raise ValueError('original context availability changed: '+row['sample_id'])
                    if labels['available']:verify_half_second_overlap(labels,row['targets'])
                    elif source=='family' and labels['reason']!=row['reason']:raise ValueError('original missing-context reason changed')
                    rows[i]=deepcopy(row)|dict(targets=labels['targets'],observation_horizon_receipt={k:v for k,v in labels.items() if k!='targets'},
                        shared_half_second_native_target_exact=labels['available'])
                verify_artifacts(root,ids);episodes+=1
        assert episodes==240 and all(r is not None for r in rows)
        assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in view.rows]
        assert sum(r['available'] for r in rows)==828
        reports=summarize(rows)
        write_json(OUTPUT/'windows.json',rows);write_json(OUTPUT/'training_schedules.json',schedules)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json','windows.json','training_schedules.json')}
        verify(launch);authenticate(SWITCH_CHECK_SHA)
        for root,ids in root_bindings.items():verify_artifacts(Path(root),ids)
        verify_artifacts(DIAGNOSTIC,diagnostic_ids);verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='OBSERVATION_HORIZON_FAMILY_TARGETS_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,summary=reports,episodes=episodes,context_slots=912,
            available_contexts=828,unavailable_contexts=84,all_shared_half_second_native_targets_exact=True,
            original_context_availability_and_roles_preserved=True,wall_s=time.perf_counter()-started,
            hardware_after=hardware(),native_execution=False,model_training=False,future_rgb_materialization=False,
            native_labels_are_target_only=True,independent_maze_evaluation=False,navigation_qualified=False,goal_achieved=False))
        print('OBSERVATION_HORIZON_TARGETS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_OBSERVATION_HORIZON_TARGET_DERIVATION_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
