"""Authenticate the complete prospective height-union audit and tensor interface.

No physics, fitting, outcome-based exclusions, new labels, or depth qualification.
"""
import argparse
import json
from lewm.geometry_progress_near_field_development import TRIALS,measurement_gate
from lewm.geometry_progress_learning_sample_development import materialize
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.read_go2_geometry_progress_science_v1 import summary as native_summary
from scripts.run_go2_geometry_progress_height_union_v1 import OUTPUT as INPUT
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_geometry_progress_height_union_science_v1_attempt_001'


def summarize(reports,audited_gate):
    gate=measurement_gate(reports)
    if gate!=audited_gate:raise ValueError('complete prospective measurement gate mismatch')
    result=native_summary(reports)
    # Preserve the old strict criterion explicitly; the new prospective contract
    # also checks clipped opaque surfaces and stable raster interiors.
    result['legacy_strict_depth_and_design_criterion']=result.pop('design_and_measurement_gate_pass')
    result['status']='GEOMETRY_PROGRESS_HEIGHT_UNION_SCIENTIFIC_READOUT'
    result['prospective_gate']=gate
    result['design_and_measurement_gate_pass']=gate['prediction_design_and_measurement_gate_pass']
    result['measurement_accounting']=dict(
        footprint_frames=sum(len(r['footprint_checks']) for r in reports),
        stable_interior_failures=sum(not f['stable_interior_metric_pass'] for r in reports for f in r['footprint_checks']),
        near_occlusion_failures=sum(f['near_occlusion_failure'] for r in reports for f in r['footprint_checks']),
        strict_failed_cases=gate['strict_depth_failed_cases'],depth_navigation_qualified=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-sha256',required=True)
    args=parser.parse_args();validate_root(INPUT);validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive readout; no retry/resume')
    ids={'near_field_audit.json':args.audit_sha256};verify_artifacts(INPUT,ids)
    audit=read_json(INPUT,'near_field_audit.json')
    if (audit['status']!='PROSPECTIVE_GEOMETRY_PROGRESS_NEAR_FIELD_AUDIT_COMPLETE'
            or audit['audited_episodes']!=24 or audit['planned_episodes']!=24):
        raise ValueError('complete prospective audit required')
    ids|=audit['collection_sha256']|audit['output_sha256'];verify_artifacts(INPUT,ids)
    collection=read_json(INPUT,'result.json');launch=read_json(INPUT,'launch.json');verify(launch)
    ids|=collection['artifact_sha256'];verify_artifacts(INPUT,ids)
    sources=discover_sources(('scripts/read_go2_geometry_progress_height_union_science_v1.py',
        'lewm/tests/test_geometry_progress_height_union_science_development.py'),launch['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition)
    reports=[read_json(INPUT,c+'_near_field_audit.json') for c in TRIALS]
    result=summarize(reports,audit['gate'])
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',dict(input_sha256=ids,source_sha256=sources,
        scope='complete audited-cohort aggregation and existing tensor interface; no fitting'))
    try:
        samples=[]
        for r in reports:
            if r['targets'] is None:
                samples.append(dict(trial=r['trial'],materialized=False,reason='MISSING_DEPARTURE'));continue
            sample=materialize(IntentReturnRGBDReplay(INPUT/r['trial']),r);t=sample['targets']
            history=sample['inputs']['observation_history']
            samples.append(dict(trial=r['trial'],materialized=True,
                input_fields=list(sample['inputs']),history_shapes={k:list(v.shape) for k,v in history.items()},
                history_sha256={k:fingerprint(v.numpy()) for k,v in history.items()},
                action_shape=list(sample['inputs']['known_action_blocks'].shape),
                motion_valid=int(t['motion_valid'].sum()),future_valid=int(t['future_valid'].sum()),
                contact_valid=int(t['contact_valid'].sum()),contact_positive=int((t['contact']==1).sum())))
        ready=[s for s in samples if s['materialized']]
        result['learning_interface_materialization']=samples
        result['model_context_identity_counts']=dict(materialized_episodes=len(ready),
            distinct_rgb_histories=len({s['history_sha256']['rgb'] for s in ready}),
            distinct_body_histories=len({s['history_sha256']['body'] for s in ready}),
            distinct_control_histories=len({s['history_sha256']['control'] for s in ready}),
            distinct_non_rgb_history_pairs=len({(s['history_sha256']['body'],s['history_sha256']['control']) for s in ready}))
        verify(definition);verify_artifacts(INPUT,ids)
        result|=dict(collection_root=str(INPUT),audit_sha256=args.audit_sha256,
            collection_sha256=audit['collection_sha256'],source_sha256=sources,
            prefix_comparisons=audit['prefix_comparisons'],conditions=audit['conditions'],
            artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')})
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:result[k] for k in ('status','episodes','successful_progress',
            'contact_episodes','design_and_measurement_gate_pass','measurement_accounting',
            'target_accounting','model_context_identity_counts')},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SCIENCE_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
