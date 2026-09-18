"""Receipt-bound pilot readout and native-centre plots; no new simulation or fit."""
import argparse
import json
from collections import Counter
import numpy as np
from lewm.geometry_progress_pilot_development import (ACTIONS, GEOMETRIES, APPEARANCES, TRIALS, GOAL_BODY_XY,
    assignments, specification, progress_outcome, panel_informativeness)
from lewm.geometry_progress_learning_sample_development import materialize
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_geometry_progress_pilot_v1 import OUTPUT as INPUT
from scripts.read_go2_geometry_progress_available_evidence_v1 import OUTPUT as RAW_READOUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_geometry_progress_science_v1_attempt_001'


def summary(reports):
    if [r['trial'] for r in reports] != list(TRIALS):
        raise ValueError('complete ordered raw-audit cohort required')
    for r in reports:
        if any(r[k] != v for k,v in assignments()[r['trial']].items()):
            raise ValueError('prospective assignment mismatch')
        if not r['raw_sensor_reconstruction_pass'] or not r['command_stop_replay_pass']:
            raise ValueError('completed raw sensor and execution replay required')
        o=r['outcome']
        expected=progress_outcome(r['terminal_displacement_departure_body_xy_m'],
            complete=o['complete_horizon'],disallowed_contact=o['contact'],
            physical_stop=o['physical_stop'],acquisition_stop=o['acquisition_stop'])
        if o != expected:raise ValueError('outcome differs from audited terminal displacement/stop')
    by_action={}
    for a in ACTIONS:
        rows=[r for r in reports if r['action']==a]
        by_action[a]=dict(episodes=len(rows),full_horizons=sum(r['outcome']['complete_horizon'] for r in rows),
            successful_progress=sum(r['outcome']['successful_progress'] for r in rows),
            contact_episodes=sum(r['outcome']['contact'] for r in rows),
            complete_horizon_progress_m=[r['outcome']['progress_m'] for r in rows if r['outcome']['complete_horizon']],
            stopped_prefix_progress_m=[r['outcome']['progress_m'] for r in rows if not r['outcome']['complete_horizon']],
            stop_reasons=dict(Counter(r['outcome']['physical_stop'] or r['outcome']['acquisition_stop']
                                     for r in rows if not r['outcome']['complete_horizon'])))
    targets=[t for r in reports if r['targets'] is not None for t in r['targets']['targets']]
    timing=np.asarray([x for r in reports for x in r['observation_and_control_wall_ms']],float)
    depth=[d for r in reports for d in r['depth_checks']]
    panel=panel_informativeness(reports)
    hard_sensor_pass=bool(depth and all(d['within1mm'] for d in depth))
    return dict(status='GEOMETRY_PROGRESS_PILOT_SCIENTIFIC_READOUT',episodes=len(reports),
        setup_admitted=sum(r['setup_admitted'] for r in reports),
        full_horizons=sum(r['outcome']['complete_horizon'] for r in reports),
        successful_progress=sum(r['outcome']['successful_progress'] for r in reports),
        contact_episodes=sum(r['outcome']['contact'] for r in reports),by_action=by_action,
        panel=panel,all_raw_depth_checks_within1mm=hard_sensor_pass,
        design_and_measurement_gate_pass=bool(panel['informative_for_next_dataset'] and hard_sensor_pass),
        target_accounting=dict(expected_slots=24*8,recorded_slots=len(targets),
            missing_departures=sum(r['targets'] is None for r in reports),
            motion_valid=sum(t['motion_valid'] for t in targets),
            future_image_valid=sum(t['future_image_valid'] for t in targets),
            contact_valid=sum(t['contact_valid'] for t in targets),
            contact_positive=sum(t['contact']==1. for t in targets),
            positive_with_missing_image=sum(t['contact']==1. and not t['future_image_valid'] for t in targets),
            positive_with_missing_motion=sum(t['contact']==1. and not t['motion_valid'] for t in targets)),
        raw_replay=dict(decisions=sum(r['decisions'] for r in reports),
            frames=sum(r['frames'] for r in reports),physics_samples=sum(r['physics_samples'] for r in reports),
            depth_frames=len(depth),maximum_depth_error_m=max((d['maximum_error_m'] for d in depth if d['maximum_error_m'] is not None),default=None),
            failed_depth_frames=[dict(trial=r['trial'],**d) for r in reports for d in r['depth_checks'] if not d['within1mm']]),
        observation_and_control_wall_ms=dict(count=len(timing),median=float(np.median(timing)) if len(timing) else None,
            maximum=float(timing.max()) if len(timing) else None,above100ms=int((timing>100).sum()),
            includes_shadow_observer=True,includes_physics_tick=False,real_time_qualified=False),
        independent_maze_evaluation_layouts=0,same_observation_counterfactual_claim=False,
        model_trained=False,rgb_benefit_established=False,navigation_qualified=False,goal_achieved=False)


def plot_paths(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    colors=dict(zip(ACTIONS,['#777777','#171717','#0072B2','#D55E00','#56B4E9','#E69F00']))
    fig,axes=plt.subplots(2,2,figsize=(10,9),sharex=True,sharey=True)
    for g,axrow in zip(GEOMETRIES,axes,strict=True):
        for seed,ax in zip(APPEARANCES,axrow,strict=True):
            rows=[r for r in reports if r['geometry']==g and r['appearance_seed']==seed]
            for b in specification(rows[0]['trial'])['geometry']['wall_boxes']:
                x,y,_=b['centre_xyz'];dx,dy,_=b['size_xyz']
                ax.add_patch(Rectangle((x-dx/2,y-dy/2),dx,dy,facecolor='#dddddd',edgecolor='#555555'))
            for r in rows:
                with np.load(INPUT/r['trial']/'physics_trace.npz',allow_pickle=False) as z:
                    path=z['base_pose_world'][899:,:2]
                    pose=z['base_pose_world'][899] if len(path) else None
                if not len(path):continue
                color=colors[r['action']]
                ax.plot(path[:,0],path[:,1],color=color,linewidth=1.5)
                ax.scatter(*path[-1],color=color,marker='x' if r['outcome']['contact'] else 'o',s=35)
                target=pose[:3]+rotation_xyzw(pose[3:])@np.array([*GOAL_BODY_XY,0.])
                ax.scatter(*target[:2],marker='*',s=130,color='#009E73')
            ax.set(title=f'{g.replace("_"," ")} / appearance {seed}',xlim=(-.18,1.32),ylim=(-.9,.9),
                   xlabel='Native world x (m)',ylabel='Native world y (m)',aspect='equal')
            ax.grid(alpha=.2)
    handles=[Line2D([0],[0],color=colors[a],label=a.replace('_',' ')) for a in ACTIONS]
    fig.legend(handles=handles,loc='lower center',ncol=3)
    fig.suptitle('Native base-centre paths; x = contact stop, circle = other endpoint\n'
                 'Training-only independent episodes; centre lines do not represent swept robot volume',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.savefig(OUTPUT/'native_paths.png',dpi=170);fig.savefig(OUTPUT/'native_paths.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-sha256',required=True)
    args=parser.parse_args();validate_root(INPUT);validate_root(RAW_READOUT);validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists():raise ValueError('exclusive science readout; no rerun')
    readout_ids={'result.json':args.audit_sha256};verify_artifacts(RAW_READOUT,readout_ids)
    audit=read_json(RAW_READOUT,'result.json')
    if (audit['status']!='GEOMETRY_PROGRESS_AVAILABLE_EVIDENCE_READOUT_COMPLETE'
            or audit['audited_episodes']!=24 or audit['original_audit_remains_failed'] is not True
            or audit['prior_command_readout_remains_failed'] is not True):
        raise ValueError('complete available-evidence readout preserving both original failures required')
    readout_ids|=audit['output_sha256'];verify_artifacts(RAW_READOUT,readout_ids)
    ids=audit['collection_sha256']|{'geometry_progress_audit_failure.json':audit['original_failure_sha256']}
    verify_artifacts(INPUT,ids)
    collected=read_json(INPUT,'result.json');ids|=collected['artifact_sha256'];verify_artifacts(INPUT,ids)
    launch=read_json(INPUT,'launch.json');verify(launch)
    sources=discover_sources(('scripts/read_go2_geometry_progress_science_v1.py',
        'lewm/tests/test_geometry_progress_science_development.py',
        'lewm/tests/test_geometry_progress_learning_sample_development.py'),launch['source_sha256']|audit['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition)
    reports=[read_json(RAW_READOUT,c+'_geometry_progress_audit.json') for c in TRIALS]
    result=summary(reports)
    if result['panel']!=audit['panel']:raise ValueError('preregistered design gate mismatch')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(input_sha256=ids,source_sha256=sources,readout_sha256=readout_ids,
        scope='authenticated raw-audit aggregation and base-centre plot; no raw replay or fitting'))
    try:
        samples=[]
        for report in reports:
            if report['targets'] is None:
                samples.append(dict(trial=report['trial'],materialized=False,reason='MISSING_DEPARTURE'))
                continue
            sample=materialize(IntentReturnRGBDReplay(INPUT/report['trial']),report)
            t=sample['targets']
            samples.append(dict(trial=report['trial'],materialized=True,input_fields=list(sample['inputs']),
                history_shapes={k:list(v.shape) for k,v in sample['inputs']['observation_history'].items()},
                history_sha256={k:fingerprint(v.numpy()) for k,v in sample['inputs']['observation_history'].items()},
                action_shape=list(sample['inputs']['known_action_blocks'].shape),
                motion_valid=int(t['motion_valid'].sum()),future_valid=int(t['future_valid'].sum()),
                contact_valid=int(t['contact_valid'].sum()),contact_positive=int((t['contact']==1).sum())))
        result['learning_interface_materialization']=samples
        ready=[s for s in samples if s['materialized']]
        result['model_context_identity_counts']=dict(materialized_episodes=len(ready),
            distinct_rgb_histories=len({s['history_sha256']['rgb'] for s in ready}),
            distinct_body_histories=len({s['history_sha256']['body'] for s in ready}),
            distinct_control_histories=len({s['history_sha256']['control'] for s in ready}),
            distinct_non_rgb_history_pairs=len({(s['history_sha256']['body'],s['history_sha256']['control']) for s in ready}))
        plot_paths(reports);verify(definition);verify_artifacts(INPUT,ids);verify_artifacts(RAW_READOUT,readout_ids)
        result|=dict(collection_root=str(INPUT),audit_sha256=args.audit_sha256,
            raw_readout_root=str(RAW_READOUT),original_audit_remains_failed=True,
            prior_command_readout_remains_failed=True,
            prior_command_readout_failure_sha256=audit['prior_command_readout_failure_sha256'],
            original_failure_sha256=audit['original_failure_sha256'],
            collection_sha256=audit['collection_sha256'],source_sha256=sources,
            prefix_comparisons=audit['prefix_comparisons'],
            conditions=audit['conditions'],
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','native_paths.png','native_paths.svg')})
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:result[k] for k in ('status','episodes','full_horizons','successful_progress','contact_episodes',
            'panel','design_and_measurement_gate_pass','target_accounting','raw_replay')},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SCIENCE_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
