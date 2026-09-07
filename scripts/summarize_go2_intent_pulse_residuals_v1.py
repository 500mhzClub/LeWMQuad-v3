"""Recorded variable-settling response diagnostic, not new model fitting.

The nominal table uses brake-20 endpoints; controller response records may
wait longer. Keep that timing distinction and incomplete pulses explicit.
Repeated closed-loop actions are not independent scene samples or a causal
per-action friction comparison. No native pose is read by this diagnostic.
"""
import json
import numpy as np
from scripts.run_go2_intent_room_return_v1 import OUTPUT,TRIALS
from scripts.fixed_nominal_pulse_table_development import load_fixed_table
from scripts.run_go2_successive_choice_maze_development_v1 import digest
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify

IDENTITIES={'launch.json':'7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91',
    'result.json':'27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3',
    'raw_return_audit.json':'a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d'}


def summary(events):
    observed=np.asarray([r['observed'] for r in events]);predicted=np.asarray([r['predicted'] for r in events])
    residual=observed-predicted
    return dict(count=len(events),mean_observed_xy_yaw=observed.mean(0).tolist(),
        mean_residual_xy_yaw=residual.mean(0).tolist(),
        planar_rmse_m=float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1)))),
        yaw_rmse_rad=float(np.sqrt(np.mean(residual[:,2]**2))),
        brake_ticks=sorted(set(r['brake_ticks'] for r in events)),
        non_brake20_count=sum(r['brake_ticks']!=20 for r in events))


def check():
    verify_artifacts(OUTPUT,IDENTITIES);launch=read_json(OUTPUT,'launch.json');verify(launch)
    collection=read_json(OUTPUT,'result.json')
    if collection['absent_expected_artifacts']:raise ValueError('complete recorded collection required')
    external=IDENTITIES|collection['artifact_sha256'];verify_artifacts(OUTPUT,external)
    sources=discover_sources(('scripts/summarize_go2_intent_pulse_residuals_v1.py',),launch['source_sha256'])
    verify_bindings(sources|launch['input_sha256']);table=load_fixed_table();reports={}
    for c in TRIALS:
        rows=read_json(OUTPUT/c,'servo_decisions.json');started={};completed={}
        for row in rows:
            local=row['decision']['execution']['local_decision']
            if local is None:continue
            d=local['diagnostic'];anchor=local['goal']['anchor_ns']
            if 'new_pulse' in d:
                pulse=d['new_pulse'];key=(anchor,pulse['pulse_index'])
                if key in started:raise ValueError('duplicate pulse start')
                effect=table.effects[pulse['action_index']]
                if effect.ticks!=pulse['ticks']:raise ValueError('pulse duration disagreement')
                started[key]=dict(tick=row['tick'],action_index=pulse['action_index'],pulse_ticks=pulse['ticks'])
            if 'completed_action_response' in d:
                response=d['completed_action_response'];key=(anchor,response['pulse_index'])
                if key not in started or key in completed:raise ValueError('unpaired or duplicate pulse completion')
                index=response['action_index']
                if index!=started[key]['action_index']:raise ValueError('pulse action disagreement')
                predicted=np.asarray(table.effects[index].delta_xy_yaw)
                observed=np.r_[response['displacement_start_body_m'][:2],response['yaw_change_rad']]
                np.testing.assert_array_equal(predicted,response['predicted_delta_xy_yaw'])
                np.testing.assert_allclose(observed-predicted,response['observed_minus_predicted'],rtol=0,atol=1e-12)
                completed[key]=started[key]|dict(finished_tick=row['tick'],brake_ticks=response['brake_ticks'],
                    observed=observed.tolist(),predicted=predicted.tolist())
        events=list(completed.values());cells={}
        for index,effect in enumerate(table.effects):
            members=[r for r in events if r['action_index']==index]
            cells[str(index)]=dict(command=list(effect.command),pulse_ticks=effect.ticks,
                summary=summary(members) if members else None)
        reports[c]=dict(started_pulses=len(started),completed_response_records=len(events),
            incomplete=[dict(anchor_ns=k[0],pulse_index=k[1],**v) for k,v in started.items() if k not in completed],
            aggregate=summary(events) if events else None,action_duration_cells=cells,
            events=events,independent_layouts=1,model_updated=False)
    verify_bindings(sources|launch['input_sha256']);verify_artifacts(OUTPUT,external)
    return dict(status='RECORDED_VARIABLE_SETTLING_PULSE_RESIDUAL_DIAGNOSTIC',conditions=reports,
        source_sha256={p:h for p,h in sources.items() if p not in launch['source_sha256']},
        collection_launch_sha256=digest(OUTPUT/'launch.json'),collection_result_sha256=digest(OUTPUT/'result.json'),
        collection_audit_sha256=digest(OUTPUT/'raw_return_audit.json'),native_pose_read=False,
        endpoint_timing_matched=False,model_fitting=False,causal_friction_effect_established=False,goal_achieved=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
