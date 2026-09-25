"""Episode outcomes using fixed physical arrival and native articulated readers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_v4_development import ArticulatedSteps
from lewm.physical_execution_development import rotation_xyzw
from lewm.navigation_capability_hold_taxonomy_development import classify, override_label
from scripts.evaluate_continuous_native_arrivals_development import evaluate as original_arrivals
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def retained_physical_arrivals(root,frames,trace,mission,requests):
    """Existing physical criteria from retained timestamps when RGB was omitted.

    This fallback restores no image or sensor evidence and cannot qualify video.
    """
    lookup={r['frame']:r for r in frames}
    physics=trace['base_pose_world'];origin=physics[frames[0]['physical_sample_index']]
    R=rotation_xyzw(origin[3:]);commands={r['simulator_ns']:r['requested_command'] for r in requests}
    rows=[]
    for arrival in mission[-1]['arrivals']:
        end=arrival['frame'];start=end-10
        a,b=lookup[start],lookup[end]
        assert b['measured_ns']-a['measured_ns']==1_000_000_000
        positions=(physics[a['physical_sample_index']:b['physical_sample_index']+1,:3]-origin[:3])@R
        distances=np.linalg.norm(positions[:,:2]-np.asarray(arrival['target_initial_body_xy_m']),axis=1)
        boundaries=(physics[[lookup[f]['physical_sample_index'] for f in range(start,end+1)],:3]-origin[:3])@R
        speeds=np.linalg.norm(np.diff(boundaries,axis=0),axis=1)/.1
        zero=all(t in commands and np.array_equal(commands[t],[0.,0.,0.]) for t in range(a['measured_ns'],b['measured_ns'],20_000_000))
        rows.append(dict(phase=arrival['phase'],frame=end,physical_radius_m=.04,dwell_seconds=1.,
            native_maximum_distance_m=float(distances.max()),native_maximum_100ms_speed_m_s=float(speeds.max()),
            all_requested_intervals_zero=zero,arrival_checks_passed=bool(np.all(distances<=.04) and np.all(speeds<=.05) and zero)))
    return dict(arrivals=rows,disallowed_contact_samples=int(np.count_nonzero(trace['physics_contact'])),
        mission_terminal=mission[-1]['terminal'],source_rgb_retained=False,video_qualified=False,
        frame_physics_indices='Exact timestamp join of retained acquisitions and native physics; no image reconstruction',
        criteria_source_sha256=sha('scripts/evaluate_continuous_native_arrivals_development.py'))


def report(root):
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root'])
    if not root.resolve().is_relative_to((base/'runs').resolve()):raise ValueError('new capability run root required')
    output.install(base);budget=Budget(base,protocol)
    result=json.loads((root/'result.json').read_text())
    if result['frames']==0:
        record=dict(status='STARTUP_FAILURE',science_episode=False,result=result)
        save(root/'episode_evaluation.json',record)
        return record
    spec=json.loads((root/'specification.json').read_text());episode=json.loads((root/'episode.json').read_text())
    planning=json.loads((root/'planning.json').read_text());mission=json.loads((root/'mission.json').read_text())
    requests=json.loads((root/'requests.json').read_text())
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as archive:
        trace={key:archive[key].copy() for key in ('base_pose_world','joint_position','physics_contact','timestamp_s')}
    metadata=root/'native/in_memory_camera_observations.json'
    if metadata.exists():
        frames=json.loads(metadata.read_text())['frames']
        native=original_arrivals(root)
    else:
        stamps=np.rint(trace['timestamp_s']*1e9).astype(np.int64)
        frames=[]
        for row in json.loads((root/'acquisitions.json').read_text()):
            index=int(np.searchsorted(stamps,row['measured_ns']))
            assert stamps[index]==row['measured_ns']
            frames.append(row|dict(physical_sample_index=index))
        native=retained_physical_arrivals(root,frames,trace,mission,requests)
    frame_lookup={r['frame']:r for r in frames}
    first=frames[0]['physical_sample_index']
    # Keep the existing dwell/speed/zero-command rules, and also require arrival
    # at the generated fixed beacon/home rather than an accidentally shifted
    # initial-frame target after settling.
    arrivals=[]
    for row in native['arrivals']:
        target=episode['beacon_xy_world'] if row['phase']=='OUTBOUND' else episode['home_se2_world'][:2]
        a,b=[frame_lookup[f]['physical_sample_index'] for f in (row['frame']-10,row['frame'])]
        error=np.linalg.norm(trace['base_pose_world'][a:b+1,:2]-np.asarray(target),axis=1)
        arrivals.append(row|dict(generated_target_maximum_distance_m=float(error.max()),
            generated_target_distance_passed=bool(np.all(error<=.04)),
            passed=row['arrival_checks_passed'] and bool(np.all(error<=.04))))
    passed={r['phase']:r for r in arrivals if r['passed']}
    selected=[r for r in planning if 'selection' in r]
    phases={r['frame']:r['phase'] for r in mission}
    holds=[classify(r)|dict(phase=phases.get(r['frame'],'unknown')) for r in selected if r.get('action',r['selection']['action'])=='hold']
    stall={}
    for phase in sorted(set(phases.values())):
        n=sum(phases.get(r['frame'])==phase for r in selected);h=sum(r['phase']==phase for r in holds)
        stall[phase]=dict(selected_plans=n,holds=h,rate=h/n if n else None)
    reader=ArticulatedSteps(spec['geometry']['wall_boxes'])
    minima=[];uppers=[];robust=[];native_started=time.monotonic()
    # Overlapping endpoints preserve every native step and every interval.
    for start in range(first,len(trace['timestamp_s'])-1,400):
        budget.check()
        end=min(start+401,len(trace['timestamp_s']))
        chunk={k:v[start:end] for k,v in trace.items()}
        evaluated=reader.evaluate(chunk)
        low=np.asarray(evaluated['per_step_primitive_separation_lower_m']).min(axis=1)
        high=np.asarray(evaluated['per_step_primitive_separation_upper_m']).min(axis=1)
        if minima:low,high=low[1:],high[1:]
        minima.extend(low);uppers.extend(high)
        robust.extend(np.asarray(evaluated['per_interval_primitive_robust_lower_m']).min(axis=1))
    low,high=np.asarray(minima),np.asarray(uppers)
    assert len(low)==len(trace['timestamp_s'])-first
    assert np.all(np.diff(np.rint(trace['timestamp_s'][first:]*1e9).astype(np.int64))==2_000_000)
    contacts=trace['physics_contact'][first:].astype(bool)
    safety={}
    for name,threshold in (('hard',.005),('operating',.02)):
        violation=(high<threshold)|contacts
        unresolved=(low<threshold)&~violation
        safety[name]=dict(threshold_m=threshold,confirmed_violation_samples=int(violation.sum()),
            unresolved_sampled_samples=int(unresolved.sum()),all_native_samples_qualified=not unresolved.any(),
            minimum_separation_lower_m=float(low.min()),minimum_separation_upper_m=float(high.min()),
            interval_robustness_failure_count=int(np.count_nonzero(np.asarray(robust)<threshold)),
            native_discrete_ground_truth=True,interval_check_secondary=True)
    with (root/'native_clearance_summary_arrays.npz').open('xb') as stream:
        np.savez_compressed(stream,timestamp_s=trace['timestamp_s'][first:],separation_lower_m=low,
            separation_upper_m=high,interval_robust_lower_m=np.asarray(robust))
    def leg(start,end,success,shortest):
        if end < start:raise ValueError('Non-monotone logged leg boundaries')
        length=float(np.linalg.norm(np.diff(trace['base_pose_world'][start:end+1,:2],axis=0),axis=1).sum())
        return dict(success=success,actual_path_m=length,shortest_path_m=shortest,
            spl=float(shortest/max(length,shortest)) if success else 0.,
            elapsed_s=float(trace['timestamp_s'][end]-trace['timestamp_s'][start]))
    # A physically rejected arrival still marks the recorded phase transition.
    # Keep phase segmentation separate from each leg's success criterion.
    logged_arrivals={r['phase']:r for r in arrivals}
    outbound_end=frame_lookup[logged_arrivals['OUTBOUND']['frame']]['physical_sample_index'] if 'OUTBOUND' in logged_arrivals else len(trace['timestamp_s'])-1
    outbound=leg(first,outbound_end,'OUTBOUND' in passed,episode['shortest_outbound_m'])
    return_end=frame_lookup[logged_arrivals['RETURN']['frame']]['physical_sample_index'] if 'RETURN' in logged_arrivals else len(trace['timestamp_s'])-1
    home=leg(outbound_end,return_end,'RETURN' in passed,episode['shortest_return_m'])
    timing=json.loads((root/'stage_timings.json').read_text())
    latencies=[r['wall_ns']/1e9 for r in timing if r.get('stage')=='planning']
    taxonomy=Counter()
    failed_arrivals=sum(not r['passed'] for r in arrivals)
    if failed_arrivals:taxonomy['observed arrival rejected by physical verification']=failed_arrivals
    if native['disallowed_contact_samples']:taxonomy['contact']+=1
    if safety['hard']['confirmed_violation_samples']:taxonomy['hard-clearance violation']+=1
    if (root/'failure.json').exists():
        error=json.loads((root/'failure.json').read_text())['reason']
        taxonomy['pose loss' if any(s in error.lower() for s in ('pose','tracking','registration')) else 'technical failure']+=1
    for row in holds:
        if row['category']=='explicit_override':taxonomy[override_label(row)]+=1
        elif row['category']=='no_eligible_movement':
            for flag,label in (('observation_action_space_exclusions','observation/view restriction'),
                               ('motion_clearance_exclusions','memory clearance'),
                               ('stopping_projection_exclusions','stopping projection')):
                if row[flag]:taxonomy[label]+=1
        elif row['category']=='movement_lost_recorded_score_or_tie':taxonomy['movement outscored']+=1
        else:taxonomy['insufficient retained evidence']+=1
    if result['policy_steps']>=24000 and 'RETURN' not in passed:
        taxonomy['budget exhaustion despite progress' if outbound['actual_path_m']>.02 else 'budget exhaustion without movement']+=1
    record=dict(schema='navigation_capability_episode_evaluation.v1',controller=json.loads((root/'config.json').read_text())['controller'],
        episode_id=episode['episode_id'],role=episode['role'],beacon_success='OUTBOUND' in passed,
        home_success='RETURN' in passed,round_trip_success=all(k in passed for k in ('OUTBOUND','RETURN')),
        source_error=result['error'],arrivals=arrivals,original_arrival_reader=native,outbound=outbound,return_leg=home,
        disallowed_contact_samples=native['disallowed_contact_samples'],safety=safety,stall_by_phase=stall,
        hold_categories=dict(Counter(r['category'] for r in holds)),failure_and_stall_taxonomy=dict(taxonomy),
        failure_taxonomy_counts_overlap=True,hold_details=holds,
        decision_latency_s=dict(median=float(np.median(latencies)) if latencies else None,p95=float(np.quantile(latencies,.95)) if latencies else None),
        wall_s=result['wall_s'],wall_seconds_per_simulated_second=result['wall_s']/result['simulated_s'],
        articulated_reader_wall_s=time.monotonic()-native_started,training_render_provenance='unverified',
        label='Development pilot; not validation capability evidence',
        input_sha256={name:sha(root/name) for name in ('config.json','result.json','planning.json','mission.json','requests.json','native/physics_trace.npz')})
    save(root/'episode_evaluation.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result=report(args.root)
    print(json.dumps({k:v for k,v in result.items() if k not in ('hold_details','original_arrival_reader','input_sha256')}))
