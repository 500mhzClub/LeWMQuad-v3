"""Read actual missing-pose, command and prediction-history evidence after a run."""
import argparse
import json
from scripts.compare_continuous_navigation_arms_development import path,read


def evaluate(root):
    launch=read(root,'launch.json');mission=read(root,'mission.json')
    poses=read(root,'poses.json');plans=read(root,'planning.json')
    requests=read(root,'requests.json');stages=read(root,'stage_events.json')
    declared=set(launch['declared_floor_pose_gap_frames'])
    rejected={r['frame'] for r in mission if 'floor_rejection' in r}
    injected={r['frame'] for r in mission if r.get('floor_rejection',{}).get('fault_injection')}
    published={r['frame'] for r in poses}
    mapped={r['frame'] for r in stages if r['stage']=='mapping'}
    tracked={r['frame'] for r in stages if r['stage']=='tracking'}
    violations=[]
    if published&rejected:violations.append('rejected pose published')
    if mapped&rejected:violations.append('rejected pose sent to mapping')
    for r in mission:
        if r['frame'] not in rejected:continue
        if (r['consumed_pose_frame'] is not None or r['quiet_intervals']!=0
                or r['observed_goal_distance_m'] is not None or r.get('arrival_confirmed_this_frame')):
            violations.append(f'rejected frame {r["frame"]} used by mission')
    for prior,current in zip(mission,mission[1:]):
        if prior['frame'] in rejected and current['frame'] not in rejected:
            settling=current.get('observed_settling')
            if settling is None or settling['previous_position_initial_body_m'] is not None:
                violations.append('arrival motion history bridges a missing pose')
    episodes=[];active=None
    for r in mission:
        waiting=r.get('floor_reacquisition_hold',False)
        if waiting and active is None:
            active=dict(first_missing_frame=r['frame'],hold_published_ns=r['published_ns'])
        if not waiting and active is not None:
            active.update(reacquired_frame=r['frame'],reacquired_published_ns=r['published_ns'],
                accepted_pose_streak=r['consecutive_accepted_floor_poses'])
            episodes.append(active);active=None
    if active is not None:episodes.append(active)
    for e in episodes:
        start=e['hold_published_ns'];end=e.get('reacquired_published_ns')
        window=[r for r in requests if r['simulator_ns']>start
            and (end is None or r['simulator_ns']<end)]
        nonzero=[r['simulator_ns'] for r in window if any(r['requested_command'])]
        e.update(requests_checked=len(window),nonzero_requests_while_waiting=nonzero,
            exact_publication_timestamp_ties_excluded=True)
        if nonzero:violations.append('nonzero request while floor reacquisition was pending')
        if end is not None:
            if e['accepted_pose_streak']<4:violations.append('reacquisition before four accepted poses')
            first=next((r for r in requests if r['simulator_ns']>=end and any(r['requested_command'])),None)
            e['first_subsequent_nonzero_request_ns']=None if first is None else first['simulator_ns']
            stale=[p['frame'] for p in plans if p.get('committed') is True
                and start<=p['completed_ns']<end]
            if stale:violations.append('plan committed during reacquisition hold')
            e['committed_plans_during_hold']=stale
            old=[p['frame'] for p in plans if p.get('committed') is True
                and p['completed_ns']>start and p['frame']<e['reacquired_frame']]
            e['old_observation_plans_committed_after_rejection']=old
            if old:violations.append('old observation plan committed after floor rejection')
    corrections=[p for p in plans if 'motion_correction' in p]
    for p in corrections:
        history=p['motion_correction']['pose_history_frames']
        if history!=list(range(p['frame']-3,p['frame']+1)) or set(history)&rejected:
            violations.append(f'incomplete or rejected prediction pose history at {p["frame"]}')
    after_gap=[p for p in corrections if p['frame']>max(declared)]
    declared_episodes=[e for e in episodes if e['first_missing_frame']<=max(declared)
        and e.get('reacquired_frame',float('inf'))>min(declared)]
    arrivals=read(root,'continuous_native_arrival_evaluation.json')
    return dict(root_name=root.name,declared_frames=sorted(declared),injected_frames=sorted(injected),
        natural_rejection_frames=sorted(rejected-injected),
        full_declared_gap_exercised=injected==declared,
        raw_tracking_continued_on_declared_frames=declared<=tracked,
        rejected_frames_absent_from_published_poses=not bool(published&rejected),
        rejected_frames_absent_from_mapping=not bool(mapped&rejected),
        reacquisition_episodes=episodes,prediction_corrections_checked=len(corrections),
        prediction_corrections_after_gap=len(after_gap),violations=violations,
        declared_gap_recovery_passed=injected==declared and not violations
            and bool(declared_episodes) and all(e.get('accepted_pose_streak',0)>=4
                and e['requests_checked']>0 and e.get('first_subsequent_nonzero_request_ns') is not None
                for e in declared_episodes),
        execution_checks_passed=injected==declared and declared<=tracked and not violations
            and bool(episodes) and all(e['requests_checked']>0
                and e.get('first_subsequent_nonzero_request_ns') is not None for e in episodes),
        verified_round_trip=arrivals['round_trip_arrival_checks_passed'],
        disallowed_contact_samples=arrivals['disallowed_contact_samples'],
        pipeline_faults=read(root,'pipeline_faults.json'),
        model_assignment=launch['model_assignment'],fault_injection=True,
        calibrated_hardware_sensor_outage=False,controller_comparison=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    root=path(parser.parse_args().root_name);result=evaluate(root)
    with (root/'floor_gap_execution_diagnostic_v1.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
