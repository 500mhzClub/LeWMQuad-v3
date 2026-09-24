"""Complete prospective role/cluster denominators, never outcome-based filtering."""
from collections import Counter
from lewm.geometry_progress_layout_family_development import TRIALS,assignments,ACTIONS,APPEARANCES,CLUSTERS


def acquisition_complete(r):
    return bool(r['setup_admitted'] and r['targets'] is not None and r['frames']>0
        and (r['outcome']['complete_horizon'] or r['outcome']['physical_stop']=='DISALLOWED_CONTACT')
        and r['outcome']['acquisition_stop'] is None)


def summarize(reports):
    if [r['trial'] for r in reports]!=list(TRIALS):raise ValueError('complete ordered 96-episode cohort required')
    expected=assignments()
    for r in reports:
        if any(r[k]!=v for k,v in expected[r['trial']].items()):raise ValueError('exact prospective role/cluster/action assignment required')
        if not r['raw_sensor_reconstruction_pass'] or not r['command_stop_replay_pass']:raise ValueError('complete raw replay required')
    roles={}
    for role in ('train','geometry_transfer'):
        rows=[r for r in reports if r['data_role']==role]
        by={(r['geometry'],r['appearance_seed'],r['action']):r['outcome']['successful_progress'] for r in rows}
        clusters=[c for c,v in CLUSTERS.items() if v[0]==role]
        reversal=[dict(cluster=c,appearance_seed=s,passes=bool(
            by[c+'_left_open',s,'left_arc'] and not by[c+'_left_open',s,'right_arc']
            and by[c+'_right_open',s,'right_arc'] and not by[c+'_right_open',s,'left_arc']))
            for c in clusters for s in APPEARANCES]
        constants=[a for a in ACTIONS if all(r['outcome']['successful_progress'] for r in rows if r['action']==a)]
        hard={r['trial']:r['hard_measurement_failed_frames'] for r in rows if r['hard_measurement_failed_frames']}
        complete=all(acquisition_complete(r) for r in rows)
        controls_fail=not any(r['outcome']['successful_progress'] for r in rows if r['action'] in ('hold','left_turn','right_turn'))
        targets=[t for r in rows if r['targets'] is not None for t in r['targets']['targets']]
        roles[role]=dict(episodes=len(rows),parameter_clusters=len(clusters),layouts=len({r['geometry'] for r in rows}),
            action_counts=dict(Counter(r['action'] for r in rows)),successful_progress=sum(r['outcome']['successful_progress'] for r in rows),
            contact_episodes=sum(r['outcome']['contact'] for r in rows),
            all_candidate_acquisitions_complete=complete,hard_measurement_failed_cases=hard,
            strict_depth_failed_cases=[r['trial'] for r in rows if not r['strict_physical_visibility_pass']],
            prediction_measurement_gate_pass=bool(complete and not hard),mirrored_reversals=reversal,
            constant_success_actions=constants,nonprogress_controls_fail=controls_fail,
            native_action_design_informative=bool(all(v['passes'] for v in reversal) and controls_fail and not constants),
            target_accounting=dict(expected_slots=len(rows)*8,recorded_slots=len(targets),
                motion_valid=sum(t['motion_valid'] for t in targets),future_image_valid=sum(t['future_image_valid'] for t in targets),
                contact_valid=sum(t['contact_valid'] for t in targets),contact_positive=sum(t['contact']==1. for t in targets)))
    return dict(episodes=len(reports),roles=roles,all_measurement_gates_pass=all(r['prediction_measurement_gate_pass'] for r in roles.values()),
        training_design_informative=roles['train']['native_action_design_informative'],
        episode_exclusions=[],mirrored_siblings_are_independent=False,model_trained=False,
        independent_maze_evaluation_layouts=0,navigation_qualified=False,goal_achieved=False)
