"""Complete cell denominators and exact causal-prefix sibling comparison."""
from collections import Counter
from lewm.moving_action_switch_family_development import TRIALS,assignments,CANONICAL,ACTIONS


def summarize(reports):
    if [r['trial'] for r in reports]!=list(TRIALS):raise ValueError('complete ordered 144-cell population required')
    expected=assignments()
    for r in reports:
        if any(r[k]!=v for k,v in expected[r['trial']].items()):raise ValueError('exact prospective assignment required')
        if not r['raw_sensor_reconstruction_pass'] or not r['command_stop_replay_pass']:raise ValueError('complete raw replay required')
    comparisons=[]
    for cluster in CANONICAL:
        for prefix in ACTIONS:
            rows=[r for r in reports if r['cluster']==cluster and r['prefix_action']==prefix]
            assert len(rows)==6 and {r['suffix_action'] for r in rows}==set(ACTIONS)
            available=[r for r in rows if r['prefix']['complete']]
            exact=bool(len(available)==6 and all(r['prefix']==rows[0]['prefix'] for r in rows))
            unavailable=not available and all(not r['outcome']['branch_available'] for r in rows)
            comparisons.append(dict(cluster=cluster,prefix_action=prefix,trials=[r['trial'] for r in rows],
                available_siblings=len(available),all_six_exact=exact,all_six_unavailable=unavailable,
                passes=exact or unavailable))
    roles={}
    for role in ('train','geometry_transfer'):
        rows=[r for r in reports if r['data_role']==role]
        targets=[t for r in rows if r['targets'] is not None for t in r['targets']['targets']]
        roles[role]=dict(cells=len(rows),clusters=sorted({r['cluster'] for r in rows}),
            prefix_counts=dict(Counter(r['prefix_action'] for r in rows)),suffix_counts=dict(Counter(r['suffix_action'] for r in rows)),
            repeat_cells=sum(r['prefix_action']==r['suffix_action'] for r in rows),
            available_branches=sum(r['outcome']['branch_available'] for r in rows),
            complete_schedules=sum(r['outcome']['complete_schedule'] for r in rows),
            physical_stops=[dict(trial=r['trial'],stop=r['physical_stop']) for r in rows if r['physical_stop'] is not None],
            hard_measurement_failed_cases=[r['trial'] for r in rows if r['hard_measurement_failed_frames']],
            strict_visibility_failed_cases=[r['trial'] for r in rows if not r['strict_physical_visibility_pass']],
            target_accounting=dict(expected_slots=len(rows)*8,recorded_slots=len(targets),
                motion_valid=sum(t['motion_valid'] for t in targets),future_image_valid=sum(t['future_image_valid'] for t in targets),
                contact_valid=sum(t['contact_valid'] for t in targets),contact_positive=sum(t['contact']==1. for t in targets)))
    gate=all(c['passes'] for c in comparisons) and all(not r['hard_measurement_failed_frames'] and r['strict_physical_visibility_pass'] for r in reports)
    return dict(cells=len(reports),roles=roles,prefix_comparisons=comparisons,
        all_measurement_and_prefix_gates_pass=bool(gate),episode_exclusions=[],model_trained=False,
        independent_maze_evaluation_layouts=0,navigation_qualified=False,goal_achieved=False)
