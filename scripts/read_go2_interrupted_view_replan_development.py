"""Evaluate navigation and subsequent view choices after visual interruptions."""
import json
from collections import Counter
import numpy as np

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_interrupted_view_replan_development as run
from scripts import read_go2_current_position_coverage_view_development as views


def physical_return_edges(root, *, outcome=None, layout=None):
    read=lambda name:json.loads((root/name).read_text())
    if outcome is None:outcome=read('continuous_native_arrival_evaluation.json')
    arrival=next((r for r in outcome['arrivals'] if r['phase']=='OUTBOUND'
        and r['arrival_checks_passed']),None)
    if arrival is None:return dict(status='NO_VERIFIED_OUTBOUND_ARRIVAL',physical_backtracking_observed=False)
    if layout is None:
        launch=read('launch.json');index=launch['layout_index']
        layout=launch['fresh_layout_inventory']['layouts'][index]['evaluation_layout']
    pitch=layout['pitch_m'];cells={tuple(c) for c in layout['cells']}
    edges={tuple(sorted((tuple(a),tuple(b)))) for a,b in layout['edges']}
    frames=sorted(read('native/in_memory_camera_observations.json')['frames'],key=lambda r:r['frame'])
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:
        positions=data['base_pose_world'][[r['physical_sample_index'] for r in frames],:2]
    nearest=np.rint(positions/pitch).astype(int)
    transitions=[];outside=[];previous=None
    for record,cell in zip(frames,nearest):
        cell=tuple(map(int,cell))
        if cell not in cells:outside.append(record['frame']);previous=None;continue
        if previous is not None and cell!=previous:
            transitions.append(dict(frame=record['frame'],source=previous,target=cell,
                graph_edge=tuple(sorted((previous,cell))) in edges,
                leg='outbound' if record['frame']<=arrival['frame'] else 'return'))
        previous=cell
    outbound={(tuple(r['source']),tuple(r['target'])) for r in transitions if r['leg']=='outbound' and r['graph_edge']}
    returning={(tuple(r['source']),tuple(r['target'])) for r in transitions if r['leg']=='return' and r['graph_edge']}
    reverse={(b,a) for a,b in outbound}&returning
    result=dict(schema='physical_return_corridor_readout.v1',
        round_trip_verified=outcome['round_trip_arrival_checks_passed'],verified_outbound_frame=arrival['frame'],
        camera_rate_physics_samples=len(frames),cell_assignment='nearest world-grid cell centre',
        pitch_m=pitch,outside_known_grid_frames=outside,
        invalid_graph_transitions=sum(not r['graph_edge'] for r in transitions),
        outbound_unique_directed_edges=len(outbound),return_unique_directed_edges=len(returning),
        return_edges_reversing_observed_outbound_edges=len(reverse),reverse_edges=sorted(reverse),
        transitions=transitions,physical_backtracking_observed=bool(reverse) and not outside and all(r['graph_edge'] for r in transitions),
        native_state_and_maze_graph_evaluator_only=True,boundary_jitter_can_repeat_transitions=True,
        unique_edge_counts_used=True,memory_causal_advantage_established=False)
    with (root/'physical_return_corridor_readout_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return {k:v for k,v in result.items() if k not in ('transitions','reverse_edges','outside_known_grid_frames')}


def main():
    root=run.BASE/run.ROOT;output=root/'interrupted_view_replan_readout_v1.json'
    if output.exists():raise ValueError('preserve completed readout')
    bind(views.main,BASE=run.BASE,ROOT=run.ROOT,PLAN=run.PLAN)()
    read=lambda name:json.loads((root/name).read_text())
    plans=[p for p in read('planning.json') if 'selection' in p]
    frontier=read('frontier_visits.json');coverage=read('current_position_coverage_view_readout_v1.json')
    records=[]
    for name,events,pending in (('frontier',frontier['events'],frontier['pending']),
            ('coverage',coverage['events'],None)):
        visits=events+([] if pending is None else [pending])
        for event in events:
            if event['completion_reason']!='WEAK_VISUAL_SUPPORT_INTERRUPTED_VIEW':continue
            later=[v for v in visits if v['started_ns']>event['completed_ns']
                and v['unknown_neighbour']==event['unknown_neighbour']]
            later.sort(key=lambda v:v['started_ns'])
            next_view=later[0] if later else None
            records.append(dict(kind=name,unknown_cell=event['unknown_neighbour'],
                started_ns=event['started_ns'],interrupted_ns=event['completed_ns'],
                original_viewpoint_map_xy_m=event['camera_viewpoint']['viewpoint_map_xy_m'],
                next_same_patch_viewpoint=None if next_view is None else next_view['camera_viewpoint']['viewpoint_map_xy_m'],
                next_same_patch_view_started_ns=None if next_view is None else next_view['started_ns'],
                subsequently_observed_in_recorded_visit=any(v.get('unknown_cell_observed',False) for v in later)))
    profiles=read('live_planning_profile.json')
    result=dict(schema='interrupted_view_replan_readout.v1',
        navigation=read('short_pulse_navigation_evaluation_v1.json'),
        physical_backtracking=physical_return_edges(root),
        interruption_count=len(records),interrupted_visits=records,
        frontier_event_reasons=dict(Counter(e['completion_reason'] for e in frontier['events'])),
        coverage_event_reasons=coverage['coverage_event_reasons'],
        current_position_patch_observations=coverage['observed_patches_after_current_position_view'],
        route_wall_ms=dict(zip(('median','p95','maximum'),np.percentile(
            [r['components']['route']['wall_ns']/1e6 for r in profiles if 'route' in r['components']],
            [50,95,100]).tolist())),
        dispatch_reasons=dict(Counter(r['reason'] for r in read('requests.json'))),
        reference_navigation=json.loads((run.REFERENCE/'short_pulse_navigation_evaluation_v1.json').read_text()),
        asynchronous_trajectories_not_matched=True,causal_navigation_improvement_proven=False,
        independent_layout_replication=False,hardware_validated=False)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('interrupted_visits','dispatch_reasons')},indent=2))


if __name__=='__main__':main()
