"""Read complete navigation and coverage outcomes of the in-place-view trial."""
from collections import Counter
import json
from types import SimpleNamespace

import numpy as np

from lewm.eligible_floor_registration_development import bind
from scripts.navigation_artifact_root_development import validate_root
from scripts.run_go2_current_position_coverage_view_development import BASE, ROOT, PLAN
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation_module


def main():
    root=BASE/ROOT; output=root/'current_position_coverage_view_readout_v1.json'
    if output.exists():raise ValueError('preserve completed readout')
    # The shared forecast evaluator also has an output-base validator. Bind
    # only that location; retain the same treatment and physical checks.
    selected=SimpleNamespace(BASE=BASE,ROOT=ROOT,PLAN=PLAN,ASSIGNMENTS=((1,'supervised_rollout'),))
    bind(evaluation_module.evaluate,study=selected,
        xy=bind(evaluation_module.xy,validate_root=bind(validate_root,BASE=BASE)))(1)
    read=lambda name:json.loads((root/name).read_text())
    evaluation=read('continuous_native_arrival_evaluation.json')
    plans=[r for r in read('planning.json') if 'selection' in r]
    requests=read('requests.json')
    events={}; rows=[]
    for plan in plans:
        selection=plan['selection']; request=selection.get('coverage_view_request',{})
        visit=request.get('visit') or {}; viewpoint=visit.get('camera_viewpoint',{})
        for event in request.get('recent_events',[]):
            events[(event['started_ns'],event['completed_ns'])]=event
        rows.append(dict(frame=plan['frame'],action=plan['action'],on_time=plan['on_time'],
            target_cell=request.get('target_cell'),view_status=request.get('status'),
            view_started_ns=visit.get('started_ns'),
            current_position_view=viewpoint.get('current_measured_position_view',False),
            viewpoint_map_xy_m=viewpoint.get('viewpoint_map_xy_m'),
            coverage_rejected=selection.get('translation_footprint_coverage',{}).get('rejected',False),
            action_before_coverage=selection.get('translation_footprint_coverage',{}).get('previous_action'),
            all_nominal_candidates_blocked=all(not c['nominal_predicted_path_clear']
                for c in selection['memory_forecast_candidates'])))
    # Physical geometry is used only after execution to diagnose the margin.
    launch=read('launch.json')
    walls=launch['fresh_layout_inventory']['layouts'][1]['geometry']['wall_boxes']
    assert all(w['yaw_rad']==0 for w in walls)
    centres=np.array([w['centre_xyz'][:2] for w in walls])
    half=np.array([w['size_xyz'][:2] for w in walls])*.5
    camera={r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:
        positions=data['base_pose_world'][[camera[r['frame']]['physical_sample_index'] for r in plans],:2]
    clearance=np.linalg.norm(np.maximum(np.abs(positions[:,None,:]-centres)-half,0),axis=2).min(1)
    for row,distance in zip(rows,clearance):row['physical_wall_clearance_m']=float(distance)
    events=list(events.values())
    result=dict(schema='current_position_coverage_view_readout.v1',evaluation=evaluation,
        pipeline_faults=read('pipeline_faults.json'),planning_records=len(plans),
        plans_on_time=sum(r['on_time'] for r in plans),
        actions=dict(Counter(r['action'] for r in plans)),
        coverage_rejections=sum(r['coverage_rejected'] for r in rows),
        view_statuses=dict(Counter(str(r['view_status']) for r in rows)),
        current_position_view_plans=sum(r['current_position_view'] for r in rows),
        current_position_view_actions=dict(Counter(r['action'] for r in rows if r['current_position_view'])),
        observed_patch_completions=sum(e['unknown_cell_observed'] for e in events),
        observed_patches_after_current_position_view=sum(e['unknown_cell_observed'] and
            e['camera_viewpoint'].get('current_measured_position_view',False) for e in events),
        coverage_event_reasons=dict(Counter(e['completion_reason'] for e in events)),
        unique_requested_cells=sorted({tuple(r['target_cell']) for r in rows if r['target_cell'] is not None}),
        physical_minimum_planning_wall_clearance_m=float(clearance.min()),
        physical_planning_samples_within_nominal_margin=int((clearance<=.45).sum()),
        dispatch_reasons=dict(Counter(r['reason'] for r in requests)),
        physical_geometry_used_only_for_posthoc_evaluation=True,
        circular_margin_not_articulated_contact_geometry=True,
        new_independent_layout=False,causal_improvement_proven=False,
        events=events,rows=rows)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','rows','dispatch_reasons','unique_requested_cells')},indent=2))


if __name__=='__main__':main()
