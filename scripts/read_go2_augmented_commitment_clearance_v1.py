"""Executed forecast errors and previously observed clearance witnesses only."""
import math
import time
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.physical_execution_development import rotation_xyzw
from lewm.observed_geometry_refinement_development import nominal_connector
from scripts.run_go2_augmented_family_switch_goal_probe_v1 import OUTPUT as INPUT,PRIOR,PRIOR_SHA,CASES,TRIAL
from scripts.read_go2_augmented_family_switch_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_augmented_commitment_clearance_v1_attempt_001'
PROTOCOL='docs/go2_augmented_commitment_clearance_v1_2026-09-08.md'
INPUT_SHA='f2e0a45a77f27b7a23ca3f621d40de68ad9fcb10c40515abcefb11e7c6e85dab'
READOUT_SHA='8d3fd2d78d016fe250a295b17d451c5392be62720999fa13292036c7e4ea49e2'


def path_witnesses(points,cells):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('nonempty finite executed XY samples required')
    checks=[nominal_connector(a,a,cells) for a in p]
    failed=[i for i,c in enumerate(checks) if not c['nominal_disk_connector_clear']]
    distances=[c['minimum_observed_cell_distance_m'] for c in checks if c['minimum_observed_cell_distance_m'] is not None]
    return dict(samples=len(p),sampled_minimum_distance_to_witness_m=min(distances,default=None),
        first_sample_inside_nominal_radius=failed[0] if failed else None,
        sampled_radius_conflicts=len(failed),all_occupied_squares_checked=False,
        continuous_articulated_motion_certified=False)


def evaluate(root,case):
    rows=read_json(root/case,'context_decisions.json');tape=read_json(root/case,'command_tape.json')
    cameras=read_json(root/case,'camera_audit.json');audit=read_json(root,case+'_audit.json')
    assert audit['raw_model_command_replay_pass'] and audit['raw_sensor_reconstruction_pass']
    assert audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames']
    collection=read_json(root/case,'result.json')
    assert collection['physical_stop'] is None and collection['acquisition_stop'] is None
    with np.load(root/case/'physics_trace.npz',allow_pickle=False) as z:poses=z['base_pose_world']
    initial=poses[749];R0=rotation_xyzw(initial[3:]);reports=[];excluded=[]
    for row in rows:
        s=row['decision']['new_selection'];tick=row['tick']
        if s is None:continue
        if s['action'] is None:
            excluded.append(dict(tick=tick,reason='no selected executable action'));continue
        interval=tape[tick:tick+5];expected=candidate_commands(s['action'])[:5]
        complete=len(interval)==5 and tick+5<len(rows) and all(t['completed'] and
            t['requested_command']==list(c) for t,c in zip(interval,expected,strict=True))
        if not complete:
            excluded.append(dict(tick=tick,reason='interrupted selected commitment'));continue
        start,end=(cameras[i]['physical_sample_index'] for i in (tick,tick+5))
        assert end-start==250 and end<len(poses)
        B=np.asarray(row['decision']['memory_receipt']['map_from_initial'])
        observed=row['decision']['evidence']['current_pose']
        p=B@np.asarray(observed['position_initial_body_m'])
        R=B@np.asarray(observed['rotation_initial_body_from_current_body'])
        prediction=np.asarray(s['prediction'][ACTIONS.index(s['action'])][0],float)
        predicted_endpoint=(p+R@np.r_[prediction[:2],0.])[:2]
        Rs=rotation_xyzw(poses[start,3:]);motion=Rs.T@(poses[end,:3]-poses[start,:3])
        relative=Rs.T@rotation_xyzw(poses[end,3:]);actual_yaw=math.atan2(relative[1,0],relative[0,0])
        if np.hypot(prediction[2],prediction[3])<=1e-8:raise ValueError('undefined selected forecast yaw')
        predicted_yaw=math.atan2(prediction[2],prediction[3]);delta_yaw=predicted_yaw-actual_yaw
        # These squares were already witnesses in the decision's own checks.
        cells={tuple(c['nearest_observed_cell']) for c in s['nominal_action_checks']
            if c['nearest_observed_cell'] is not None}
        current=s['proposal']['start_clearance']['nearest_observed_cell']
        if current is not None:cells.add(tuple(current))
        cells=sorted(cells)
        native_map=((poses[start:end+1,:3]-initial[:3])@R0@B.T)[:,:2]
        later=rows[tick+5]['decision']['evidence'];observed_end=None
        if later is not None and later['current_pose'] is not None:
            observed_end=(B@np.asarray(later['current_pose']['position_initial_body_m']))[:2]
        check=s['nominal_action_checks'][s['action_index']]
        terminal_cell=rows[tick+5]['decision']['new_selection']
        terminal_cell=None if terminal_cell is None else terminal_cell['proposal']['start_clearance']['nearest_observed_cell']
        reports.append(dict(tick=tick,action=s['action'],mode=s['mode'],native_start_sample=start,native_end_sample=end,
            actual_body_xy_yaw=[float(motion[0]),float(motion[1]),actual_yaw],
            predicted_body_xy_yaw=[float(prediction[0]),float(prediction[1]),predicted_yaw],
            translation_error_m=float(np.linalg.norm(prediction[:2]-motion[:2])),
            signed_translation_error_m=(prediction[:2]-motion[:2]).tolist(),
            yaw_error_rad=abs(math.atan2(math.sin(delta_yaw),math.cos(delta_yaw))),
            recorded_predicted_all_occupied_clearance=check,
            already_observed_witness_cells=[list(c) for c in cells],
            next_selection_nearest_cell=terminal_cell,
            next_selection_nearest_cell_was_prior_witness=terminal_cell is not None and tuple(terminal_cell) in cells,
            predicted_endpoint_map_xy_m=predicted_endpoint.tolist(),native_endpoint_map_xy_m=native_map[-1].tolist(),
            observed_endpoint_map_xy_m=None if observed_end is None else observed_end.tolist(),
            sampled_native_path=path_witnesses(native_map,cells),
            native_endpoint_witness=nominal_connector(native_map[-1],native_map[-1],cells),
            observed_endpoint_witness=None if observed_end is None else nominal_connector(observed_end,observed_end,cells),
            no_future_geometry_added=True,only_executed_action_labeled=True))
    return dict(root=str(root),case=case,commitments=reports,excluded=excluded,
        mean_translation_error_m=float(np.mean([r['translation_error_m'] for r in reports])) if reports else None,
        maximum_translation_error_m=max((r['translation_error_m'] for r in reports),default=None),
        commitments_with_sampled_native_witness_conflict=sum(r['sampled_native_path']['sampled_radius_conflicts']>0 for r in reports),
        independent_maze_evaluation=False,causal_command_input_changed=False)


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive executed-clearance diagnostic')
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('diagnostic resource allowance unavailable')
    roots={}
    for root,sha in ((INPUT,INPUT_SHA),(PRIOR,PRIOR_SHA),(READOUT,READOUT_SHA)):
        verify_artifacts(root,{'result.json':sha});r=read_json(root,'result.json')
        ids={'result.json':sha,**r.get('artifact_sha256',{})}
        if 'launch_sha256' in r:ids['launch.json']=r['launch_sha256']
        verify_artifacts(root,ids);roots[str(root)]=ids
    original=read_json(READOUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_augmented_commitment_clearance_v1.py',
        'lewm/tests/test_augmented_commitment_clearance_development.py',
        'docs/go2_augmented_family_switch_goal_probe_result_2026-09-08.md'),original['source_sha256'])
    launch=original|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        diagnostic_input_sha256=roots,hardware=resources,workers=1,threads=1,native_execution=False,
        concurrency_reason='three short recorded trajectories; no model forward, fitting or native execution',
        predictions_already_saved_and_raw_replayed=True,evaluator_only_native_labels=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    try:
        reports=[evaluate(INPUT,c[0]) for c in CASES]+[evaluate(PRIOR,'full_direct_'+TRIAL)]
        verify(launch)
        for root,ids in roots.items():
            from pathlib import Path
            verify_artifacts(Path(root),ids)
        write_json(OUTPUT/'result.json',dict(status='AUGMENTED_COMMITMENT_CLEARANCE_DIAGNOSTIC_COMPLETE',
            conditions=reports,source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),
            wall_s=time.perf_counter()-started,no_future_geometry_added=True,
            native_execution=False,model_training=False,original_outcomes_changed=False,
            navigation_qualified=False,goal_achieved=False))
        print('AUGMENTED_COMMITMENT_CLEARANCE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUGMENTED_COMMITMENT_CLEARANCE_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
