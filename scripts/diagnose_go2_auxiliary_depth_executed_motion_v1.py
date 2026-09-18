"""Native error and stopping witnesses, restricted to actually executed prefixes."""
import time
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.executed_prefix_motion_diagnosis_development import diagnose
from lewm.physical_execution_development import rotation_xyzw
from lewm.observed_geometry_refinement_development import segment_cell_distances
from scripts.run_go2_auxiliary_depth_goal_probe_v1 import OUTPUT as INPUT,CASES
from scripts.read_go2_auxiliary_depth_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json

INPUT_SHA='f8bcad37fdf1623c30eb6782c4514b483473e8c6a4f3e404d77ab71cc0f80f21'
READOUT_SHA='db1440338a68b8d8c2a04a7c11fa7b43e6bccd79ee428f607202936e6ddcfacb'
OUTPUT=BASE/'go2_auxiliary_depth_executed_motion_diagnosis_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_depth_executed_motion_diagnosis_v1_2026-09-08.md'


def analyze(case):
    directory=INPUT/case;rows=read_json(directory,'context_decisions.json');tape=read_json(directory,'command_tape.json')
    with np.load(directory/'physics_trace.npz',allow_pickle=False) as z:poses=z['base_pose_world']
    anchor=poses[749];initial=rotation_xyzw(anchor[3:]);reports=[]
    for row in rows:
        tick=row['tick'];selection=row['decision']['new_selection']
        if not selection or 'prediction' not in selection:continue
        B=np.asarray(row['decision']['memory_receipt']['map_from_initial'])
        for i,action in enumerate(ACTIONS):
            targets=diagnose(selection['prediction'][i],candidate_commands(action)[:8],tape,poses,tick=tick)
            if not targets:continue
            first=targets[0];check=selection['nominal_action_checks'][i];cell=check['nearest_observed_cell']
            curve=(poses[first['start_sample']:first['end_sample']+1,:3]-anchor[:3])@initial@B.T
            clearance=None if cell is None else [float(segment_cell_distances(p[:2],p[:2],[cell])[0]) for p in curve]
            reports.append(dict(tick=tick,forecast_action=action,selected_action=selection['action'],
                terminal=row['decision']['terminal'],executed_command_prefix_horizons=len(targets),targets=targets,
                original_first_step_nominal_check=check,
                original_surface_possible_intersection=selection['surface_checks'][i]['possible_intersection'],
                native_initial_anchor_map_curve_start_xy_m=curve[0,:2].tolist(),
                native_initial_anchor_map_curve_end_xy_m=curve[-1,:2].tolist(),
                sampled_native_centre_clearance_to_forecast_nearest_cell_m=clearance,
                minimum_sampled_native_centre_clearance_to_forecast_nearest_cell_m=min(clearance) if clearance is not None else None,
                all_obstacles_reaudited=False,continuous_path_certified=False,native_data_evaluator_only=True))
    first=[r['targets'][0]['xy_error_m'] for r in reports]
    terminal=[r for r in reports if r['terminal'] is not None]
    assert len(terminal)==1 and terminal[0]['forecast_action']=='hold' and terminal[0]['executed_command_prefix_horizons']==8
    crossings=[dict(tick=r['tick'],action=r['forecast_action'],
        predicted_clearance_m=r['original_first_step_nominal_check']['minimum_observed_cell_distance_m'],
        actual_sampled_clearance_m=r['minimum_sampled_native_centre_clearance_to_forecast_nearest_cell_m'])
        for r in reports if r['original_first_step_nominal_check']['nominal_disk_connector_clear']
        and r['minimum_sampled_native_centre_clearance_to_forecast_nearest_cell_m'] is not None
        and r['minimum_sampled_native_centre_clearance_to_forecast_nearest_cell_m']<=.45+1e-12]
    return dict(case=case,forecasts_with_actual_prefix=len(reports),first_100ms_xy_error_mean_m=float(np.mean(first)),
        first_100ms_xy_error_max_m=max(first),predicted_clear_but_actual_sampled_centre_nominal_conflicts=crossings,
        terminal_hold=terminal[0],last_five_forecasts=reports[-5:],all_forecasts=reports,
        unexecuted_outcomes_inferred=False)


def main():
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive diagnosis required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});result=read_json(INPUT,'result.json')
    assert result['status']=='AUXILIARY_DEPTH_GOAL_PROBE_COMPLETE' and result['all_measurement_gates_pass']
    ids={'result.json':INPUT_SHA,**result['artifact_sha256']};verify_artifacts(INPUT,ids)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['status']=='AUXILIARY_DEPTH_GOAL_READOUT_COMPLETE' and readout['probe_result_sha256']==INPUT_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    old=read_json(READOUT,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_auxiliary_depth_executed_motion_v1.py',
        'lewm/tests/test_executed_prefix_motion_diagnosis_development.py',
        'docs/go2_auxiliary_depth_goal_probe_result_2026-09-08.md'),old['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded diagnostic resource allowance unavailable')
    launch=old|dict(source_sha256=sources,output_root=str(OUTPUT),protocol=PROTOCOL,hardware=resources,
        input_artifact_sha256=ids,readout_artifact_sha256=readout_ids,native_execution=False,model_training=False,
        controller_changed=False,workers=1,diagnostic_only=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    try:
        reports=[analyze(c[0]) for c in CASES]
        write_json(OUTPUT/'motion.json',dict(conditions=reports))
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(READOUT,readout_ids)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json','motion.json')};verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_DEPTH_EXECUTED_MOTION_DIAGNOSIS_COMPLETE',
            source_sha256=sources,input_result_sha256=INPUT_SHA,readout_result_sha256=READOUT_SHA,
            artifact_sha256=bindings,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            native_execution=False,model_training=False,controller_changed=False,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_DEPTH_EXECUTED_MOTION_DIAGNOSIS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_DEPTH_MOTION_DIAGNOSIS_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
