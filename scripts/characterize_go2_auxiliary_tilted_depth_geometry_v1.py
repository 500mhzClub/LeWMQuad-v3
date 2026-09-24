"""Hypothetical fixed-camera projection using only recorded observed poses."""
import json
import numpy as np
from lewm.auxiliary_tilted_depth_geometry_development import project_square,body_from_optical,CALIBRATION_ID
from scripts.diagnose_go2_training_bias_floor_visibility_v1 import OUTPUT as VISIBILITY
from scripts.run_go2_training_bias_goal_probe_v1 import OUTPUT as INPUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_auxiliary_tilted_depth_geometry_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_tilted_depth_geometry_v1_2026-09-08.md'
VISIBILITY_SHA='a9880fb3a6dd8ae760348a6ff27430448b88373c508a8444e3ac127ea5e71741'
INPUT_SHA='5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5'
CASE='full_jepa_family_episode_039'


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive geometric characterization')
    verify_artifacts(VISIBILITY,{'result.json':VISIBILITY_SHA});prior=read_json(VISIBILITY,'result.json')
    assert prior['status']=='TRAINING_BIAS_FLOOR_VISIBILITY_COMPLETE'
    prior_ids={'result.json':VISIBILITY_SHA,**prior['artifact_sha256']};verify_artifacts(VISIBILITY,prior_ids)
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});probe=read_json(INPUT,'result.json')
    assert probe['status']=='TRAINING_BIAS_GOAL_PROBE_COMPLETE'
    input_ids={'result.json':INPUT_SHA,**probe['artifact_sha256']};verify_artifacts(INPUT,input_ids)
    old=read_json(VISIBILITY,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/characterize_go2_auxiliary_tilted_depth_geometry_v1.py',
        'lewm/tests/test_auxiliary_tilted_depth_geometry_development.py',
        'docs/go2_training_bias_floor_visibility_result_2026-09-08.md'),prior['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<2*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded read-only characterization resources unavailable')
    launch=old|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        visibility_artifact_sha256=prior_ids,probe_artifact_sha256=input_ids,
        auxiliary_body_from_optical=body_from_optical().tolist(),calibration_id=CALIBRATION_ID,
        native_execution=False,rendering_performed=False,geometric_characterization_only=True)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        diagnostic=read_json(VISIBILITY,'visibility.json')['terminal_diagnostics'][0]
        rows=read_json(INPUT/CASE,'context_decisions.json');tape=read_json(INPUT/CASE,'command_tape.json')
        first_translation=next(i for i,r in enumerate(tape) if r['requested_command'][:2]!=[0.,0.])
        reports=[]
        for candidate in diagnostic['candidates']:
            for foot in candidate['feet']:
                projections=[]
                for tick,row in enumerate(rows[:diagnostic['tick']+1]):
                    assert row['tick']==tick
                    decision=row['decision'];receipt=decision['memory_receipt'];pose=decision['evidence']['current_pose']
                    B=np.asarray(receipt['map_from_initial']);R=B@np.asarray(pose['rotation_initial_body_from_current_body'])
                    p=B@np.asarray(pose['position_initial_body_m'])
                    projections.append(dict(frame=tick,**project_square(foot['centre_map_xy_m'],
                        receipt['floor_height_map_m'],R,p)))
                visible=[r['frame'] for r in projections if r['entire_square_in_frustum']]
                reports.append(dict(action=candidate['action'],shape_id=foot['shape_id'],
                    centre_map_xy_m=foot['centre_map_xy_m'],earliest_complete_frustum_frame=min(visible,default=None),
                    complete_frustum_frames_before_first_translation=[i for i in visible if i<=first_translation],
                    projections=projections))
        write_json(OUTPUT/'projection.json',dict(first_translating_command_tick=first_translation,
            includes_observation_before_first_translating_command=True,candidates=reports,
            terminal_foot_regions_used_retrospectively=True,prospective_view_policy=False,
            hypothetical_geometry_only=True,measured_floor_coverage=False))
        verify(launch);verify_artifacts(VISIBILITY,prior_ids);verify_artifacts(INPUT,input_ids)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_TILTED_DEPTH_GEOMETRY_COMPLETE',
            source_sha256=sources,visibility_result_sha256=VISIBILITY_SHA,probe_result_sha256=INPUT_SHA,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','projection.json')},
            hypothetical_geometry_only=True,rendering_performed=False,native_execution=False,
            model_training=False,model_rgb_stream_changed=False,measured_floor_coverage=False,
            robot_self_occlusion_checked=False,hardware_mount_validated=False,
            prospective_navigation_evaluated=False,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_TILTED_DEPTH_GEOMETRY_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_GEOMETRY_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
