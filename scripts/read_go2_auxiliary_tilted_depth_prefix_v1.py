"""Summarize all raw sensor frames and quantify primary image compatibility."""
import numpy as np
from PIL import Image
from scripts.capture_go2_auxiliary_tilted_depth_prefix_integrity_v2 import OUTPUT as INPUT
from scripts.run_go2_training_bias_goal_probe_v1 import OUTPUT as PRIOR
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_auxiliary_tilted_depth_prefix_readout_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_tilted_depth_prefix_readout_v1_2026-09-08.md'
INPUT_SHA='e8de0873c71f50c5a01793c0464ba2014c1a7fe72f77daf47204eaeee1c8c38a'
PRIOR_SHA='5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5'
CASE='full_jepa_family_episode_039'


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive sensor readout required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});result=read_json(INPUT,'result.json')
    assert result['status']=='AUXILIARY_TILTED_DEPTH_PREFIX_INTEGRITY_V2_COMPLETE' and result['raw_audit_pass']
    ids={'result.json':INPUT_SHA,**result['artifact_sha256']};verify_artifacts(INPUT,ids)
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA});previous=read_json(PRIOR,'result.json')
    old_ids={'result.json':PRIOR_SHA,**previous['artifact_sha256']};verify_artifacts(PRIOR,old_ids)
    old=read_json(INPUT,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_auxiliary_tilted_depth_prefix_v1.py'),result['source_sha256'])
    launch=old|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        input_artifact_sha256=ids,prior_artifact_sha256=old_ids,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        audit=read_json(INPUT,'raw_audit.json');new=INPUT/'sensor_prefix';prior=PRIOR/CASE;differences=[]
        camera=read_json(new,'auxiliary_camera_audit.json');assert len(camera)==20
        for i in range(20):
            with Image.open(prior/f'rgb_{i:04d}.png') as im:a=np.array(im)
            with Image.open(new/f'rgb_{i:04d}.png') as im:b=np.array(im)
            assert a.shape==b.shape==(480,640,3) and a.dtype==b.dtype==np.uint8
            delta=np.abs(a.astype(np.int16)-b.astype(np.int16));pixels=np.any(delta!=0,axis=-1)
            with np.load(prior/f'native_depth_{i:04d}.npz',allow_pickle=False) as z:d0=z['optical_depth_m']
            with np.load(new/f'native_depth_{i:04d}.npz',allow_pickle=False) as z:d1=z['optical_depth_m']
            same_depth=np.array_equal(d0,d1)
            same_rgb=not bool(pixels.any());assert same_rgb==audit['primary_rgb_exact_by_frame'][i]
            differences.append(dict(frame=i,primary_rgb_exact=same_rgb,changed_rgb_pixels=int(pixels.sum()),
                maximum_absolute_channel_difference=int(delta.max()),mean_absolute_channel_difference=float(delta.mean()),
                primary_native_depth_exact=same_depth,changed_native_depth_pixels=int(np.count_nonzero(d0!=d1))))
        coverage=[]
        for j,c in enumerate(audit['coverage_by_frame'][-1]['candidates']):
            first=next((r['frame'] for r in audit['coverage_by_frame'] if r['candidates'][j]['complete_nominal_foot_patch']),None)
            coverage.append(dict(action=c['action'],first_complete_measured_floor_frame=first,terminal_prefix_witness=c['coverage_witness']))
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(PRIOR,old_ids)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_TILTED_DEPTH_PREFIX_READOUT_COMPLETE',
            source_sha256=sources,input_result_sha256=INPUT_SHA,launch_sha256=digest(OUTPUT/'launch.json'),
            raw_prefix_array_sha256=audit['exact_physical_and_public_prefix_sha256'],frames=20,command_ticks=19,
            primary_differences=differences,all_primary_depth_exact=all(r['primary_native_depth_exact'] for r in differences),
            all_primary_rgb_exact=all(r['primary_rgb_exact'] for r in differences),coverage=coverage,
            robot_pixels_by_frame=audit['robot_pixels_by_frame'],
            auxiliary_capture_wall_s=[r['capture_wall_s'] for r in camera],
            includes_diagnostic_rgb_segmentation_and_io=True,simultaneous_camera_timing_validated=False,
            retrospective_original_observed_poses=True,retrospective_terminal_regions=True,
            new_observer_evaluated=False,model_input_compatibility_validated=False,
            native_execution=False,model_inference=False,prospective_navigation_evaluated=False,
            hardware_mount_validated=False,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_TILTED_DEPTH_PREFIX_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_SENSOR_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
