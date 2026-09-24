"""Reconstruct the unchanged floor-extent rejection from the closed eleventh collection."""
import argparse
import gzip
import json
import time
import cv2
import numpy as np
import torch
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.floor_registered_pose_readout_development import json_identity
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.floor_pose_registration_development import measured_candidates, ROWS, COLUMNS
from lewm.joint_measured_floor_plane_development import fit_joint_plane, compose_plane, validate_joint_plane
from lewm.physical_execution_development import rotation_xyzw
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.run_go2_dual_camera_settled_maze_pilot_v1 import OUTPUT as INPUT, CASE, verify_inputs as verify_native

OUTPUT = BASE/'go2_dual_camera_floor_extent_diagnosis_v1_attempt_001'
PROTOCOL = 'docs/go2_dual_camera_floor_extent_diagnosis_v1_2026-09-09.md'
NATIVE_LAUNCH = 'afb8485c377a12989004cdc7827476948d5308b3b89387a7327b385c2739cba6'
COLLECTION = 'dc5d54e1372b67ae88c783cfb06034777c3e50eccdba40f2aeb1dc5bb3f4fdc5'
FRAMES = (0, 1868, 1872, *range(1894, 1905))
BUDGET = 32*1024**2


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(INPUT, launch['closed_input_sha256'])
    verify_native(read_json(INPUT, 'launch.json'))


def decisions(directory):
    selected = {}; first_failure = None
    with gzip.open(directory/'context_decisions.jsonl.gz', 'rb') as stream:
        for i in range(1905):
            line = stream.readline(32*1024**2+1)
            if not line.endswith(b'\n') or len(line) > 32*1024**2:
                raise ValueError('complete bounded closed decision required')
            # Inspect every terminal field to bind the first failure, not just selected successes.
            row = json.loads(line)
            if row['tick'] != i: raise ValueError('consecutive closed decisions required')
            if row['decision']['failure'] is not None and first_failure is None: first_failure = i
            if i in FRAMES: selected[i] = row['decision']
    if first_failure != 1904: raise ValueError('declared first floor-admission failure changed')
    return selected


def diagnose(directory):
    saved = decisions(directory); reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    reference = saved[0]['evidence']['floor_registration']['reference']
    up = np.asarray(reference['initial_up_body'])
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
        native_poses = z['base_pose_world']
    reports = []; artifacts = {}
    for frame in FRAMES:
        policy, primary, fast, now = reader.packet(frame)
        image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        raw = json_identity(saved[frame]['original_visual_evidence'])
        position, R, pose = current_dual_camera_pose(raw, policy, image, auxiliary, identity=(0,0,0), now_ns=now)
        if frame == 0:
            force = policy['sensor_state']['sensed']['specific_force']['values'].mean(0)
            np.testing.assert_array_equal(up, force/np.linalg.norm(force))
        clouds = []; masks = []
        for depth, E in ((primary, np.asarray(BODY_FROM_OPTICAL)), (auxiliary, body_from_optical())):
            points, mask = measured_candidates(depth['depth_m'], depth['valid'], E, R.T@up)
            clouds.append(points); masks.append(mask)
        fit = fit_joint_plane(*clouds, R.T@up)
        composed = compose_plane(fit['camera_statistics'], R.T@up)
        differences = [k for k,v in composed.items() if fit[k] != v]
        error = None
        try: validate_joint_plane(fit, R.T@up)
        except ValueError as exception: error = str(exception)
        admitted = saved[frame]['evidence']
        if admitted is not None and fit != admitted['floor_registration']['joint_plane']:
            raise ValueError('reconstructed admitted plane differs at frame '+str(frame))
        if frame == 1904:
            if (fit['available'] is not False or fit['reason'] != 'insufficient_combined_two_axis_extent'
                    or differences or error != saved[frame]['failure']):
                raise ValueError('exact original extent rejection must reconstruct')
        elif error is not None: raise ValueError('selected earlier admitted frame unexpectedly failed')
        # All native geometry and segmentation below are postfit diagnostics only.
        native = native_poses[749+50*frame]; native_R = rotation_xyzw(native[3:])
        normal = np.asarray(fit['normal_body']); offset = fit['offset_body_m']
        packet_arrays = {}; point_reports = []
        with np.load(directory/f'auxiliary_depth_{frame:04d}.npz', allow_pickle=False) as z:
            segmentation = z['diagnostic_segmentation']
        for camera, points, mask in zip(('primary','auxiliary'), clouds, masks, strict=True):
            rr, cc = np.nonzero(mask); pixels = np.column_stack((ROWS[rr], COLUMNS[cc]))
            residuals = points@normal+offset
            heights = (points@native_R.T+native[:3])[:,2]
            packet_arrays[camera+'_body_points_m'] = points
            packet_arrays[camera+'_pixels_row_column'] = pixels
            packet_arrays[camera+'_postfit_world_height_m'] = heights
            labels = None
            if camera == 'auxiliary':
                labels = segmentation[pixels[:,0],pixels[:,1]]
                packet_arrays['auxiliary_postfit_segmentation'] = labels
            point_reports.append(dict(camera=camera, count=len(points),
                unadmitted_fit_residual_maximum_m=float(np.abs(residuals).max()) if len(points) else None,
                unadmitted_fit_residual_rms_m=float(np.sqrt(np.mean(residuals**2))) if len(points) else None,
                postfit_world_height_quantiles_m=np.quantile(heights,[0.,.05,.5,.95,1.]).tolist() if len(points) else None,
                auxiliary_robot_candidates=None if labels is None else int(np.isin(labels,acquisitions[frame]['robot_segmentation_ids']).sum()),
                postfit_statistics_change_no_admission_gate=True))
        filename=f'frame_{frame:04d}_diagnostic_candidates.npz'
        np.savez_compressed(OUTPUT/filename, **packet_arrays); artifacts[filename]=digest(OUTPUT/filename)
        eigenvalues=fit['covariance_eigenvalues_m2']
        row=dict(frame=frame, controller_terminal=saved[frame]['terminal'], controller_failure=saved[frame]['failure'],
            selected_visual_camera=raw['camera_selection']['selected_camera'], current_raw_visual_pose_validated=True,
            original_plane_fit=fit, composition_differing_fields=differences, original_validator_error=error,
            stored_admitted_plane_exact=None if admitted is None else True, candidate_diagnostics=point_reports,
            second_eigenvalue_m2=eigenvalues[1], required_second_eigenvalue_m2=.05**2,
            second_axis_standard_deviation_m=float(np.sqrt(max(0.,eigenvalues[1]))),
            native_floor_normal_postfit_angle_rad=float(np.arccos(np.clip(normal@(native_R.T@np.array([0.,0.,1.])), -1., 1.))),
            candidate_artifact=filename, no_candidates_trimmed=True, native_pose_used_for_fitting=False)
        reports.append(row)
        print('FLOOR_EXTENT_FRAME',frame,fit['reason'],eigenvalues[1],flush=True)
    return reports, artifacts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive closed-collection diagnosis required')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    name=CASE[0];directory=INPUT/name
    ids={'launch.json':NATIVE_LAUNCH,name+'/result.json':COLLECTION};verify_artifacts(INPUT,ids)
    collection=read_json(directory,'result.json')
    if (collection['status']!='DUAL_CAMERA_SETTLED_MAZE_TERMINAL_AUDIT_REQUIRED'
            or collection['decisions']!=1915 or collection['physics_samples']!=96450
            or collection['completed_ticks']!=1914 or collection['terminal_zero_ticks']!=10):
        raise ValueError('complete fixed eleventh collection required')
    names=['context_decisions.jsonl.gz','policy_observations.json','policy_histories.npz','depth_observations.json',
        'fast_gyro_histories.npz','auxiliary_camera_audit.json','physics_trace.npz']
    for frame in FRAMES:
        names.extend((f'rgb_{frame:04d}.png',f'depth_{frame:04d}.npz',
            f'auxiliary_rgb_{frame:04d}.png',f'auxiliary_depth_{frame:04d}.npz'))
    ids.update({name+'/'+n:digest(directory/n) for n in names})
    old=read_json(INPUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_dual_camera_floor_extent_v1.py',
        'lewm/tests/test_joint_measured_floor_plane_development.py'),old['source_sha256'])
    resources=hardware()
    launch=dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),closed_input_sha256=ids,
        hardware=resources,frames=list(FRAMES),first_failure_frame=1904,minimum_available_ram_bytes=8*1024**3,
        output_allowance_bytes=BUDGET,minimum_free_bytes=40*1024**3,os_resource_limits_enforced=False,
        cpu_processes=1,numerical_threads=1,native_scene_workers=0,native_execution=False,model_inference=False,
        native_pose_used_for_fitting=False,native_geometry_and_segmentation_postfit_only=True,
        concurrency_reason='one bounded CPU diagnosis beside the existing native raw audit; no scene',
        full_native_audit_still_separate=True,final_native_artifact_binding_match_required_before_future_navigation=True)
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=8*1024**3
    storage_ok=resources['artifact_free_bytes']>=40*1024**3+BUDGET
    if args.preflight_only:
        print('FLOOR_EXTENT_PREFLIGHT',json.dumps(dict(source_count=len(sources),input_count=len(ids),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('bounded floor diagnosis resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('FLOOR_EXTENT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        reports,bindings=diagnose(directory)
        bindings['launch.json']=digest(OUTPUT/'launch.json');verify_all(launch);verify_artifacts(OUTPUT,bindings)
        if sum((OUTPUT/n).stat().st_size for n in bindings)>BUDGET//2:raise ValueError('diagnosis artifact headroom exceeded')
        write_json(OUTPUT/'result.json',dict(status='DUAL_CAMERA_FLOOR_EXTENT_DIAGNOSIS_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,frames=reports,first_failure_frame=1904,
            original_rejection_reconstructed=True,original_outcome_unchanged=True,thresholds_changed=False,
            model_inference=False,native_execution=False,native_pose_used_for_fitting=False,
            full_native_audit_still_separate=True,final_native_artifact_binding_match_required_before_future_navigation=True,
            navigation_qualified=False,goal_achieved=False,wall_s=time.perf_counter()-started,hardware_after=hardware()))
        print('FLOOR_EXTENT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()
