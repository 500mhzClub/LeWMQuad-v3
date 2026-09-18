"""Evaluator-only native motion for every previously identified hold/turn pulse."""
from datetime import datetime, timezone
import json
import numpy as np
from scripts import diagnose_go2_hold_reorientation_stall_v1 as prior
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/verify_go2_hold_turn_pulse_native_motion_v1.py'
PRIOR_SHA = 'a8357d863f59d911b3c8fb535a2f1ceff3644afaf8ae84831a9e35de63259ac6'
OUTPUT = ROOT/'docs/go2_hold_turn_pulse_native_motion_2026-09-11.json'


def main():
    if digest(prior.OUTPUT) != PRIOR_SHA: raise ValueError('exact completed pulse population required')
    previous = json.loads(prior.OUTPUT.read_text())
    sources = discover_sources((SOURCE,), previous['source_sha256'] | {str(prior.OUTPUT.relative_to(ROOT)):PRIOR_SHA})
    verify(sources)
    native = prior.prior
    root = native.wait.native.OUTPUT
    verify_artifacts(root, {'result.json':native.NATIVE_SHA, 'launch.json':native.NATIVE_LAUNCH})
    result = read_json(root, 'result.json'); record = result['conditions'][0]; name = record['case']
    ids = {n:result['artifact_sha256'][n] for n in (name+'/physics_trace.npz', name+'/command_tape.json')}
    verify_artifacts(root, ids); tape = read_json(root/name, 'command_tape.json')
    with np.load(root/name/'physics_trace.npz', allow_pickle=False) as archive:
        pose = archive['base_pose_world']; time = archive['timestamp_s']
        requested = archive['requested_command']; applied = archive['applied_command']
        slew = archive['post_slew_applied_command']
    if (pose.shape != (record['collection']['physics_samples'],7) or len(time) != len(pose)
            or not np.isfinite(pose).all() or not np.isfinite(time).all()
            or not np.allclose(np.diff(time), .002, rtol=0, atol=1e-10)
            or not np.allclose(np.linalg.norm(pose[:,3:],axis=1),1.,rtol=0,atol=1e-6)
            or any(c.shape!=(len(pose),3) or not np.isfinite(c).all() for c in (requested,applied,slew))):
        raise ValueError('complete finite original 2ms native pose and command trace required')
    # Recorded quaternions are xyzw; yaw is the native forward azimuth in world XY.
    q = pose[:,3:]; yaw = np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),
        1-2*(q[:,1]**2+q[:,2]**2)))
    windows = []
    for event in previous['intervention_windows']:
        frame = event['frame']; start = 749+50*frame; turn_end = start+50
        stop = start+550 if event['ten_following_completed_holds'] else turn_end
        for f in range(frame, frame+(11 if event['ten_following_completed_holds'] else 1)):
            c = tape[f]; lo = 749+50*f; hi = lo+50
            expected = [0.,0.,.45] if f==frame else [0.,0.,0.]
            if (c['tick']!=f or not c['completed'] or c['requested_command']!=expected
                    or c['pre_sample_index']!=lo or c['post_sample_index']!=hi
                    or not np.allclose(requested[lo+1:hi+1], expected, rtol=0, atol=1e-8)):
                raise ValueError('each actual pulse/hold request and native interval must match')
        row = event | dict(native_first_interval_yaw_rad=float(yaw[turn_end]-yaw[start]),
            native_turn_xy_displacement_m=float(np.linalg.norm(pose[turn_end,:2]-pose[start,:2])),
            turn_applied_yaw_command_range=[float(applied[start+1:turn_end+1,2].min()),float(applied[start+1:turn_end+1,2].max())],
            turn_slew_yaw_command_range=[float(slew[start+1:turn_end+1,2].min()),float(slew[start+1:turn_end+1,2].max())],
            native_sample_bounds=[start,turn_end,stop])
        if event['ten_following_completed_holds']:
            row.update(native_turn_plus_ten_holds_yaw_rad=float(yaw[stop]-yaw[start]),
                native_following_ten_holds_yaw_rad=float(yaw[stop]-yaw[turn_end]),
                native_net_xy_displacement_m=float(np.linalg.norm(pose[stop,:2]-pose[start,:2])))
        windows.append(row)
    pulse = [r for r in windows if r['ten_following_completed_holds']]
    stats = {key:prior.distribution([r[key] for r in population]) for key,population in (
        ('native_first_interval_yaw_rad',windows),('native_turn_plus_ten_holds_yaw_rad',pulse),
        ('native_following_ten_holds_yaw_rad',pulse),('native_net_xy_displacement_m',pulse))}
    verify(sources); verify_artifacts(root, ids | {'result.json':native.NATIVE_SHA,'launch.json':native.NATIVE_LAUNCH})
    write_json(OUTPUT, dict(status='NATIVE_HOLD_TURN_PULSE_REVERSAL_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        prior_diagnosis_sha256=PRIOR_SHA, native_result_sha256=native.NATIVE_SHA,
        reread_native_artifact_sha256=ids, complete_native_physics_samples=len(pose),
        pulse_statistics=stats, intervention_windows=windows,
        native_state_used_only_for_offline_evaluation=True, native_state_supplied_to_controller=False,
        continuous_turn_outcome_inferred=False, full_raw_audit_reexecuted=False,
        native_execution=False, model_inference_reexecuted=False, policy_selected=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
    print('NATIVE_PULSE_REVERSAL_VERIFIED', digest(OUTPUT), len(sources), flush=True)
    print('NATIVE_PULSE_STATISTICS', json.dumps(stats), flush=True)


if __name__ == '__main__': main()
