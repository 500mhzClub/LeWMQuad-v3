"""Separate native-tape mismatch from horizon-dependent motion forecast error."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw

ROOT = Path('.generated/navigation_development_artifacts_v1/go2_dense_horizon_untimed_exposed_maze_pilot_v1_attempt_002')
OUTPUT = Path('docs/go2_dense_horizon_pilot_motion_diagnostic_2026-09-18.json')


def main():
    assert not OUTPUT.exists()
    read = lambda name:json.loads((ROOT/name).read_text())
    assert read('result.json')['status']=='BOUNDED_PILOT_COMPLETE'
    frames = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    plans = {p['measured_ns']:p for p in read('planning.json') if 'selection' in p}
    requests = {r['simulator_ns']:r for r in read('requests.json')}
    with np.load(ROOT/'native/physics_trace.npz', allow_pickle=False) as archive:
        physics = archive['base_pose_world'].copy()
    rows, skipped = [], defaultdict(int)
    for call in read('dense_model_calls.json'):
        now = call['observed_ns']
        p = plans[now]
        i = ACTIONS.index(p['action'])
        frame = p['frame']
        origin = physics[frames[frame]['physical_sample_index']]
        rotation = rotation_xyzw(origin[3:])
        for h in range(1, 9):
            if frame+h not in frames:
                skipped['recording_end'] += 1
                continue
            times = [now+j*100_000_000+k*20_000_000 for j in range(h) for k in range(5)]
            if any(t not in requests for t in times):
                skipped['missing_requests'] += 1
                continue
            actual_requests = np.asarray([requests[t]['requested_command'] for t in times])
            expected_requests = np.repeat(np.asarray(call['requested_commands'])[i, :h], 5, axis=0)
            if not np.allclose(actual_requests, expected_requests, rtol=0, atol=1e-7):
                skipped['requested_tape_changed'] += 1
                continue
            actual_applied = np.asarray([requests[t]['applied_command'] for t in times])
            expected_applied = np.repeat(np.asarray(call['applied_commands'])[i, :h], 5, axis=0)
            error = float(np.abs(actual_applied-expected_applied).max())
            if error>1e-6:
                skipped['applied_tape_mismatch'] += 1
                continue
            target = physics[frames[frame+h]['physical_sample_index']]
            future_rotation = rotation_xyzw(target[3:])
            yaw = np.arctan2(future_rotation[1, 0], future_rotation[0, 0])-np.arctan2(rotation[1, 0], rotation[0, 0])
            actual = np.r_[((target[:3]-origin[:3])@rotation)[:2], np.arctan2(np.sin(yaw), np.cos(yaw))]
            neural = np.asarray(call['motion_xy_yaw'])[i, h-1]
            command = np.asarray(p['motion_correction']['command_history_forecast_xy_yaw'])[i, h-1]
            errors = {}
            for name, prediction in (('dense', neural), ('command_history', command)):
                difference = prediction-actual
                difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
                errors[name] = difference.tolist()
            rows.append(dict(frame=frame, horizon_ms=h*100, action=p['action'],
                actual_xy_yaw=actual.tolist(), dense_xy_yaw=neural.tolist(), errors=errors,
                maximum_applied_tape_discrepancy=error))
    summary = {}
    for h in range(100, 801, 100):
        selected = [r for r in rows if r['horizon_ms']==h]
        summary[str(h)] = dict(windows=len(selected), models={})
        for name in ('dense', 'command_history'):
            errors = np.asarray([r['errors'][name] for r in selected])
            if len(errors):
                summary[str(h)]['models'][name] = dict(
                    xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(errors[:, :2]**2, axis=1)))),
                    yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(errors[:, 2]**2)))),
                    mean_xy_error_mm=(errors[:, :2].mean(0)*1000).tolist(),
                    mean_yaw_error_deg=float(np.degrees(errors[:, 2].mean())))
    result = dict(root=str(ROOT), summaries=summary, skipped=dict(skipped), rows=rows,
        applied_tapes_checked_at_every_20ms_step=True, native_state_evaluator_only=True,
        no_gpu=True, no_refitting=True, exposed_pilot_only=True, overlapping_selected_windows=True,
        interpretation='Errors at the trained 500-ms readout horizon distinguish a general motion-decoding problem from a problem confined to other horizons; oracle feature decoding is still needed to separate predictor and readout causes.')
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(summaries=summary, skipped=dict(skipped)), indent=2))


if __name__=='__main__':
    main()
