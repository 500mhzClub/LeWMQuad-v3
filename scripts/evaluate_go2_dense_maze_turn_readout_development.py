"""CPU-only true-future readout diagnosis on completed maze-00 turn windows.

Eight chronologically spaced windows per turn/history group; no error-based
selection, fitting, navigation, or new predictor inference. Future RGB and
native poses are offline diagnostic inputs only.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import train_go2_full_heading_readout_development as fit

ROOT = Path('.generated/navigation_development_artifacts_v1/'
    'go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001')
OUTPUT = ROOT/'turn_true_future_readout_v1'


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


def metrics(rows):
    result = {'windows': len(rows)}
    for arm in ('saved_prediction', 'observed_future', 'command_history', 'identity'):
        errors = np.asarray([r['errors'][arm] for r in rows])
        result[arm] = dict(
            xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(errors[:, :2]**2, axis=1)))),
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(errors[:, 2]**2)))),
            mean_yaw_deg=float(np.degrees(np.mean([r['predictions'][arm][2] for r in rows]))),
            wrong_yaw_sign_when_actual_exceeds_one_degree=sum(
                abs(r['actual'][2]) > np.pi/180 and r['actual'][2]*r['predictions'][arm][2] < 0
                for r in rows))
    result['actual_exceeds_one_degree'] = sum(abs(r['actual'][2]) > np.pi/180 for r in rows)
    result['mean_actual_yaw_deg'] = float(np.degrees(np.mean([r['actual'][2] for r in rows])))
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    read = lambda name: json.loads((ROOT/name).read_text())
    assert read('result.json')['status'] == 'FULL_MISSION_RUN_COMPLETE'
    launch = read('launch.json')
    assert launch['model_assignment'] == 'action'
    assert launch['motion_readout']['arm'] == 'mixed_data'
    head = fit.load('mixed_data')
    head_sha = fit.original.digest(fit.OUTPUT/'mixed_data_final.pt')
    assert head_sha == launch['motion_readout']['sha256']
    # Check any retirement receipt before historical image access. This run
    # was recorded RGB-only; depth is not required by this diagnostic.
    retention = {}
    for name in ('depth_retention.json', 'native/depth_retention.json'):
        if (ROOT/name).exists():
            retention[name] = read(name)
    plans = {p['frame']: p for p in read('planning.json') if 'selection' in p}
    by_time = {p['measured_ns']: p for p in plans.values()}
    calls = {by_time[c['observed_ns']]['frame']: c for c in read('dense_model_calls.json')}
    previous = read('dense_navigation_readout.json')
    yaw_rows = {r['frame']: r for r in previous['same_window_yaw']['rows']}
    groups = defaultdict(list)
    for row in previous['executed_windows']['rows']:
        frame = row['frame']; action = plans[frame]['action']
        if action not in ('left_turn', 'right_turn'):
            continue
        last = calls[frame]['last_applied_command'][2]
        if abs(last) < .01:
            continue
        direction = 1 if action == 'left_turn' else -1
        group = action + ('_same' if last*direction > 0 else '_opposite')
        groups[group].append(frame)
    assert len(groups) == 4 and all(len(v) >= 8 for v in groups.values())
    selected = {group: [sorted(frames)[i] for i in np.linspace(0, len(frames)-1, 8, dtype=int)]
        for group, frames in sorted(groups.items())}
    frames = {r['frame']: r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']: r for r in read('requests.json')}
    with np.load(ROOT/'native/physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world'].copy()
    rows = []
    for group, departures in selected.items():
        for frame in departures:
            plan, call = plans[frame], calls[frame]
            i = ACTIONS.index(plan['action']); now = call['observed_ns']
            times = [now+j*100_000_000+k*20_000_000 for j in range(7) for k in range(5)]
            for field in ('requested', 'applied'):
                actual = np.asarray([requests[t][field+'_command'] for t in times])
                expected = np.repeat(np.asarray(call[field+'_commands'])[i, :7], 5, axis=0)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            origin = poses[frames[frame]['physical_sample_index']]
            rotation = rotation_xyzw(origin[3:])
            for h in (5, 7):
                future = poses[frames[frame+h]['physical_sample_index']]
                future_rotation = rotation_xyzw(future[3:])
                yaw = wrapped(np.arctan2(future_rotation[1, 0], future_rotation[0, 0])
                    - np.arctan2(rotation[1, 0], rotation[0, 0]))
                actual = np.r_[((future[:3]-origin[:3])@rotation)[:2], yaw]
                predicted = np.asarray(call['motion_xy_yaw'])[i, h-1]
                command = np.asarray(plan['motion_correction']['command_history_forecast_xy_yaw'])[i, h-1]
                if h == 7:
                    # The existing reader decodes the stored float32 sin/cos
                    # representation, not the pre-conversion raw yaw scalar.
                    encoded = np.asarray(plan['motion_correction']['upstream_prediction_for_yaw_ablation'])[i, h-1]
                    stored_yaw = np.arctan2(encoded[2], encoded[3])
                    np.testing.assert_allclose(wrapped(stored_yaw-predicted[2]), 0., rtol=0, atol=1e-6)
                    for arm, value in [('neural', stored_yaw), ('command_history', command[2])]:
                        np.testing.assert_allclose(wrapped(value-yaw),
                            yaw_rows[frame]['errors_rad'][arm], rtol=0, atol=1e-10)
                rows.append(dict(frame=frame, group=group, horizon_ms=100*h,
                    actual=actual.tolist(), predictions=dict(saved_prediction=predicted.tolist(),
                        command_history=command.tolist())))
    image_frames = sorted({r['frame']+offset for r in rows for offset in (0, r['horizon_ms']//100)})
    assert all((ROOT/'native'/f'rgb_{f:04d}.png').is_file() for f in image_frames)
    OUTPUT.mkdir(exist_ok=False)
    save('plan.json', dict(root=str(ROOT), device='cpu', selected=selected,
        eligible_groups={k: len(v) for k, v in groups.items()}, horizons_ms=[500, 700],
        selection='eight equally spaced chronological indices per turn/last-command-sign group',
        head_sha256=head_sha, source_sha256=fit.original.digest(__file__),
        depth_retention_receipts=retention, unique_images=len(image_frames),
        no_fitting=True, future_images_offline_only=True, no_navigation=True,
        requested_and_applied_tapes_matched=True, saved_yaw_errors_reproduced=True))
    started = time.monotonic()
    try:
        encoder = VJepa21Arm(); encoder.build(torch.device('cpu'), torch.float32)
        features = {}
        for j, frame in enumerate(image_frames):
            pixels = encoder.preprocess(str(ROOT/'native'/f'rgb_{frame:04d}.png'))[None]
            tokens = F.layer_norm(encoder.tokens(pixels).float(), (1024,))
            features[frame] = pool_tokens(tokens)
            print('TURN_TRUE_FUTURE_FEATURE', j+1, len(image_frames), round(time.monotonic()-started, 1), flush=True)
        del encoder
        for row in rows:
            current = features[row['frame']]
            future = features[row['frame']+row['horizon_ms']//100]
            row['predictions']['observed_future'] = head(current, future)[0].tolist()
            row['predictions']['identity'] = head(current, current)[0].tolist()
            row['errors'] = {}
            for arm, prediction in row['predictions'].items():
                error = np.asarray(prediction)-row['actual']; error[2] = wrapped(error[2])
                row['errors'][arm] = error.tolist()
        summaries = {str(h): {g: metrics([r for r in rows if r['horizon_ms'] == h and r['group'] == g])
            for g in selected} for h in (500, 700)}
        save('result.json', dict(status='COMPLETE', rows=rows, summaries=summaries,
            wall_s=time.monotonic()-started,
            limitations=['post hoc development diagnosis; selected overlapping windows',
                'true-future readout error combines encoder and readout limitations',
                'predicted versus true features is not an additive error decomposition',
                'no new navigation outcome, fitting, or causal training intervention']))
        print('TURN_TRUE_FUTURE_COMPLETE', json.dumps(summaries), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
