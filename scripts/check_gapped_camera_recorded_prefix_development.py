"""Prospective fixed camera-gap pattern on 201 recorded moving observations.

All gyro packets are consumed. Native pose is loaded only after tracking stops,
for an evaluator-only accuracy readout. This executes no new navigation.
"""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.gapped_camera_plane_tracker_development import GappedCameraPlaneTracker
from lewm.physical_execution_development import rotation_xyzw
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts import profile_stop_conditioned_early_decisions_development as source

BASE = source.trial.run.BASE
CASE = 'independent_00_frozen_reference_seed_2026091001_full_jepa'
INPUT_ROOT = BASE/f'go2_stop_conditioned_{CASE}_v1_attempt_001'
INPUT = INPUT_ROOT/CASE
OUTPUT = BASE/'go2_gapped_camera_plane_tracker_prefix_v1_attempt_001'
COUNT = 201
GAPS = (1, 2, 3, 4, 5)


def write(name, value):
    with (OUTPUT/name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False); out.write('\n')


def main():
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        artifact_free_bytes=psutil.disk_usage(BASE).free,
        cpu_percent=psutil.cpu_percent(interval=.2), shared_host=True)
    if hardware['available_ram_bytes'] < 16*1024**3 or hardware['artifact_free_bytes'] < 40*1024**3:
        raise ValueError('retain memory and artifact reserve')
    assert json.loads((INPUT_ROOT/'result.json').read_text())['verified_round_trip'] is True
    OUTPUT.mkdir(); owner = psutil.Process()
    write('launch.json', dict(owner=dict(pid=owner.pid, created=owner.create_time()),
        input=str(INPUT), observations=COUNT, repeated_camera_gap_intervals=GAPS,
        hardware=hardware, native_execution=False, high_level_model_loaded=False,
        source_result_sha256=source.digest(INPUT_ROOT/'result.json'),
        sources={p:source.digest(Path(p)) for p in (
            'lewm/gapped_camera_plane_tracker_development.py',
            'lewm/camera_independent_gyro_development.py',
            'scripts/check_gapped_camera_recorded_prefix_development.py')}, automatic_retry=False))
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    tracker = GappedCameraPlaneTracker(); records = []; failure = None
    next_visual = 0; gap_index = 0; ingested = 0; started_all = time.perf_counter()
    try:
        reader = source.packets.ExtendedReturnBudgetRGBDReplay(INPUT)
        auxiliary = json.loads((INPUT/'auxiliary_camera_audit.json').read_text())
        with closing(source.packets.read_rows(INPUT)) as recorded, (OUTPUT/'visual_results.jsonl').open('x') as output:
            for frame in range(COUNT):
                expected = next(recorded); assert expected['tick'] == frame
                p, d, f, now = reader.packet(frame)
                tracker.ingest_gyro(p, f, now_ns=now); ingested += 1
                if frame != next_visual:
                    continue
                image, depth = source.packets.rgb_packet(INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                started = time.perf_counter()
                result = tracker.observe(p, d, f, now_ns=now,
                    auxiliary_depth=depth, auxiliary_rgb=image)
                duration = time.perf_counter()-started
                assert result['acquisition_frame'] == frame and result['measured_ns'] == now
                assert result['continuous_gyro_intervals'] == frame*50
                output.write(json.dumps(dict(source_frame=frame, result=result,
                    continuity=tracker.last_continuity))+'\n'); output.flush()
                dense = expected['decision']['original_visual_evidence']['current_pose']
                records.append(dict(source_frame=frame, processed_frame=result['frame'],
                    pre_sample_index=expected['pre_sample_index'], measured_ns=now,
                    visual_interval_ns=result['visual_interval_ns'], tracker_ms=duration*1000,
                    position_initial_body_m=result['position_initial_body_m'],
                    rotation_initial_body_from_current_body=result['rotation_initial_body_from_current_body'],
                    dense_position_initial_body_m=dense['position_initial_body_m'],
                    dense_rotation_initial_body_from_current_body=dense['rotation_initial_body_from_current_body'],
                    camera=result['camera_selection']['selected_camera'],
                    continuity_status=tracker.last_continuity['status']))
                next_visual += GAPS[gap_index%len(GAPS)]; gap_index += 1
        assert ingested == COUNT
    except BaseException as error:
        failure = dict(reason=repr(error), gyro_packets_ingested=ingested,
            accepted_visual_observations=len(records), next_visual_frame=next_visual,
            tracker_failed=tracker.failed, automatic_retry=False)
        write('failure.json', failure)

    # Evaluation begins after the tracker has stopped consuming public inputs.
    with np.load(INPUT/'physics_trace.npz', allow_pickle=False) as raw:
        poses = raw['base_pose_world']
    origin = poses[749]; R0 = rotation_xyzw(origin[3:])
    for row in records:
        assert row['pre_sample_index'] == 749+50*row['source_frame']
        native = poses[row['pre_sample_index']]
        p = (native[:3]-origin[:3])@R0
        R = R0.T@rotation_xyzw(native[3:])
        row.update(native_position_error_m=float(np.linalg.norm(np.asarray(row['position_initial_body_m'])-p)),
            native_rotation_error_rad=angle(np.asarray(row['rotation_initial_body_from_current_body']).T@R),
            dense_native_position_error_m=float(np.linalg.norm(np.asarray(row['dense_position_initial_body_m'])-p)),
            dense_native_rotation_error_rad=angle(np.asarray(row['dense_rotation_initial_body_from_current_body']).T@R))
    report = dict(status='GAPPED_CAMERA_RECORDED_PREFIX_COMPLETE' if failure is None else 'GAPPED_CAMERA_RECORDED_PREFIX_FAILED',
        gyro_packets_ingested=ingested, accepted_visual_observations=len(records), records=records,
        failure=failure, wall_s=time.perf_counter()-started_all,
        native_pose_loaded_only_after_tracking=True, native_pose_used_for_estimation=False,
        native_trace_sha256=source.digest(INPUT/'physics_trace.npz'),
        processing_delay_simulated=False, recorded_camera_subsampling_only=True,
        fixed_cadence_controller_compatible=False, navigation_executed=False,
        real_time_qualified=False, hardware_qualified=False)
    for field in ('native_position_error_m', 'native_rotation_error_rad',
            'dense_native_position_error_m', 'dense_native_rotation_error_rad', 'tracker_ms'):
        values = [r[field] for r in records]
        report[field] = dict(median=float(np.median(values)), maximum=max(values)) if values else None
    write('result.json' if failure is None else 'partial_result.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)
    if failure is not None:
        raise RuntimeError(failure['reason'])


if __name__ == '__main__':
    main()
