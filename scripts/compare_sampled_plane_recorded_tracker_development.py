"""Time the sampled tracker against complete recorded visual evidence.

One sequential traversal of the completed outward/return sensor history.
No high-level model, map, planner, simulator or command execution is created.
"""
from collections import Counter
from contextlib import closing
import hashlib
import json
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedVisualMotion
from scripts import profile_stop_conditioned_early_decisions_development as source

OUTPUT = source.trial.run.BASE/'go2_sampled_plane_recorded_tracker_v1_attempt_001'
COUNT = 4740


def write(name, value):
    with (OUTPUT/name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False); out.write('\n')


def main():
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        artifact_free_bytes=psutil.disk_usage(source.trial.run.BASE).free,
        cpu_percent=psutil.cpu_percent(interval=.2), shared_host=True,
        one_additional_cpu_tracker=True)
    if hardware['available_ram_bytes'] < 16*1024**3 or hardware['artifact_free_bytes'] < 40*1024**3:
        raise ValueError('retain memory and artifact reserve')
    collection = json.loads((source.INPUT/'result.json').read_text())
    assert collection['decisions'] == 4750
    assert collection['schedule_terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    assert json.loads((source.ROOT/'result.json').read_text())['verified_round_trip'] is True
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    OUTPUT.mkdir()
    owner = psutil.Process()
    write('launch.json', dict(owner=dict(pid=owner.pid, created=owner.create_time()),
        observations=COUNT, hardware=hardware, input=str(source.INPUT),
        verified_native_result_sha256=source.digest(source.ROOT/'result.json'),
        source_sha256={p:source.digest(source.Path(p)) for p in (
            'lewm/sampled_plane_candidates_development.py',
            'lewm/sampled_plane_stop_conditioned_controller_development.py',
            'scripts/compare_sampled_plane_recorded_tracker_development.py')},
        native_execution=False, high_level_model_loaded=False, automatic_retry=False))
    print('SAMPLED_TRACKER_STARTED', owner.pid, owner.create_time(), str(OUTPUT), flush=True)
    durations = []; cameras = Counter(); fallbacks = 0; digest = hashlib.sha256()
    started_all = time.perf_counter()
    try:
        tracker = SampledPlaneChainedVisualMotion()
        reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
        auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
        with closing(source.packets.read_rows(source.INPUT)) as recorded, (OUTPUT/'frames.jsonl').open('x') as progress:
            for frame in range(COUNT):
                expected_row = next(recorded)
                assert expected_row['tick'] == frame
                expected = expected_row['decision']['original_visual_evidence']
                p, d, f, now = reader.packet(frame)
                image, depth = source.packets.rgb_packet(source.INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                started = time.perf_counter()
                result = tracker.observe(p, d, f, now_ns=now,
                    auxiliary_depth=depth, auxiliary_rgb=image)
                duration = time.perf_counter()-started
                normalized = json.loads(json.dumps(result))
                if normalized != expected:
                    write('difference.json', dict(frame=frame, recorded=expected, candidate=normalized))
                    raise ValueError(f'complete recorded tracker result differs at frame {frame}')
                if result.get('failure') is not None or result.get('current_pose') is None:
                    raise ValueError(f'recorded and candidate trackers both unavailable at frame {frame}')
                digest.update(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode())
                camera = result['camera_selection'].get('selected_camera')
                fallback = bool((result.get('chained_anchor_fallback') or {}).get('accepted'))
                cameras[str(camera)] += 1; fallbacks += fallback; durations.append(duration)
                progress.write(json.dumps(dict(frame=frame, full_tracker_result_equal=True,
                    tracker_s=duration, camera=camera, accepted_chained_fallback=fallback,
                    retained_references=len(tracker.model.references)))+'\n'); progress.flush()
                if frame%250 == 0:
                    print('SAMPLED_TRACKER_FRAME', frame, flush=True)
        values = np.asarray(durations)*1000
        write('result.json', dict(status='SAMPLED_RECORDED_TRACKER_COMPARISON_COMPLETE',
            observations=COUNT, full_recorded_tracker_results_equal=True,
            cameras=dict(cameras), accepted_chained_fallback_frames=fallbacks,
            tracker_total_s=sum(durations), tracker_median_ms=float(np.median(values)),
            tracker_p95_ms=float(np.percentile(values, 95)), tracker_p99_ms=float(np.percentile(values, 99)),
            tracker_max_ms=float(values.max()), tracker_calls_over_100ms=int((values>100).sum()),
            recorded_tracker_results_sha256=digest.hexdigest(), wall_s=time.perf_counter()-started_all,
            shared_host=True, acquisition_included_in_timing=False, profiler_used=False,
            model_loaded=False, map_or_planner_executed=False, native_execution=False,
            full_controller_equivalence_proven=False, real_time_qualified=False, adopted=False))
        print('SAMPLED_TRACKER_COMPLETE', float(np.median(values)), int((values>100).sum()), flush=True)
    except BaseException as error:
        write('failure.json', dict(reason=repr(error), completed_observations=len(durations),
            automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
