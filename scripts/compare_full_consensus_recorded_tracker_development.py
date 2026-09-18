"""Check every recorded tracker output, excluding only proposal-work counts.

Alternate baseline/candidate order for the first 32 observations to obtain a
small paired timing comparison. Continue the candidate through the full saved
journey; no model, planner or new native scene is created.
"""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.full_consensus_tracker_development import FullConsensusVisualMotion
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedVisualMotion
from scripts import profile_stop_conditioned_early_decisions_development as source

OUTPUT = source.trial.run.BASE/'go2_full_consensus_recorded_tracker_v1_attempt_001'
COUNT = 4740


def without_work_counts(value):
    if isinstance(value, dict):
        return {k:without_work_counts(v) for k,v in value.items() if k != 'valid_proposals'}
    if isinstance(value, list): return [without_work_counts(v) for v in value]
    return value


def write(name, value):
    with (OUTPUT/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def main():
    assert json.loads((source.ROOT/'result.json').read_text())['verified_round_trip'] is True
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        free_disk_bytes=psutil.disk_usage(source.trial.run.BASE).free, shared_host=True)
    if hardware['available_ram_bytes'] < 16*1024**3 or hardware['free_disk_bytes'] < 40*1024**3:
        raise ValueError('retain memory and artifact reserve')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    OUTPUT.mkdir(); owner = psutil.Process()
    write('launch.json', dict(owner=dict(pid=owner.pid, created=owner.create_time()),
        input=str(source.INPUT), observations=COUNT, paired_prefix_observations=32,
        hardware=hardware, source_result_sha256=source.digest(source.ROOT/'result.json'),
        sources={p:source.digest(Path(p)) for p in (
            'lewm/full_consensus_early_exit_development.py',
            'lewm/full_consensus_tracker_development.py',
            'scripts/compare_full_consensus_recorded_tracker_development.py')},
        ignored_comparison_key='valid_proposals', native_execution=False,
        high_level_model_loaded=False, automatic_retry=False))
    candidate = FullConsensusVisualMotion(); baseline = SampledPlaneChainedVisualMotion()
    reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    durations = []; paired = []; started_all = time.perf_counter()
    try:
        with closing(source.packets.read_rows(source.INPUT)) as recorded, (OUTPUT/'frames.jsonl').open('x') as log:
            for frame in range(COUNT):
                row = next(recorded); assert row['tick'] == frame
                expected = without_work_counts(row['decision']['original_visual_evidence'])
                p, d, f, now = reader.packet(frame)
                rgb, depth = source.packets.rgb_packet(source.INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                arms = [('candidate', candidate)]
                if frame < 32:
                    arms += [('baseline', baseline)]
                    if frame % 2: arms.reverse()
                times = {}
                for label, tracker in arms:
                    started = time.perf_counter()
                    actual = tracker.observe(p, d, f, now_ns=now,
                        auxiliary_depth=depth, auxiliary_rgb=rgb)
                    times[label] = time.perf_counter()-started
                    normalized = without_work_counts(json.loads(json.dumps(actual)))
                    if normalized != expected:
                        write('difference.json', dict(frame=frame, arm=label,
                            recorded=expected, candidate=normalized))
                        raise ValueError(f'tracker evidence differs beyond work count at {frame}, {label}')
                    if actual.get('failure') is not None or actual.get('current_pose') is None:
                        raise ValueError('accepted current pose required')
                durations.append(times['candidate'])
                record = dict(frame=frame, candidate_s=times['candidate'],
                    evidence_equal_except_valid_proposals=True)
                if frame < 32:
                    record.update(baseline_s=times['baseline'], timed=frame>=3)
                    paired.append(record)
                log.write(json.dumps(record)+'\n'); log.flush()
                if frame == 31:
                    timed = paired[3:]; old = sum(r['baseline_s'] for r in timed)
                    new = sum(r['candidate_s'] for r in timed)
                    write('paired_prefix.json', dict(rows=paired,
                        baseline_total_s=old, candidate_total_s=new,
                        reduction_percent=100*(1-new/old),
                        baseline_median_ms=float(np.median([r['baseline_s'] for r in timed])*1000),
                        candidate_median_ms=float(np.median([r['candidate_s'] for r in timed])*1000)))
                    print('PAIRED_PREFIX_COMPLETE', old, new, 100*(1-new/old), flush=True)
                if frame % 500 == 0: print('CONSENSUS_TRACKER_FRAME', frame, flush=True)
        values = np.asarray(durations)*1000
        report = dict(status='FULL_CONSENSUS_RECORDED_TRACKER_COMPLETE', observations=COUNT,
            evidence_equal_except_valid_proposals=True, tracker_total_s=sum(durations),
            tracker_median_ms=float(np.median(values)), tracker_p95_ms=float(np.percentile(values, 95)),
            tracker_p99_ms=float(np.percentile(values, 99)), tracker_max_ms=float(values.max()),
            tracker_calls_over_100ms=int((values>100).sum()), wall_s=time.perf_counter()-started_all,
            all_camera_observations_processed=True, camera_hz=10, gyro_role_unchanged=True,
            model_loaded=False, map_or_planner_executed=False, native_execution=False,
            shared_host=True, acquisition_included_in_timing=False,
            full_controller_equivalence_proven=False, real_time_qualified=False, adopted=False)
        write('result.json', report); print(json.dumps(report), flush=True)
    except BaseException as error:
        write('failure.json', dict(reason=repr(error), accepted_observations=len(durations),
            automatic_retry=False)); raise


if __name__ == '__main__': main()
