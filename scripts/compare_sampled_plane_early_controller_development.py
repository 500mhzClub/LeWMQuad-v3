"""Thirteen recorded controller decisions, alternating baseline/candidate order.

Measures ten calls after three warmup observations. No simulator is created,
and this diagnostic does not install the candidate in any experiment runner.
"""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from scripts import profile_stop_conditioned_early_decisions_development as source
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneStopConditionedController

OUTPUT = Path('docs/go2_sampled_plane_early_controller_2026-09-13.json')
FAILURE = OUTPUT.with_suffix('.failure.json')


def main():
    if OUTPUT.exists() or FAILURE.exists():
        raise ValueError('preserve existing diagnostic output')
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        available_disk_bytes=psutil.disk_usage(source.trial.run.BASE).free,
        cpu_percent=psutil.cpu_percent(interval=.2), shared_host=True,
        one_short_cpu_comparison=True)
    if hardware['available_ram_bytes'] < 16*1024**3 or hardware['available_disk_bytes'] < 40*1024**3:
        raise ValueError('retain memory and artifact reserve')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    try:
        admission_root = source.trial.run.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
        admission = json.loads((admission_root/'launch.json').read_text())['input_admission']['correction_admission']
        model, condition, variant = source.load_assigned(admission, 'seed_2026091001_no_rgb_direct')
        before = source.trial.previous.state_digest(model.state_dict())
        assert before == json.loads((source.ROOT/'launch.json').read_text())['model_state_sha256']
        public_mission = json.loads((source.INPUT/'public_mission.json').read_text())
        arms = [cls(model, source.trial.previous.geometry_factory(source.trial.previous.URDF),
            public_mission=public_mission, navigation_ticks=8000, condition=condition,
            variant=variant, persistent=True) for cls in
            (source.StopConditionedSettlingController, SampledPlaneStopConditionedController)]
        reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
        auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
        with closing(source.packets.read_rows(source.INPUT)) as recorded:
            for frame in range(13):
                expected = next(recorded)['decision']
                p, d, f, now = reader.packet(frame)
                image, depth = source.packets.rgb_packet(source.INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                durations = [None, None]
                for arm in ((0, 1) if frame%2 == 0 else (1, 0)):
                    started = time.perf_counter()
                    actual = arms[arm].observe(p, d, f, now_ns=now,
                        auxiliary_depth=depth, auxiliary_rgb=image)
                    durations[arm] = time.perf_counter()-started
                    if arm == 1:
                        assert actual.pop('sampled_plane_candidates_enabled') is True
                    if json.loads(json.dumps(actual)) != expected:
                        raise ValueError(f'complete controller decision differs: frame={frame}, arm={arm}')
                rows.append(dict(frame=frame, both_complete_decisions_equal=True,
                    baseline_s=durations[0], candidate_s=durations[1], timed=frame>=3))
                print('SAMPLED_PLANE_FRAME', frame, durations, flush=True)
        assert source.trial.previous.state_digest(model.state_dict()) == before
        assert all(p.grad is None for p in model.parameters())
        timed = rows[3:]
        totals = [sum(r[k] for r in timed) for k in ('baseline_s', 'candidate_s')]
        report = dict(status='EARLY_CONTROLLER_COMPARISON_COMPLETE', rows=rows,
            baseline_total_s=totals[0], candidate_total_s=totals[1],
            total_reduction_percent=100*(1-totals[1]/totals[0]),
            baseline_median_ms=float(np.median([r['baseline_s'] for r in timed])*1000),
            candidate_median_ms=float(np.median([r['candidate_s'] for r in timed])*1000),
            hardware=hardware, model_state_sha256=before, model_unchanged=True,
            input=str(source.INPUT), input_launch_sha256=source.digest(source.ROOT/'launch.json'),
            sources={p:source.digest(Path(p)) for p in (
                'lewm/sampled_plane_candidates_development.py',
                'lewm/sampled_plane_stop_conditioned_controller_development.py',
                'scripts/compare_sampled_plane_early_controller_development.py')},
            profiler_used=False, early_history_only=True, native_execution=False,
            full_history_equivalence_proven=False, real_time_qualified=False, adopted=False)
        with OUTPUT.open('x') as out:
            json.dump(report, out, indent=2, allow_nan=False); out.write('\n')
        print(json.dumps({k: report[k] for k in ('status', 'total_reduction_percent',
            'baseline_median_ms', 'candidate_median_ms')}), flush=True)
    except BaseException as error:
        with FAILURE.open('x') as out:
            json.dump(dict(reason=repr(error), completed_rows=rows), out, indent=2)
        raise


if __name__ == '__main__':
    main()
