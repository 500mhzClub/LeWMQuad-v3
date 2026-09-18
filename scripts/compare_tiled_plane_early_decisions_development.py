"""Uninstrumented early-prefix comparison of existing and tiled plane tracking."""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.tiled_plane_stop_conditioned_controller_development import TiledPlaneStopConditionedController
from scripts import profile_stop_conditioned_early_decisions_development as setup

OUTPUT = Path('docs/go2_tiled_plane_early_comparison_2026-09-13.json')


def main():
    if OUTPUT.exists():
        raise ValueError('preserve existing comparison')
    # Reuse the explicit input roster from the completed short diagnostic.
    admitted = json.loads(setup.OUTPUT.read_text())
    input_sha = admitted['input_sha256']
    if input_sha != {name: setup.digest(setup.INPUT/name) for name in input_sha}:
        raise ValueError('recorded diagnostic sensor inputs changed')
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        cpu_percent=psutil.cpu_percent(interval=.2), shared_host=True)
    if hardware['available_ram_bytes'] < 8*1024**3:
        raise ValueError('insufficient memory for short comparison')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    launch_root = setup.trial.run.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
    admission = json.loads((launch_root/'launch.json').read_text())['input_admission']['correction_admission']
    model, condition, variant = setup.load_assigned(admission, 'seed_2026091001_no_rgb_direct')
    before = setup.trial.previous.state_digest(model.state_dict())
    if before != admitted['model_state_sha256']:
        raise ValueError('same recorded model required')
    mission = json.loads((setup.INPUT/'public_mission.json').read_text())
    arms = [cls(model, setup.trial.previous.geometry_factory(setup.trial.previous.URDF),
        public_mission=mission, navigation_ticks=8000, condition=condition,
        variant=variant, persistent=True)
        for cls in (setup.StopConditionedSettlingController, TiledPlaneStopConditionedController)]
    reader = setup.packets.ExtendedReturnBudgetRGBDReplay(setup.INPUT)
    auxiliary = json.loads((setup.INPUT/'auxiliary_camera_audit.json').read_text())
    rows = []; started_all = time.perf_counter()
    with closing(setup.packets.read_rows(setup.INPUT)) as recorded:
        for frame in range(setup.FRAMES):
            expected = next(recorded)['decision']
            p, d, f, now = reader.packet(frame)
            image, depth = setup.packets.rgb_packet(setup.INPUT, frame, p,
                setup.public_acquisition(auxiliary[frame]), now_ns=now)
            durations = [None, None]; decisions = [None, None]
            for arm in ((0, 1) if frame % 2 == 0 else (1, 0)):
                started = time.perf_counter()
                result = arms[arm].observe(p, d, f, now_ns=now,
                    auxiliary_depth=depth, auxiliary_rgb=image)
                durations[arm] = time.perf_counter()-started
                if arm == 1:
                    if result.pop('tiled_plane_tracking_kernel_enabled') is not True:
                        raise ValueError('explicit candidate kernel identity required')
                decisions[arm] = json.loads(json.dumps(result))
            if any(decision != expected for decision in decisions):
                with OUTPUT.open('x') as output:
                    json.dump(dict(status='EARLY_COMPARISON_DECISION_DIFFERENCE',
                        first_difference_frame=frame, previous_rows=rows,
                        baseline_matches=decisions[0] == expected, candidate_matches=decisions[1] == expected,
                        stop_before_next_observation=True, adopted=False), output, indent=2)
                raise ValueError(f'complete decision differs at frame {frame}')
            rows.append(dict(frame=frame, baseline_s=durations[0], candidate_s=durations[1],
                complete_recorded_decisions_equal=True))
    if before != setup.trial.previous.state_digest(model.state_dict()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('model changed')
    if input_sha != {name: setup.digest(setup.INPUT/name) for name in input_sha}:
        raise ValueError('recorded sensor files changed during comparison')
    active = rows[3:]
    totals = {arm: sum(row[arm+'_s'] for row in active) for arm in ('baseline', 'candidate')}
    report = dict(status='EARLY_COMPARISON_COMPLETE', rows=rows, hardware=hardware,
        active_decisions=len(active), total_seconds=totals,
        median_seconds={arm: float(np.median([row[arm+'_s'] for row in active])) for arm in totals},
        total_time_reduction_percent=100*(1-totals['candidate']/totals['baseline']),
        model_state_sha256=before, recorded_input_sha256=input_sha,
        comparison_wall_s=time.perf_counter()-started_all,
        all_recorded_decisions_equal=True, profiler_used=False, alternating_arm_order=True,
        late_history_tested=False, full_state_equivalence_proven=False,
        whole_loop_latency_measured=False, native_execution=False, adopted=False)
    with OUTPUT.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False); output.write('\n')
    print(json.dumps({key: value for key, value in report.items()
        if key not in ('rows', 'recorded_input_sha256')}), flush=True)


if __name__ == '__main__':
    main()
