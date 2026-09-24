"""Profile ten early decisions after actual warmup, without a full replay.

Only closed sensor files from the collected stopping-rule trial are consumed.
This diagnostic neither replaces its ongoing audit nor changes the controller.
"""
import cProfile
from contextlib import closing
import hashlib
import json
from pathlib import Path
import pstats
import time

import cv2
import psutil
import torch

from lewm.stop_conditioned_settling_development import StopConditionedSettlingController
from scripts import extended_return_budget_maze_pipeline_development as packets
from scripts.all_phase_planner_model_admission_development import load_assigned
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition
from scripts import run_go2_stop_conditioned_settling_maze02_v1 as trial

ROOT = trial.OUTPUT
INPUT = ROOT/trial.CASE
OUTPUT = Path('docs/go2_stop_conditioned_early_controller_profile_2026-09-13.json')
PROFILE = Path('docs/go2_stop_conditioned_early_controller_profile_2026-09-13.prof')
FRAMES = 13


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    if OUTPUT.exists() or PROFILE.exists():
        raise ValueError('preserve existing diagnostic output')
    collection = json.loads((INPUT/'result.json').read_text())
    if collection['decisions'] < FRAMES:
        raise ValueError('thirteen completed observations required')
    names = ['policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
        'fast_gyro_histories.npz', 'auxiliary_camera_audit.json', 'public_mission.json']
    names += [f'{stem}_{frame:04d}.{suffix}' for frame in range(FRAMES)
        for stem, suffix in [('rgb', 'png'), ('depth', 'npz'),
            ('auxiliary_rgb', 'png'), ('auxiliary_depth', 'npz')]]
    input_sha = {name: digest(INPUT/name) for name in names}
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        cpu_percent=psutil.cpu_percent(interval=.2),
        active_trial_pid=3130098, one_additional_short_cpu_replay=True)
    if hardware['available_ram_bytes'] < 8*1024**3:
        raise ValueError('insufficient memory for diagnostic')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    admission_root = trial.run.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
    admission = json.loads((admission_root/'launch.json').read_text())['input_admission']['correction_admission']
    model, condition, variant = load_assigned(admission, 'seed_2026091001_no_rgb_direct')
    before = trial.previous.state_digest(model.state_dict())
    launch = json.loads((ROOT/'launch.json').read_text())
    if before != launch['model_state_sha256']:
        raise ValueError('same actual collected model required')
    controller = StopConditionedSettlingController(model,
        trial.previous.geometry_factory(trial.previous.URDF),
        public_mission=json.loads((INPUT/'public_mission.json').read_text()),
        navigation_ticks=8000, condition=condition, variant=variant, persistent=True)
    reader = packets.ExtendedReturnBudgetRGBDReplay(INPUT)
    auxiliary_rows = json.loads((INPUT/'auxiliary_camera_audit.json').read_text())
    profile = cProfile.Profile(); rows = []
    with closing(packets.read_rows(INPUT)) as recorded:
        for frame in range(FRAMES):
            expected = next(recorded)
            p, d, f, now = reader.packet(frame)
            image, auxiliary = packets.rgb_packet(INPUT, frame, p,
                public_acquisition(auxiliary_rows[frame]), now_ns=now)
            started = time.perf_counter()
            if frame >= 3:
                profile.enable()
            try:
                actual = controller.observe(p, d, f, now_ns=now,
                    auxiliary_depth=auxiliary, auxiliary_rgb=image)
            finally:
                profile.disable()
            duration = time.perf_counter()-started
            if json.loads(json.dumps(actual)) != expected['decision']:
                raise ValueError(f'complete decision differs at frame {frame}; stop before next observation')
            rows.append(dict(frame=frame, complete_decision_equal=True,
                controller_wall_s=duration, profiled=frame >= 3,
                recorded_decision_sha256=hashlib.sha256(
                    json.dumps(expected['decision'], sort_keys=True).encode()).hexdigest()))
    if before != trial.previous.state_digest(model.state_dict()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('model changed during diagnostic')
    if input_sha != {name: digest(INPUT/name) for name in names}:
        raise ValueError('consumed sensor files changed')
    profile.dump_stats(PROFILE)
    stats = pstats.Stats(profile)
    functions = [dict(filename=k[0], line=k[1], function=k[2], primitive_calls=v[0],
        calls=v[1], self_s=v[2], cumulative_s=v[3]) for k, v in stats.stats.items()]
    report = dict(profiled_frames=list(range(3, FRAMES)), rows=rows, hardware=hardware,
        model_state_sha256=before, input_sha256=input_sha, current_launch_sha256=digest(ROOT/'launch.json'),
        profile_sha256=digest(PROFILE), total_exclusive_profile_s=stats.total_tt,
        top_self=sorted(functions, key=lambda row: row['self_s'], reverse=True)[:25],
        top_cumulative=sorted(functions, key=lambda row: row['cumulative_s'], reverse=True)[:25],
        profiler_overhead_included=True, cumulative_times_overlap=True, shared_host=True,
        early_history_only=True, complete_raw_audit_pending=True,
        model_unchanged=True, whole_loop_latency_measured=False, native_execution=False)
    with OUTPUT.open('x') as destination:
        json.dump(report, destination, indent=2, allow_nan=False); destination.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), total_profile_s=stats.total_tt,
        decisions_equal=len(rows), top_self=report['top_self'][:8],
        top_cumulative=report['top_cumulative'][:8])), flush=True)


if __name__ == '__main__':
    main()
