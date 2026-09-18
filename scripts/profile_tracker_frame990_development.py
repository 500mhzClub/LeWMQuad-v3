"""Reconstruct raw tracker history, then profile only the recurrent slow frame."""
import cProfile
import json
from pathlib import Path
import pstats
import time

import cv2
import psutil
import torch

from lewm.full_consensus_tracker_development import FullConsensusVisualMotion
from scripts import profile_stop_conditioned_early_decisions_development as source

OUTPUT = Path('docs/go2_tracker_frame990_full_profile_2026-09-13.json')


def main():
    assert not OUTPUT.exists()
    if psutil.virtual_memory().available < 16*1024**3: raise ValueError('retain memory reserve')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    owner = psutil.Process(); print('FRAME990_PROFILE_STARTED', owner.pid, owner.create_time(), flush=True)
    tracker = FullConsensusVisualMotion()
    reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    profiler = cProfile.Profile(); started_all = time.perf_counter()
    for frame in range(991):
        p, d, f, now = reader.packet(frame)
        rgb, aux = source.packets.rgb_packet(source.INPUT, frame, p,
            source.public_acquisition(auxiliary[frame]), now_ns=now)
        if frame == 990: profiler.enable()
        started = time.perf_counter()
        result = tracker.observe(p, d, f, now_ns=now, auxiliary_rgb=rgb, auxiliary_depth=aux)
        duration = time.perf_counter()-started
        if frame == 990: profiler.disable()
        assert result.get('failure') is None and result.get('current_pose') is not None
        if frame % 250 == 0: print('FRAME990_PROFILE_HISTORY', frame, flush=True)
    stats = pstats.Stats(profiler); functions = []
    for (file, line, name), (_, calls, self_s, total_s, _) in stats.stats.items():
        functions.append(dict(file=file, line=line, name=name, calls=calls,
            self_s=self_s, cumulative_s=total_s))
    functions.sort(key=lambda r:r['cumulative_s'], reverse=True)
    report = dict(frame=990, history_observations=991, input=str(source.INPUT),
        tracker_s=duration, wall_s=time.perf_counter()-started_all,
        camera=tracker.model.last_camera_selection, reference_selection=tracker.model.last_selection,
        functions=functions, profiler='cProfile', shared_host=True,
        source_sha256=source.digest(Path(__file__)), native_pose_loaded=False,
        model_loaded=False, map_or_planner_executed=False, native_execution=False)
    with OUTPUT.open('x') as out: json.dump(report, out, indent=2); out.write('\n')
    print('FRAME990_PROFILE_COMPLETE', duration, flush=True)
    for row in functions[:25]: print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
