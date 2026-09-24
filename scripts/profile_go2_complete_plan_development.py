"""Reconstruct one registered decision and profile the complete planning call."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import statistics
import time

import cv2
import numpy as np
import torch

from scripts.profile_go2_current_planning_development import ROOT, study, RecordedMap, NoisyPublicReplay
from lewm import process_registered_round_trip_development as registration_process
from lewm import process_mapped_runtime_development as pose_process
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime
from lewm.paced_multirate_controller_development import AcquiredFrame


class OfflineRuntime(PersistentLocalVisualRuntime):
    def _worker(self, name, function):
        pass

    def _prefix_commands(self, observed_ns):
        return self.recorded_prefix

    def _store_plan(self, plan, completed, prefix):
        pass


def main():
    root = study.study.BASE/ROOT
    if (root/'DEPTH_RETIRED').exists():
        raise ValueError('retained full sensor recording required')
    output = study.study.BASE/'go2_complete_plan_profile_v1_attempt_001'
    output.mkdir(exist_ok=False)
    (output/'profile.py').write_text(Path(__file__).read_text())
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    study.study.cohort.stable.floor.configure()
    study.initialize_pose(str(output))
    study.study.previous.reference.previous.initialize_registration()
    read = lambda name: json.loads((root/name).read_text())
    frame = 300
    plan = next(p for p in read('planning.json') if p['frame'] == frame)
    poses = {p['frame']: p['registered_pose'] for p in read('poses.json')}
    reader = NoisyPublicReplay(root/'native'); mapper = RecordedMap()
    began = time.monotonic()
    for f in range(frame+1):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(f)
        raw = pose_process._motion.observe(policy, depth, fast, now_ns=now,
            auxiliary_rgb=rgb, auxiliary_depth=auxiliary)
        evidence = registration_process._registration.observe(policy, depth, auxiliary, raw, now_ns=now)
        np.testing.assert_array_equal(evidence['current_pose']['position_initial_body_m'], poses[f]['position_initial_body_m'])
        np.testing.assert_array_equal(evidence['current_pose']['rotation_initial_body_from_current_body'], poses[f]['rotation_initial_body_from_current_body'])
        if f % 4 == 0 and f <= plan['map_frame']:
            snapshot = mapper.update(policy, depth, poses[f], auxiliary_depth=auxiliary, measured_ns=now)
        if f % 100 == 0:
            print('FULL_PROFILE_REPLAY', f, round(time.monotonic()-began, 2), flush=True)
    history = tuple(reader.packet(f)[0] for f in range(frame-3, frame+1))
    packet = AcquiredFrame(frame, now, policy, depth, fast, rgb, auxiliary, history)
    model, condition, variant = study.study.load_model('jepa')
    mission = next(m for m in read('mission.json') if m['frame'] == frame)
    runtime = OfflineRuntime(model, prediction_source='neural', condition=condition, variant=variant,
        registration_executor=None, mapping_executor=None, pose_executor=None, obstacle_executor=None,
        goal_initial_xy=mission['active_goal_initial_body_xy_m'], clock_ns=lambda: now+250_000_000,
        planning_delay_ticks=3, navigation_ticks=4800, arrival_radius_m=.02)
    runtime.latest_map = snapshot; runtime.mission_latest = mission
    runtime.recorded_prefix = plan['committed_prefix']; runtime.correction_poses = poses
    runtime.initial_panorama.state = read('initial_survey.json')
    assert runtime.initial_panorama.complete and runtime.initial_panorama.state['completed_ns'] <= now
    rows = []
    for label, call in [('pose_read', lambda: runtime._pose(evidence, identity=(0, 0, 0), now_ns=now)),
                        ('complete_plan', lambda: runtime._plan((packet, evidence)))]:
        call(); elapsed = []
        for _ in range(10):
            start = time.perf_counter_ns(); call()
            elapsed.append((time.perf_counter_ns()-start)/1e6)
        profile = cProfile.Profile(); profile.runcall(call)
        stream = io.StringIO(); pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(30)
        (output/(label+'.txt')).write_text(stream.getvalue())
        profile.dump_stats(str(output/(label+'.pstats')))
        row = dict(component=label, elapsed_ms=elapsed, median_ms=statistics.median(elapsed))
        rows.append(row); print(json.dumps(row), flush=True); print(stream.getvalue(), flush=True)
    result = dict(rows=rows, frame=frame, exact_registered_pose_prefix_frames=frame+1,
        source_root=ROOT, native_state_used=False, native_concurrency_reproduced=False,
        frontier_visit_history_reconstructed=False, navigation_outcome_tested=False)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
