"""Profile current action selection on retained measured observations and maps."""
import cProfile
import argparse
import io
import json
from pathlib import Path
import pstats
import statistics
import time

import cv2
import numpy as np
import torch

from scripts import run_go2_persistent_visual_learning_comparison_development as study
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.diagnose_go2_short_pulse_direct_contact_development import RecordedMap
from scripts.diagnose_alignment_route_switches_development import saved_pose
from lewm.paced_multirate_controller_development import AcquiredFrame
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime

ROOT = 'go2_persistent_visual_learning_comparison_jepa_noise_2mm_native_layout03_4800_v1_attempt_001'
OUTPUT = 'go2_current_planning_profile_v1_attempt_001'


class OfflineRuntime(PersistentLocalVisualRuntime):
    def _worker(self, name, function):
        pass

    _pose = staticmethod(saved_pose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routing', action='store_true')
    args = parser.parse_args()
    root = study.study.BASE/ROOT
    if (root/'DEPTH_RETIRED').exists():
        raise ValueError('retained public depth required')
    output = study.study.BASE/(OUTPUT if not args.routing else 'go2_current_routing_profile_v1_attempt_001')
    output.mkdir(exist_ok=False)
    (output/'profile.py').write_text(Path(__file__).read_text())
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    study.study.cohort.stable.floor.configure()
    read = lambda name: json.loads((root/name).read_text())
    plans = {p['frame']: p for p in read('planning.json') if 'selection' in p}
    poses = {p['frame']: p['registered_pose'] for p in read('poses.json')}
    reader = NoisyPublicReplay(root/'native')
    mapper = RecordedMap()
    wanted = (300, 700, 1100)
    snapshots = {}
    for f in range(0, max(plans[n]['map_frame'] for n in wanted)+1, 4):
        policy, depth, _, _, auxiliary, now = reader.packet(f)
        snapshot = mapper.update(policy, depth, poses[f], auxiliary_depth=auxiliary, measured_ns=now)
        if f in {plans[n]['map_frame'] for n in wanted}:
            snapshots[f] = snapshot
        if f % 400 == 0:
            print('PROFILE_MAP', f, flush=True)
    model, condition, variant = study.study.load_model('jepa')
    rows = []
    for frame in wanted:
        plan = plans[frame]; snapshot = snapshots[plan['map_frame']]
        history = tuple(reader.packet(f)[0] for f in range(frame-3, frame+1))
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        packet = AcquiredFrame(frame, now, policy, depth, fast, rgb, auxiliary, history)
        runtime = OfflineRuntime(model, prediction_source='neural', condition=condition,
            variant=variant, registration_executor=None, mapping_executor=None,
            pose_executor=None, obstacle_executor=None, goal_initial_xy=[0., 1.3],
            clock_ns=lambda: now+250_000_000, planning_delay_ticks=3,
            navigation_ticks=4800, arrival_radius_m=.02)
        runtime.correction_poses = poses
        runtime.exact_terminal_target = False
        runtime.terminal_target_distance = None
        runtime.terminal_position_approach = plan['motion_correction']['terminal_translation_pulse']
        selection = plan['selection']
        B = np.asarray(snapshot.map_from_initial)
        p, R, _ = saved_pose(poses[frame]); q, Q = B@p, B@R
        prefix = plan['committed_prefix']
        goal = np.asarray(selection['waypoint_body_xy_m'])
        scan = selection.get('scan_heading_error_rad')

        def call():
            runtime.clearance_turn = None
            with torch.inference_mode():
                return runtime._select_action(packet, poses[frame], prefix, goal, scan, snapshot, q, Q)

        if args.routing:
            proposer = runtime._routing_proposer(snapshot)
            def call():
                return proposer(snapshot.floor, snapshot.occupied, q[:2], (B@np.array([0., 1.3, 0.]))[:2])
            call()
        else:
            _, correction = call()
            np.testing.assert_allclose(correction['raw_forecast_xy_m'],
                plan['motion_correction']['raw_forecast_xy_m'], atol=1e-7, rtol=1e-6)
        elapsed = []
        for _ in range(10):
            begin = time.perf_counter_ns(); call()
            elapsed.append((time.perf_counter_ns()-begin)/1e6)
        profile = cProfile.Profile()
        profile.runcall(call)
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(35)
        (output/f'frame{frame}_profile.txt').write_text(stream.getvalue())
        profile.dump_stats(str(output/f'frame{frame}_profile.pstats'))
        row = dict(frame=frame, map_frame=snapshot.frame, elapsed_ms=elapsed,
            median_ms=statistics.median(elapsed), original_raw_neural_forecast_reproduced=not args.routing,
            routing_state_replayed=False, clearance_turn_reset_for_each_call=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
        print(stream.getvalue(), flush=True)
    result = dict(rows=rows, source_root=ROOT, native_state_used=False,
        scope=('routing proposer only; excludes frontier visit state and upstream/concurrent workers' if args.routing else
            'action selection only; excludes routing, upstream sensing and concurrent workers'),
        navigation_outcome_tested=False, source_model='frozen JEPA')
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
