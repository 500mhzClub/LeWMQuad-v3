"""Profile one retained public startup decision, without running physics."""
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import cv2
import torch

from scripts.run_go2_obstacle_grouping_navigation_development import study, ROOT
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.diagnose_go2_short_pulse_direct_contact_development import RecordedMap
from scripts.diagnose_alignment_route_switches_development import saved_pose
from lewm.paced_multirate_controller_development import AcquiredFrame


class OfflineRuntime(study.PulsePredictiveRuntime):
    def _worker(self, name, function):
        pass

    _pose = staticmethod(saved_pose)

    def _store_plan(self, plan, completed, prefix):
        pass


def main():
    root = study.BASE/ROOT.format(condition='packed')
    output = root/'startup_planning_profile_v1'
    output.mkdir(exist_ok=False)
    (output/'profile.py').write_text(Path(__file__).read_text())
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    study.cohort.stable.floor.configure()
    read = lambda name: json.loads((root/name).read_text())
    plan = next(p for p in read('planning.json') if 'selection' in p)
    frame = plan['frame']
    poses = {p['frame']: p['registered_pose'] for p in read('poses.json')}
    mission = next(m for m in read('mission.json') if m['frame'] == frame)
    reader = NoisyPublicReplay(root/'native')
    mapper = RecordedMap()
    for f in range(0, plan['map_frame']+1, 4):
        policy, depth, _, _, auxiliary, now = reader.packet(f)
        snapshot = mapper.update(policy, depth, poses[f],
            auxiliary_depth=auxiliary, measured_ns=now)
    history = tuple(reader.packet(f)[0] for f in range(frame-3, frame+1))
    policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
    packet = AcquiredFrame(frame, now, policy, depth, fast, rgb, auxiliary, history)
    model, condition, variant = study.load_model('supervised_rollout')
    runtime = OfflineRuntime(model, prediction_source='neural', condition=condition,
        variant=variant, registration_executor=None, mapping_executor=None,
        pose_executor=None, obstacle_executor=None,
        goal_initial_xy=mission['active_goal_initial_body_xy_m'],
        clock_ns=lambda: now+250_000_000, planning_delay_ticks=3,
        navigation_ticks=4800, arrival_radius_m=.02)
    runtime.latest_map = snapshot
    runtime.mission_latest = mission
    runtime.correction_poses = poses
    # Warm-up is separate. Repeated calls exercise the same startup observation;
    # they do not reconstruct later route/mission state or alternative motion.
    runtime._plan((packet, poses[frame]))
    profile = cProfile.Profile()
    elapsed = []
    profile.enable()
    for _ in range(20):
        start = time.perf_counter_ns()
        runtime._plan((packet, poses[frame]))
        elapsed.append((time.perf_counter_ns()-start)/1e6)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(45)
    (output/'profile.txt').write_text(stream.getvalue())
    profile.dump_stats(str(output/'profile.pstats'))
    result = dict(status='COMPLETE', frame=frame, map_frame=plan['map_frame'],
        profiled_calls=20, elapsed_ms=elapsed,
        native_concurrency_reproduced=False, startup_decision_only=True,
        native_state_read=False, navigation_outcome_tested=False)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    print(stream.getvalue(), flush=True)


if __name__ == '__main__':
    main()
