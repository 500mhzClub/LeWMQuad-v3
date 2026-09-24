"""Replay public map evidence at the first blocked initial-survey decision."""
import json
import time
import cv2
import numpy as np
from scripts.diagnose_go2_short_pulse_direct_contact_development import RecordedMap
from scripts.diagnose_alignment_route_switches_development import saved_pose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.navigation_artifact_root_development import BASE
from lewm.two_cm_floor_extent_development import configure
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.memory_forecast_clearance_development import select_clear_prediction
from lewm.geometry_progress_pilot_development import ACTIONS

ROOT = BASE/'go2_short_pulse_navigation_instantaneous_noise_2mm_native_layout01_4800_v1_attempt_001'
OUTPUT = BASE/'go2_short_pulse_initial_survey_stall_map_replay_v1_attempt_001'


def main():
    if OUTPUT.exists():
        raise ValueError('preserve prior diagnostic')
    OUTPUT.mkdir(); started = time.monotonic()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); configure()
    read = lambda name: json.loads((ROOT/name).read_text())
    plans = [p for p in read('planning.json') if 'selection' in p]
    plan = next(p for p in plans if p['selection']['memory_forecast_status'] ==
        'NO_CLEAR_CANDIDATE_ZERO_REQUESTED')
    poses = {p['frame']:p['registered_pose'] for p in read('poses.json')}
    frames = sorted(e['frame'] for e in read('stage_events.json')
        if e['stage'] == 'mapping' and e['frame'] <= plan['map_frame'])
    assert frames == list(range(0, plan['map_frame']+1, 4))
    reader = NoisyPublicReplay(ROOT/'native'); mapper = RecordedMap(); first_seen = {}
    for frame in frames:
        policy, depth, _, _, auxiliary, now = reader.packet(frame)
        snapshot = mapper.update(policy, depth, poses[frame],
            auxiliary_depth=auxiliary, measured_ns=now)
        for cell in snapshot.fine_occupied:
            first_seen.setdefault(cell, frame)
        if frame % 400 == 0:
            print('SURVEY_MAP_REPLAY', frame, flush=True)
    B = np.asarray(snapshot.map_from_initial)
    p, R, _ = saved_pose(poses[plan['frame']]); q = B@p; Q = B@R
    prediction = np.asarray(plan['motion_correction']['applied_prediction_after_yaw_ablation'])
    selected = dict(action='hold', candidates=[dict(action=a, utility_m=0.) for a in ACTIONS])
    replayed = select_clear_prediction(selected, prediction, snapshot.fine_occupied,
        q, Q, translation_reserve_m=.03, reserve_recovery=True)
    recorded = plan['selection']['memory_forecast_candidates']
    for actual, expected in zip(replayed['memory_forecast_candidates'], recorded, strict=True):
        assert actual['action'] == expected['action']
        np.testing.assert_allclose(actual['segment_clearances_m'],
            expected['segment_clearances_m'], rtol=0, atol=1e-12)
    assert len(snapshot.fine_occupied) == plan['selection']['routing_memory_scope']['retained_fine_obstacle_cells']
    cells = np.asarray(sorted(snapshot.fine_occupied), dtype=int)
    gaps = np.maximum(np.maximum(cells*.01-q[:2], q[:2]-(cells+1)*.01), 0)
    nearest = int(np.linalg.norm(gaps, axis=1).argmin())
    with (OUTPUT/'stored_obstacles.npz').open('xb') as f:
        np.savez_compressed(f, cells=cells, map_from_initial=B, position_map=q,
            floor_height=snapshot.floor_height, rotation_map_from_body=Q,
            first_seen_frames=np.array([first_seen[tuple(c)] for c in cells]))
    result = dict(status='COMPLETE', root_name=ROOT.name, mapped_frames=len(frames),
        plan_frame=plan['frame'], map_frame=plan['map_frame'], fine_cells=len(cells),
        body_centre_map_clearance_m=cached_clearance(snapshot.fine_occupied).minimum(q[:2],q[:2]),
        nearest_cell=cells[nearest].tolist(), nearest_cell_first_seen_frame=first_seen[tuple(cells[nearest])],
        all_six_eight_segment_clearance_sequences_reproduced=True,
        public_packets_and_recorded_estimator_only=True, native_geometry_or_pose_read=False,
        controller_unchanged=True, wall_s=time.monotonic()-started)
    with (OUTPUT/'result.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
