"""Recorded-map diagnostic of the first post-panorama no-frontier decision.

Fixed admitted public poses are inputs, not new pose-acceptance evidence.
The base route proposer is compared with/without saved exclusions; this does
not replay the full controller or execute counterfactual navigation.
"""
import hashlib
import json
import time
import numpy as np
from scripts.probe_go2_noisy_routing_floor_development import LocalCoverageMap
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE
from lewm.fine_stored_obstacle_routing_development import proposer


def main():
    root = BASE/'go2_live_local_floor_mapping_noise_2mm_native_layout02_4800_v1_attempt_001'
    output = root/'completed_frontier_exclusion_probe_v1.json'
    if output.exists():
        raise ValueError('preserve prior diagnostic')
    plans = json.loads((root/'planning.json').read_text())
    plan = next(p for p in plans if p.get('route_status') == 'OBSERVED_COMPONENT_HAS_NO_FRONTIER')
    end = plan['map_frame']
    visits = json.loads((root/'frontier_visits.json').read_text())
    assert len(visits['events']) == 1 and visits['events'][0]['map_frame'] == end
    excluded = {tuple(c) for c in visits['excluded_cells']}
    events = json.loads((root/'stage_events.json').read_text())
    frames = sorted({e['frame'] for e in events if e['stage'] == 'mapping' and e['frame'] <= end})
    assert frames[0] == 0 and frames[-1] == end
    witnesses = {p['map_frame']:p['selection']['routing_memory_scope']
        for p in plans if 'selection' in p and p['map_frame'] <= end}
    poses = {p['frame']:p['registered_pose'] for p in json.loads((root/'poses.json').read_text())}
    launch = json.loads((root/'launch.json').read_text())
    reader = NoisyPublicReplay(root/'native'); mapper = LocalCoverageMap()
    checked = []; started = time.monotonic()
    for frame in frames:
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        snapshot = mapper.update(policy, depth, poses[frame], auxiliary_depth=auxiliary, measured_ns=now)
        if frame in witnesses:
            w = witnesses[frame]
            assert len(snapshot.floor) == w['retained_floor_cells'], frame
            assert len(snapshot.fine_occupied) == w['retained_fine_obstacle_cells'], frame
            checked.append(frame)
    goal = np.asarray(snapshot.map_from_initial)@np.r_[launch['public_mission']['goal_initial_body_xy_m'], 0.]
    route = proposer(snapshot)
    variants = {name:route(snapshot.floor, snapshot.occupied, snapshot.position_map[:2], goal[:2],
        excluded_frontiers=cells) for name,cells in [('saved_exclusions',excluded),('no_exclusions',set())]}
    report = dict(layout_index=2,planning_frame=plan['frame'],map_frame=end,
        mapping_frames=len(frames),exact_saved_count_witness_frames=checked,
        excluded_cells=sorted(excluded),variants=variants,position_map=snapshot.position_map,
        original_live_route_status=plan['route_status'],
        fixed_recorded_public_poses=True,full_pose_acceptance_replayed=False,
        actual_delivered_noisy_packet_digests_verified=True,full_controller_state_replayed=False,
        counterfactual_navigation_executed=False,native_physics_used=False,
        wall_seconds=time.monotonic()-started,
        source_sha256={__file__:hashlib.sha256(open(__file__,'rb').read()).hexdigest()})
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
