"""Fixed recorded-pose map counterfactual through first view-budget exhaustion.

Reconstruct delivered noisy pixels and replay the recorded mapping-frame prefix.
This consumes compact previously admitted poses, not full registration witnesses,
and does not execute a controller or establish counterfactual navigation success.
"""
import argparse
import hashlib
import json
import time
import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.multirate_routing_map_development import Geometry, MultirateRoutingMap
from lewm.local_inverse_depth_floor_development import local_depth
from lewm.fine_stored_obstacle_routing_development import proposer
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


def recorded_pose(pose, *, identity, now_ns):
    if identity != (0, 0, 0) or pose['measured_ns'] != now_ns:
        raise ValueError('matching recorded public pose required')
    return (np.asarray(pose['position_initial_body_m']),
        np.asarray(pose['rotation_initial_body_from_current_body']), pose)


class LocalCoverageGeometry(Geometry):
    def floor_coverage(self, depth, valid, *args, **kwargs):
        estimated, supported = local_depth(depth, valid)
        return super().floor_coverage(estimated, supported, *args, **kwargs)


class RecordedMap(MultirateRoutingMap):
    retain_fine_obstacles = True
    update = bind(MultirateRoutingMap.update, current_measured_floor_pose=recorded_pose)


class LocalCoverageMap(RecordedMap):
    update = bind(RecordedMap.update, Geometry=LocalCoverageGeometry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    root = BASE/f'go2_live_local_floor_obstacle_noise_2mm_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    output = root/'routing_floor_prefix_counterfactual_v1.json'
    if output.exists():
        raise ValueError('preserve completed counterfactual')
    plans = json.loads((root/'planning.json').read_text())
    first_exhausted = next(p['frame'] for p in plans if p.get('reason') == 'VIEW_BUDGET_EXHAUSTED')
    selected = [p for p in plans if 'selection' in p and p['frame'] < first_exhausted]
    witnesses = {p['map_frame']:p['selection']['routing_memory_scope'] for p in selected}
    last_frame = max(witnesses)
    events = json.loads((root/'stage_events.json').read_text())
    frames = sorted({e['frame'] for e in events if e['stage'] == 'mapping' and e['frame'] <= last_frame})
    if not frames or frames[0] != 0:
        raise ValueError('complete recorded mapping prefix required')
    poses = {p['frame']:p['registered_pose'] for p in json.loads((root/'poses.json').read_text())}
    launch = json.loads((root/'launch.json').read_text())
    reader = NoisyPublicReplay(root/'native')
    maps = dict(raw=RecordedMap(), local_floor_coverage=LocalCoverageMap())
    rows = []; started = time.monotonic()
    for frame in frames:
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        snapshots = {name:mapper.update(policy, depth, poses[frame],
            auxiliary_depth=auxiliary, measured_ns=now) for name, mapper in maps.items()}
        a, b = snapshots.values()
        if a.floor_height != b.floor_height or a.occupied != b.occupied or a.fine_occupied != b.fine_occupied:
            raise ValueError('floor-height hypothesis and original obstacle sets must remain identical')
        reproduced = None
        if frame in witnesses:
            w = witnesses[frame]
            reproduced = (len(a.floor) == w['retained_floor_cells']
                and len(a.fine_occupied) == w['retained_fine_obstacle_cells'])
            if not reproduced:
                raise ValueError(f'recorded map counts did not reproduce at {frame}')
        row = dict(frame=frame, saved_map_counts_exactly_reproduced=reproduced, variants={})
        for name, snapshot in snapshots.items():
            data = dict(retained_floor_cells=len(snapshot.floor),
                primary_current_floor_cells=snapshot.primary_current_floor_cells,
                auxiliary_current_floor_cells=snapshot.auxiliary_current_floor_cells,
                fine_obstacle_cells=len(snapshot.fine_occupied))
            if frame in (frames[0], frames[-1]):
                goal = np.asarray(snapshot.map_from_initial)@np.r_[launch['public_mission']['goal_initial_body_xy_m'], 0.]
                route = proposer(snapshot)(snapshot.floor, snapshot.occupied,
                    snapshot.position_map[:2], goal[:2])
                data['base_fine_obstacle_route'] = route
                data['floor_cells'] = sorted(snapshot.floor)
            row['variants'][name] = data
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(completed=len(rows), total=len(frames), frame=frame,
                raw_floor=len(a.floor), local_floor=len(b.floor))), flush=True)
    report = dict(layout_index=args.layout_index, first_view_budget_exhausted_frame=first_exhausted,
        frames=rows, wall_seconds=time.monotonic()-started,
        actual_delivered_noisy_packet_digests_verified=True,
        fixed_recorded_public_poses=True, full_pose_acceptance_replayed=False,
        original_initial_floor_height_and_obstacle_sets_identical=True,
        floor_coverage_thresholds_changed=False, invalid_depth_filled=False,
        derived_floor_depth_is_not_raw_pixel_depth=True,
        route_probe_uses_base_fine_obstacle_proposer=True,
        full_controller_route_state_replayed=False, counterfactual_navigation_executed=False,
        native_physics_used=False,
        source_sha256={p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in
            (__file__, 'lewm/multirate_routing_map_development.py',
             'lewm/local_inverse_depth_floor_development.py',
             'lewm/body_projected_floor_geometry_development.py')})
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(output=str(output), reproduced_witnesses=sum(r['saved_map_counts_exactly_reproduced'] is True for r in rows),
        last={k:{x:y for x,y in v.items() if x != 'floor_cells'} for k,v in rows[-1]['variants'].items()})), flush=True)


if __name__ == '__main__':
    main()
