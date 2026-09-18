"""Reconstruct recorded mapping inputs; query frontiers without native execution.

Registered poses are taken from the original accepted pose receipts. This is a
mapping replay, not a new validation of registration or an alternative trajectory.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lewm.current_plane_floor_coverage_development import CurrentPlaneFloorRoutingMap
from lewm.observed_floor_waypoint_development import centre
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.run_go2_contact_score_ablation_development import ContactScoreRuntime


def saved_pose(pose, *, identity, now_ns):
    if identity != (0, 0, 0) or pose['measured_ns'] != now_ns:
        raise ValueError('recorded registered pose must match mapping observation')
    return (np.asarray(pose['position_initial_body_m']),
        np.asarray(pose['rotation_initial_body_from_current_body']), pose)


class RecordedPoseMap(CurrentPlaneFloorRoutingMap):
    _read_pose = staticmethod(saved_pose)


def save(directory, name, value):
    with (directory / name).open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    args = parser.parse_args(); root = path(args.root_name)
    if (root / 'depth_retention.json').exists():
        raise ValueError('full retained depth required')
    output = root / 'frontier_map_reconstruction_v1'; output.mkdir()
    launch = read(root, 'launch.json')
    sources = launch['source_sha256'] | launch['extra_sources']
    checked = {}
    for name in ('lewm/current_plane_floor_coverage_development.py',
            'lewm/multirate_routing_map_development.py',
            'lewm/current_pair_routing_memory_development.py',
            'lewm/frontier_visit_runtime_development.py'):
        digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        expected = sources.get(name, sources.get(str(Path(name).resolve())))
        if digest != expected:
            raise ValueError(f'recorded mapping source differs: {name}')
        checked[name] = digest
    poses = {r['frame']: r['registered_pose'] for r in read(root, 'poses.json')}
    maps = sorted((r for r in read(root, 'stage_events.json') if r['stage'] == 'mapping'),
        key=lambda r: r['completed_ns'])
    plans = [r for r in read(root, 'planning.json') if 'selection' in r]
    by_map = defaultdict(list)
    for row in plans: by_map[row['map_frame']].append(row)
    visits = read(root, 'frontier_visits.json')
    events_by_map = defaultdict(list)
    for event in visits['events']: events_by_map[event['map_frame']].append(event)
    probe_frames = {plans[-1]['frame']}
    for event in visits['events']:
        for stamp in (event['started_ns'], event['completed_ns']):
            probe_frames.add(min(plans, key=lambda r: abs(r['measured_ns']-stamp))['frame'])
    for status in ('OBSERVED_COMPONENT_HAS_NO_FRONTIER',):
        found = [r for r in plans if r['route_status'] == status]
        if found: probe_frames.add(found[0]['frame'])
    probe_frames.add(min(plans, key=lambda r: abs(r['frame']-2540))['frame'])
    save(output, 'launch.json', dict(root=root.name, sources=checked,
        mapping_frames=[r['frame'] for r in maps], probe_planning_frames=sorted(probe_frames),
        original_registered_pose_receipts_used=True, registration_revalidated=False,
        delivered_noise_hashes_checked_by_reader=True, native_state_input=False,
        closed_loop_execution=False))
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    mapper = RecordedPoseMap(); reader = NoisyPublicReplay(root / 'native')
    runtime = ContactScoreRuntime.__new__(ContactScoreRuntime)
    runtime.mission_latest = dict(phase='OUTBOUND')
    goal = launch['public_mission']['goal_initial_body_xy_m']
    exclusion_history = []; matched = 0; probes = []; started = time.monotonic()
    for index, row in enumerate(maps):
        frame = row['frame']; p, d, _, _, auxiliary, now = reader.packet(frame)
        snapshot = mapper.update(p, d, poses[frame], auxiliary_depth=auxiliary, measured_ns=now)
        for event in events_by_map[frame]:
            cells = {c for c in snapshot.floor if np.linalg.norm(centre(c)-event['target_xy_m']) <= .10}
            deferred = event.get('exclusion_deferred_until_arrival', False)
            expected = event['deferred_excluded_cells'] if deferred else event['excluded_cells']
            if len(cells) != expected or (deferred and event['excluded_cells'] != 0):
                raise ValueError('reconstructed completed-view exclusion differs')
            exclusion_history.append((event['completed_ns'], set() if deferred else cells))
        selected = [r for r in by_map[frame] if r['frame'] in probe_frames]
        if frame == maps[-1]['frame']:
            selected.append(dict(frame=frame, measured_ns=now, route_status=None, synthetic_query=True))
        for plan in by_map[frame]:
            receipt = plan['selection']['routing_memory_scope']
            if (receipt['routing_floor_cells'] != len(snapshot.floor)
                    or receipt['routing_fine_obstacle_cells'] != len(snapshot.fine_occupied)):
                raise ValueError(f'map counts differ at planning frame {plan["frame"]}')
            matched += 1
        if selected:
            s = asdict(snapshot)
            for key, value in s.items():
                if isinstance(value, frozenset): s[key] = sorted(value)
            save(output, f'map_{frame:04d}.json', s)
        for plan in selected:
            excluded = set().union(*(cells for stamp, cells in exclusion_history
                if stamp <= plan['measured_ns']))
            B = np.asarray(snapshot.map_from_initial)
            position = (B @ np.asarray(poses[plan['frame']]['position_initial_body_m']))[:2]
            mapped_goal = (B @ np.r_[goal, 0.])[:2]
            results = {}
            for mode, excluded_cells in (('recorded_exclusions', excluded), ('no_exclusions', set())):
                runtime.frontier_visits = SimpleNamespace(excluded=excluded_cells)
                propose = runtime._routing_proposer(snapshot)
                results[mode] = propose(snapshot.floor, snapshot.occupied, position, mapped_goal)
            if (plan['route_status'] in ('OBSERVED_COMPONENT_HAS_NO_FRONTIER',
                    'OBSERVED_FLOOR_ROUTE_TO_FRONTIER', 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL')
                    and results['recorded_exclusions']['status'] != plan['route_status']):
                raise ValueError('reconstructed base route differs from recorded route status')
            probes.append(dict(planning_frame=plan['frame'], map_frame=frame,
                measured_ns=plan['measured_ns'], position_map_xy_m=position.tolist(),
                excluded_cells=sorted(excluded), recorded_route_status=plan['route_status'],
                synthetic_query=plan.get('synthetic_query', False), **results))
            print('FRONTIER_QUERY', plan['frame'], {k:v['status'] for k,v in results.items()}, flush=True)
        if index % 100 == 0: print('MAP_REPLAY', frame, 'elapsed_s', round(time.monotonic()-started, 1), flush=True)
    union = set().union(*(cells for _, cells in exclusion_history))
    if union != {tuple(c) for c in visits['excluded_cells']}:
        raise ValueError('final reconstructed frontier exclusions differ')
    save(output, 'probes.json', probes)
    result = dict(mapping_updates=len(maps), planning_map_count_receipts_matched=matched,
        completed_view_exclusions_matched=len(exclusion_history), final_excluded_cells=len(union),
        final_exclusion_set_exactly_matched=True, probes=len(probes),
        elapsed_s=time.monotonic()-started, native_state_input=False,
        same_cell_counts_do_not_alone_prove_identical_maps=True,
        alternative_route_queries_are_not_alternative_navigation=True)
    save(output, 'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
