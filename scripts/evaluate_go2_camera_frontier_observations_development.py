"""Reconstruct acquired mapping evidence behind camera-view completion records."""
import argparse
from collections import defaultdict
import json
import time
import cv2
import torch

from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.reconstruct_go2_frontier_stall_development import RecordedPoseMap


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root-name', required=True)
    root = path(parser.parse_args().root_name)
    output = root/'camera_frontier_observation_replay_v1'; output.mkdir()
    poses = {r['frame']:r['registered_pose'] for r in read(root, 'poses.json')}
    updates = sorted((r for r in read(root, 'stage_events.json') if r['stage'] == 'mapping'),
        key=lambda r:r['completed_ns'])
    events = read(root, 'frontier_visits.json')['events']; by_frame = defaultdict(list)
    for index, event in enumerate(events): by_frame[event['map_frame']].append((index, event))
    plans = defaultdict(list)
    for row in read(root, 'planning.json'):
        if 'selection' in row: plans[row['map_frame']].append(row)
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = NoisyPublicReplay(root/'native'); mapper = RecordedPoseMap()
    rows = []; matched = 0; first_observed = {}; started = time.monotonic()
    requested = {tuple(e['unknown_neighbour']) for e in events if 'unknown_neighbour' in e}
    for index, update in enumerate(updates):
        frame = update['frame']; p, d, _, _, auxiliary, now = reader.packet(frame)
        snapshot = mapper.update(p, d, poses[frame], auxiliary_depth=auxiliary, measured_ns=now)
        observed = snapshot.floor | snapshot.occupied
        for cell in requested & observed:
            first_observed.setdefault(cell, frame)
        for plan in plans[frame]:
            scope = plan['selection']['routing_memory_scope']
            if (scope['routing_floor_cells'] != len(snapshot.floor)
                    or scope['routing_fine_obstacle_cells'] != len(snapshot.fine_occupied)):
                raise ValueError(f'recorded mapping counts differ at plan {plan["frame"]}')
            matched += 1
        for event_index, event in by_frame[frame]:
            cell = tuple(event['unknown_neighbour']) if 'unknown_neighbour' in event else None
            observed_now = None if cell is None else cell in observed
            reason = event['completion_reason']
            if reason == 'REQUESTED_PATCH_OBSERVED' and observed_now is not True:
                raise ValueError('claimed patch observation missing from reconstructed map')
            if reason == 'FRESH_VIEW_PATCH_STILL_UNKNOWN' and observed_now is not False:
                raise ValueError('claimed unknown patch already observed in reconstructed map')
            rows.append(dict(event_index=event_index, map_frame=frame, reason=reason,
                requested_cell=None if cell is None else list(cell),
                patch_in_observed_floor=None if cell is None else cell in snapshot.floor,
                patch_in_observed_obstacles=None if cell is None else cell in snapshot.occupied,
                first_mapped_observation_frame=first_observed.get(cell),
                observation_claim_matched=observed_now is True if
                    reason == 'REQUESTED_PATCH_OBSERVED' else None))
        if index % 100 == 0:
            print('CAMERA_VIEW_MAP_REPLAY', frame, 'elapsed_s', round(time.monotonic()-started, 1), flush=True)
    if len(rows) != len(events): raise ValueError('not every event mapping frame was replayed')
    result = dict(mapping_updates=len(updates), planning_map_count_receipts_matched=matched,
        events=len(rows), observed_patch_events=sum(r['reason']=='REQUESTED_PATCH_OBSERVED' for r in rows),
        fresh_unresolved_view_events=sum(r['reason']=='FRESH_VIEW_PATCH_STILL_UNKNOWN' for r in rows),
        rows=rows, elapsed_s=time.monotonic()-started, original_registered_poses_used=True,
        raw_tracking_revalidated=False, delivered_noise_packet_hashes_checked=True,
        native_state_used=False, hypothetical_projection_used_to_admit_floor=False,
        alternative_trajectory_evaluated=False)
    with (output/'result.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__': main()
