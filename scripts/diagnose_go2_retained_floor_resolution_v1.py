"""Later measured floor visibility of the actual terminal ambiguous enclosures."""
import itertools
import json
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.joint_visual_floor_map_development import floor_coverage
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.joint_floor_registered_maze_episode_development import artifacts
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

INPUT = BASE/'go2_joint_floor_registered_maze_pilot_v1_attempt_001'
CASE = 'full_jepa_novel_maze_00'
OUTPUT = BASE/'go2_retained_floor_resolution_diagnosis_v1_attempt_001'
PROTOCOL = 'docs/go2_retained_floor_resolution_diagnosis_v1_2026-09-09.md'
COLLECTION_SHA = 'a07f53205344d5e6fc58f754d3180268eacae8e2e8ea684e5850837124caf261'
STREAM_SHA = 'e92d311fed32b3886f26eefc4f14c999cd5b37a7bfde6c6b54532cd0524fa64a'
LAUNCH_SHA = '7478a094256d99aa3c25958806707730efb07eb3277dd83e411437b7d9ee98a5'


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive provisional diagnosis required')
    initial = {'launch.json': LAUNCH_SHA, CASE+'/result.json': COLLECTION_SHA,
        CASE+'/context_decisions.jsonl.gz': STREAM_SHA}
    verify_artifacts(INPUT, initial)
    old = read_json(INPUT, 'launch.json'); verify(old)
    collection = read_json(INPUT, CASE+'/result.json')
    assert collection['command_ticks'] == 1067 and collection['mission_receipt']['frame'] == 1057
    bindings = {CASE+'/'+n:digest(INPUT/CASE/n) for n in artifacts(0, collection)} | initial
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_retained_floor_resolution_v1.py'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('provisional diagnosis resource envelope unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        input_artifact_sha256=bindings, hardware=resources, native_execution=False, model_training=False,
        model_loaded=False, native_scene_workers=0, cpu_processes=1, numerical_threads=1,
        native_final_audit_required=True, contact_policy_modified=False)
    verify(launch); verify_artifacts(INPUT, bindings)
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('RETAINED_FLOOR_RESOLUTION_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        terminal = next(r['decision'] for r in read_rows(INPUT/CASE) if r['tick'] == 1057)
        memory = terminal['memory_receipt']; B = np.asarray(memory['map_from_initial']); h = memory['floor_height_map_m']
        targets = {}
        for check in terminal['new_selection']['surface_checks']:
            for camera in ('shapes', 'auxiliary_shapes'):
                for hit in check[camera]:
                    if not hit['intersecting_voxels']: continue
                    assert hit['intersecting_voxels'] == 1 and hit['shape_id'] in ('FL_foot:0', 'FR_foot:0', 'RL_foot:0', 'RR_foot:0')
                    key = (camera, tuple(hit['first_cell']))
                    if key in targets: continue
                    low, high = np.asarray(hit['first_bounds_m'])
                    points = np.array(list(itertools.product(*zip(low, high))))@B.T
                    lo, hi = points.min(0), points.max(0)
                    a, b = np.floor(lo[:2]/.05).astype(int), np.floor(hi[:2]/.05).astype(int)
                    cells = [[i,j] for i in range(a[0],b[0]+1) for j in range(a[1],b[1]+1)]
                    assert 1 <= len(cells) <= 16
                    targets[key] = dict(source_camera=camera, cell=list(key[1]), bounds_initial_body_m=hit['first_bounds_m'],
                        latest_sample_frame=hit['first_bounds_latest_frame'], map_cells=cells,
                        height_relative_floor_m=[float(lo[2]-h),float(hi[2]-h)],
                        entire_enclosure_in_original_height_band=bool(lo[2]>=h-.01 and hi[2]<=h+.01),
                        later_complete_coverage=None)
        assert len(targets) == 6
        cells = sorted({tuple(c) for t in targets.values() for c in t['map_cells']})
        positions = {c:i for i,c in enumerate(cells)}
        queries = np.asarray(cells, dtype=np.int64)
        reader = IntentReturnRGBDReplay(INPUT/CASE)
        acquisitions = read_json(INPUT, CASE+'/auxiliary_camera_audit.json')
        minimum = min(t['latest_sample_frame'] for t in targets.values())
        examined = 0
        for row in read_rows(INPUT/CASE):
            frame = row['tick']
            if frame <= minimum: continue
            if frame > 1057: break
            d = row['decision']; pose = d['evidence']['current_pose']; m = d['memory_receipt']
            assert m['floor_height_map_m'] == h and m['map_from_initial'] == B.tolist()
            R = B@np.asarray(pose['rotation_initial_body_from_current_body']); p = B@np.asarray(pose['position_initial_body_m'])
            policy, depth, _, now = reader.packet(frame)
            auxiliary = packet(INPUT/CASE, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
            Q, q = reference_pose(R, p)
            views = [('primary',depth,R,p),('auxiliary',auxiliary,Q,q)]
            for camera, packet_depth, rotation, position in views:
                coverage = floor_coverage(packet_depth['depth_m'],packet_depth['valid'],rotation,position,h,queries)
                for t in targets.values():
                    if t['later_complete_coverage'] is not None or frame <= t['latest_sample_frame']: continue
                    ids = [positions[tuple(c)] for c in t['map_cells']]
                    if t['entire_enclosure_in_original_height_band'] and coverage['covered'][ids].all():
                        t['later_complete_coverage'] = dict(frame=frame, measured_ns=now, camera=camera,
                            map_cells=t['map_cells'], all_enclosing_grid_squares_covered=True,
                            pose_and_flat_floor_hypotheses_required=True)
                        print('RETAINED_FLOOR_RESOLVED', t['source_camera'], t['cell'], frame, camera, flush=True)
            examined += 1
            if all(t['later_complete_coverage'] is not None for t in targets.values()): break
            if frame % 100 == 0: print('RETAINED_FLOOR_FRAME',frame,flush=True)
        verify(launch); verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='RETAINED_FLOOR_RESOLUTION_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), source_sha256=sources, targets=list(targets.values()),
            observations_examined=examined, native_final_audit_required=True, hardware_after=hardware(),
            contact_policy_modified=False, original_classifications_unchanged=True,
            calibrated_enclosure_or_pose_bounds=False, native_execution=False, model_training=False,
            navigation_qualified=False, goal_achieved=False))
        print('RETAINED_FLOOR_RESOLUTION_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='RETAINED_FLOOR_RESOLUTION_DIAGNOSIS_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
