"""Complete-trace surface-memory replay and fixed saved-plan footprint queries."""
import json
import time
import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_family_transition_goal_probe_v1 import IDS, INPUT, TRIALS, timing
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE/'go2_joint_visual_surface_memory_v1_attempt_001'
PROTOCOL = 'docs/go2_joint_visual_surface_memory_v1_2026-09-08.md'


def replay(trial, geometry):
    rows = read_json(INPUT/trial, 'context_decisions.json')
    reader = IntentReturnRGBDReplay(INPUT/trial)
    observer = CornerSupportVisualLedMotion(identity=(0, 0, 0))
    memory = JointVisualSurfaceMemory(identity=(0, 0, 0))
    records, predictions, elapsed = [], [], []
    stopped = False
    for tick, original in enumerate(rows):
        p, d, fast, now = reader.packet(tick)
        if stopped:
            assert original['decision']['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
            assert original['decision']['requested_command'] == [0., 0., 0.]
            try:
                memory.backtrack(0, now_ns=now)
            except SensorContractError:
                records.append(dict(frame=tick, status='UNAVAILABLE_AFTER_VISUAL_FAILURE'))
                continue
            raise AssertionError('failed memory query was admitted')
        evidence = observer.observe(p, d, fast, now_ns=now)
        assert json.loads(json.dumps(evidence)) == original['decision']['evidence'], ('raw observer replay', tick)
        start = time.perf_counter_ns()
        try:
            receipt = memory.observe(p, d, evidence, now_ns=now)
        except SensorContractError:
            assert evidence['current_pose'] is None and memory.failed
            stopped = True
            records.append(dict(frame=tick, status='CURRENT_POSE_UNAVAILABLE', failure_latched=True))
            continue
        elapsed.append((time.perf_counter_ns()-start)/1e6)
        current = memory.footprint(geometry, [0., 0.], 0., now_ns=now)
        records.append(dict(frame=tick, status='CURRENT_MEASURED_MEMORY', receipt=receipt,
            current_posture_intersections=[r for r in current['shapes'] if r['intersecting_voxels']]))
        selected = original['decision']['new_selection']
        if selected is not None:
            array = np.asarray(selected['prediction'], float)
            assert array.shape == (6, 8, 5) and np.isfinite(array).all()
            for action, path in zip(selected['candidates'], array, strict=True):
                for index in (0, 7):
                    dx, dy, sy, cy, _ = path[index]
                    assert np.hypot(sy, cy) > 1e-8
                    yaw = float(np.arctan2(sy, cy))
                    views = {}
                    for persistent in (False, True):
                        result = memory.footprint(geometry, [dx, dy], yaw, now_ns=now, persistent=persistent)
                        views[result['memory']] = [r for r in result['shapes'] if r['intersecting_voxels']]
                    # Every current return is also retained; memory cannot erase conflict.
                    assert {r['shape_id'] for r in views['current_frame']} <= {r['shape_id'] for r in views['persistent']}
                    predictions.append(dict(frame=tick, action=action['action'],
                        selected=action['action'] == selected['action'], horizon_ns=(index+1)*500_000_000,
                        discrete_predicted_body_xy_m=[float(dx), float(dy)], views=views,
                        clearance_established=False, candidate_executed_by_readout=False))
    if not memory.failed:
        route = memory.backtrack(0, now_ns=memory.last_ns)
        assert [r['frame'] for r in route['targets']] == list(range(len(memory.route)-2, -1, -1))
    else:
        route = dict(status='UNAVAILABLE_AFTER_VISUAL_FAILURE', motion_permitted=False)
    return dict(trial=trial, frames=len(rows), indexed_frames=len(memory.route),
        raw_observer_replay_pass=True, records=records, predictions=predictions,
        backtrack_proposal=route, memory_update_timing=timing(elapsed),
        maximum_retained_voxels=len(memory.index.cells), native_execution=False,
        clearance_established=False, exploration_demonstrated=False, physical_backtracking_demonstrated=False)


def main():
    if not __debug__:
        raise ValueError('audit assertions required')
    cv2.setNumThreads(1)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive new surface-memory replay')
    verify_artifacts(INPUT, IDS)
    original = read_json(INPUT, 'launch.json')
    result = read_json(INPUT, 'result.json')
    assert result['status'] == 'FAMILY_TRANSITION_GOAL_PROBE_COMPLETE'
    bindings = IDS | result['artifact_sha256']
    verify_artifacts(INPUT, bindings)
    geometry = ArticulatedCollisionGeometry(URDF)
    sources = discover_sources((PROTOCOL, 'scripts/probe_go2_joint_visual_surface_memory_v1.py',
        'lewm/tests/test_joint_visual_surface_memory_development.py'), original['source_sha256'])
    launch = original | dict(source_sha256=sources, protocol=PROTOCOL, output_root=str(OUTPUT),
        probe_artifact_sha256=bindings, native_execution=False, model_training=False,
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF))
    verify(launch)
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    started = time.perf_counter()
    try:
        summaries = []
        artifacts = {}
        for trial in TRIALS:
            report = replay(trial, geometry)
            path = OUTPUT/(trial+'_memory.json'); write_json(path, report)
            artifacts[path.name] = digest(path)
            summaries.append({k:v for k,v in report.items() if k not in ('records', 'predictions', 'backtrack_proposal')})
            print('MEMORY_REPLAY_CASE', trial, report['indexed_frames'], flush=True)
        verify(launch); verify_artifacts(INPUT, bindings)
        assert digest(URDF) == launch['robot_urdf_sha256']
        verify_artifacts(OUTPUT, artifacts)
        write_json(OUTPUT/'result.json', dict(status='JOINT_VISUAL_SURFACE_MEMORY_REPLAY_COMPLETE',
            conditions=summaries, artifact_sha256=artifacts, source_sha256=sources,
            launch_sha256=digest(OUTPUT/'launch.json'), wall_s=time.perf_counter()-started,
            source_and_input_unchanged=True, native_execution=False, original_outcomes_changed=False,
            navigation_qualified=False, goal_achieved=False))
        print('MEMORY_REPLAY_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEMORY_REPLAY_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
