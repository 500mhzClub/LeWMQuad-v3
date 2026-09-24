"""Complete recorded-sensor floor-map/waypoint replay, no native execution."""
from collections import Counter
import json
import time
import cv2
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.joint_visual_floor_map_development import JointVisualFloorMap
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.read_go2_family_transition_goal_probe_v1 import INPUT, IDS, TRIALS, timing
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

OUTPUT = BASE/'go2_observed_floor_waypoint_v1_attempt_001'
SURFACE = BASE/'go2_joint_visual_surface_memory_v1_attempt_001'
SURFACE_IDS = {'launch.json':'21b63a30376c25fcc20dbfd86dc035eb4d411d576a20b10a9869b078d6b298c0',
    'result.json':'d19b8254779c7aafb7ef4d0bee612a0d266da5236f1bea831f376d775c79dcfe'}
PROTOCOL = 'docs/go2_observed_floor_waypoint_v1_2026-09-08.md'


def replay(trial):
    rows = read_json(INPUT/trial, 'context_decisions.json')
    reader = IntentReturnRGBDReplay(INPUT/trial)
    observer = CornerSupportVisualLedMotion(identity=(0, 0, 0))
    mapper = JointVisualFloorMap()
    reports = []; updates = []; planning = []
    for i, original in enumerate(rows):
        policy, depth, fast, now = reader.packet(i)
        if original['decision']['evidence'] is None:
            assert mapper.failed and original['decision']['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
            reports.append(dict(frame=i, status='UNAVAILABLE_AFTER_ORIGINAL_VISUAL_FAILURE'))
            continue
        evidence = observer.observe(policy, depth, fast, now_ns=now)
        assert json.loads(json.dumps(evidence)) == original['decision']['evidence'], ('raw observer replay', i)
        if mapper.failed:
            reports.append(dict(frame=i, status='FLOOR_MAP_FAILURE_LATCHED'))
            continue
        start = time.perf_counter_ns()
        try:
            receipt = mapper.observe(policy, depth, evidence, now_ns=now)
        except SensorContractError as error:
            reports.append(dict(frame=i, status='FLOOR_MAP_FAILURE', reason=str(error),
                cause=str(error.__cause__), current_visual_pose_available=evidence['current_pose'] is not None))
            continue
        updates.append((time.perf_counter_ns()-start)/1e6)
        start = time.perf_counter_ns()
        proposal = mapper.waypoint([1.2, 0.], now_ns=now)
        planning.append((time.perf_counter_ns()-start)/1e6)
        assert not proposal['motion_permitted'] and not proposal['navigation_qualified']
        reports.append(dict(frame=i, status='CURRENT_FLOOR_MAP', receipt=receipt, proposal=proposal))
    return dict(trial=trial, frames=len(rows), raw_observer_replay_pass=True, records=reports,
        status_counts=dict(Counter(r.get('proposal',r)['status'] for r in reports)),
        update_timing=timing(updates), proposal_timing=timing(planning),
        floor_cells=[dict(cell=list(c),first_frame=f) for c,f in sorted(mapper.floor.items())],
        occupied_cells=[dict(cell=list(c),first_frame=f) for c,f in sorted(mapper.occupied.items())],
        original_goal_reached=False, native_execution=False, navigation_qualified=False)


def main():
    if not __debug__: raise ValueError('audit assertions required')
    cv2.setNumThreads(1)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive floor-waypoint replay')
    verify_artifacts(INPUT, IDS); verify_artifacts(SURFACE, SURFACE_IDS)
    original = read_json(INPUT,'launch.json'); native = read_json(INPUT,'result.json')
    surface = read_json(SURFACE,'result.json')
    assert native['status']=='FAMILY_TRANSITION_GOAL_PROBE_COMPLETE'
    assert surface['status']=='JOINT_VISUAL_SURFACE_MEMORY_REPLAY_COMPLETE'
    bindings = IDS|native['artifact_sha256']; verify_artifacts(INPUT,bindings)
    verify_artifacts(SURFACE,surface['artifact_sha256'])
    inherited = original['source_sha256'].copy()
    for name, sha in surface['source_sha256'].items():
        assert inherited.get(name,sha)==sha; inherited[name]=sha
    sources = discover_sources((PROTOCOL,'scripts/probe_go2_observed_floor_waypoint_v1.py',
        'lewm/tests/test_observed_floor_waypoint_development.py'),inherited)
    launch = original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        probe_artifact_sha256=bindings,surface_sha256=SURFACE_IDS,native_execution=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch)
    started=time.perf_counter()
    try:
        artifacts={};summaries=[]
        for trial in TRIALS:
            report=replay(trial);name=trial+'_floor_map.json';write_json(OUTPUT/name,report)
            artifacts[name]=digest(OUTPUT/name)
            summaries.append({k:v for k,v in report.items() if k not in ('records','floor_cells','occupied_cells')}
                |dict(retained_floor_cells=len(report['floor_cells']),retained_occupied_cells=len(report['occupied_cells'])))
            print('FLOOR_WAYPOINT_CASE',trial,report['status_counts'],flush=True)
        verify(launch);verify_artifacts(INPUT,bindings)
        verify_artifacts(SURFACE,SURFACE_IDS|surface['artifact_sha256']);verify_artifacts(OUTPUT,artifacts)
        write_json(OUTPUT/'result.json',dict(status='OBSERVED_FLOOR_WAYPOINT_REPLAY_COMPLETE',
            conditions=summaries,artifact_sha256=artifacts,source_sha256=sources,
            launch_sha256=digest(OUTPUT/'launch.json'),wall_s=time.perf_counter()-started,
            native_execution=False,original_outcomes_changed=False,navigation_qualified=False,goal_achieved=False))
        print('FLOOR_WAYPOINT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FLOOR_WAYPOINT_REPLAY_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
