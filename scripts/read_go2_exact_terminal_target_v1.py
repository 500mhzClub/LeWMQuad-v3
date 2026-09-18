"""Bounded exact-terminal-target scoring on authenticated executed contexts."""
import json
import time
import numpy as np
from lewm.exact_terminal_waypoint_development import exact_terminal_target
from scripts.run_go2_overlap_retention_goal_probe_v1 import OUTPUT as INPUT
from scripts.read_go2_overlap_retention_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_exact_terminal_target_v1_attempt_001'
PROTOCOL = 'docs/go2_exact_terminal_target_v1_2026-09-08.md'
INPUT_SHA = 'fa67b645ae0439b803adce6d1a73c90e29862daeb8f8cd9b037183acd1477765'
READOUT_SHA = '4f38b4fc8cca26808eb248db5d6484c5b68f29c73c2e14ea2f2a5dd82bc76799'
CASE = 'full_direct_family_episode_039'


def analyze():
    rows = read_json(INPUT/CASE, 'context_decisions.json')
    records = []
    for row in rows:
        d = row['decision']; selection = d['new_selection']
        if selection is None:
            continue
        B = np.asarray(d['memory_receipt']['map_from_initial'])
        pose = d['evidence']['current_pose']
        p = B@np.asarray(pose['position_initial_body_m'])
        R = B@np.asarray(pose['rotation_initial_body_from_current_body'])
        g = (B@np.array([1.2, 0., 0.]))[:2]
        changed = exact_terminal_target(selection, p, R, g)
        for k in ('prediction', 'surface_checks', 'nominal_action_checks', 'proposal'):
            assert changed[k] == selection[k], k
        if changed['action_index'] is not None:
            i = changed['action_index']
            assert not changed['surface_checks'][i]['possible_intersection']
            assert changed['nominal_action_checks'][i]['nominal_disk_connector_clear']
        records.append(dict(tick=row['tick'], original_action=selection['action'],
            retargeted=changed is not selection, action_changed=changed['action'] != selection['action'],
            mission_map_xy_m=g.tolist(), observed_goal_distance_m=d['observed_goal_distance_m'],
            exact_target_selection=changed, unexecuted_outcome_inferred=False))
    assert len(rows) == 240 and len(records) == 46
    return records


def main():
    if not __debug__:
        raise ValueError('enabled audit assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive exact-target diagnostic; no retry/resume')
    verify_artifacts(INPUT, {'result.json': INPUT_SHA})
    prior = read_json(INPUT, 'result.json')
    assert prior['status'] == 'OVERLAP_RETENTION_GOAL_PROBE_COMPLETE'
    ids = {'result.json': INPUT_SHA} | prior['artifact_sha256']
    verify_artifacts(INPUT, ids)
    verify_artifacts(READOUT, {'result.json': READOUT_SHA})
    readout = read_json(READOUT, 'result.json')
    assert readout['status'] == 'OVERLAP_RETENTION_GOAL_READOUT_COMPLETE'
    assert readout['probe_result_sha256'] == INPUT_SHA
    readout_ids = {'result.json': READOUT_SHA, 'launch.json': readout['launch_sha256']}
    verify_artifacts(READOUT, readout_ids)
    original = read_json(READOUT, 'launch.json'); verify(original)
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_exact_terminal_target_v1.py',
        'lewm/tests/test_exact_terminal_waypoint_development.py',
        'docs/go2_overlap_retention_goal_probe_result_2026-09-08.md'), original['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+256*1024**2:
        raise ValueError('diagnostic resource allowance unavailable')
    launch = original | dict(source_sha256=sources, output_root=str(OUTPUT), protocol=PROTOCOL,
        probe_artifact_sha256=ids, readout_artifact_sha256=readout_ids, hardware=resources,
        native_execution=False, workers=1, threads=1,
        concurrency_reason='46 small dependent-context readouts; no independent native or model work',
        mission_goal_initial_body_xy_m=[1.2, 0.], maximum_output_bytes=256*1024**2)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('EXACT_TERMINAL_TARGET_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        records = analyze()
        assert len(json.dumps(records, allow_nan=False).encode()) < 256*1024**2
        write_json(OUTPUT/'contexts.json', records)
        verify(launch); verify_artifacts(INPUT, ids); verify_artifacts(READOUT, readout_ids)
        products = {f: digest(OUTPUT/f) for f in ('launch.json', 'contexts.json')}
        write_json(OUTPUT/'result.json', dict(status='EXACT_TERMINAL_TARGET_DIAGNOSTIC_COMPLETE',
            contexts=len(records), retargeted_contexts=sum(r['retargeted'] for r in records),
            conditional_action_changes=[dict(tick=r['tick'], original=r['original_action'],
                retargeted=r['exact_target_selection']['action']) for r in records if r['action_changed']],
            source_sha256=sources, artifact_sha256=products, hardware_after=hardware(),
            wall_s=time.perf_counter()-started, original_outcomes_changed=False,
            all_forecasts_and_geometric_constraints_unchanged=True,
            native_execution=False, model_training=False, goal_achieved=False, navigation_qualified=False))
        print('EXACT_TERMINAL_TARGET_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_EXACT_TARGET_DIAGNOSTIC_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
