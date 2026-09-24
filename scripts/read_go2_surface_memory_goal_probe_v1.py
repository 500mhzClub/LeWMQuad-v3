"""Paired actual-navigation outcomes and first command divergence witnesses."""
import argparse
from collections import Counter
import json
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_surface_memory_goal_probe_v1 import OUTPUT as INPUT, CASES, TRIALS
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts, validate_root
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing

OUTPUT = BASE/'go2_surface_memory_goal_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_surface_memory_goal_readout_v1_2026-09-08.md'
LAUNCH_SHA = '055f7eafac1e275bc579d7e98ff830da2ada750b20e9b639b3385177d790ecb9'


def exact_nested(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a: exact_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True): exact_nested(x, y)
    else:
        assert a == b


def case_summary(record):
    name = record['case']
    rows = read_json(INPUT/name, 'context_decisions.json')
    audit = read_json(INPUT, name+'_audit.json')
    selections = [r['decision']['new_selection'] for r in rows if r['decision']['new_selection'] is not None]
    assert len(selections) == record['selection_count']
    return dict(case=name, trial=record['trial'], persistent=record['persistent'],
        collection=record['collection'], goal=record['goal'],
        selections=len(selections), selected_actions=dict(Counter(str(s['action']) for s in selections)),
        changed_from_unfiltered=int(sum(s['action'] != s['unfiltered_action'] for s in selections)),
        total_filtered_candidates=sum(6-s['no_surface_conflict_candidates'] for s in selections),
        raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
        raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
        strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=audit['hard_measurement_failed_frames'],
        maximum_observed_pose_xy_error_m=max(audit['observed_pose_xy_errors_m'], default=None),
        timing={k: timing([r[k] for r in rows if k in r]) for k in
            ('acquisition_wall_ms','controller_wall_ms','observation_and_control_wall_ms','iteration_with_command_wall_ms')})


def paired(trial, summaries):
    names = ['current_frame_'+trial, 'persistent_'+trial]
    rows = [read_json(INPUT/n, 'context_decisions.json') for n in names]
    first_command = next((i for i in range(min(map(len, rows)))
        if rows[0][i]['decision']['requested_command'] != rows[1][i]['decision']['requested_command']), None)
    first = next((i for i in range(min(map(len, rows)))
        if any(rows[0][i]['decision'][k] != rows[1][i]['decision'][k]
            for k in ('selected_action', 'terminal', 'requested_command'))), None)
    assert first_command is None or first is not None and first <= first_command
    witness = None
    if first is not None:
        readers = [IntentReturnRGBDReplay(INPUT/n) for n in names]
        # Require the complete public histories through divergence, not just an
        # equal scene label or final RGB frame. Never compare post-divergence sensors.
        for i in range(first+1):
            exact_nested(readers[0].packet(i), readers[1].packet(i))
            exact_nested(rows[0][i]['decision']['evidence'], rows[1][i]['decision']['evidence'])
        choices = [r[first]['decision']['new_selection'] for r in rows]
        assert all(c is not None for c in choices)
        exact_nested(choices[0]['prediction'], choices[1]['prediction'])
        exact_nested(choices[0]['candidates'], choices[1]['candidates'])
        witness = dict(tick=first, same_complete_public_history_and_observer=True,
            identical_model_predictions_and_original_utilities=True,
            choices=[dict(variant=c['surface_memory_variant'], action=c['action'],
                unfiltered_action=c['unfiltered_action'],
                no_surface_conflict_candidates=c['no_surface_conflict_candidates'],
                filtered_shapes=[[s['shape_id'] for s in check['shapes'] if s['intersecting_voxels']]
                    for check in c['surface_checks']]) for c in choices])
    a, b = [next(s for s in summaries if s['case'] == n) for n in names]
    return dict(trial=trial, first_selection_or_terminal_divergence=witness,
        first_command_divergence_tick=first_command,
        identical_commands_over_shared_observed_prefix=first_command is None,
        persistent_minus_current_terminal_goal_distance_m=b['goal']['terminal_distance_m']-a['goal']['terminal_distance_m'],
        current_verified_goal=a['goal']['verified_goal_reached'], persistent_verified_goal=b['goal']['verified_goal_reached'],
        independent_maze_pair=False, statistical_benefit_claim=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe-result-sha256', required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('audit assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired result readout')
    ids = {'launch.json': LAUNCH_SHA, 'result.json': args.probe_result_sha256}
    verify_artifacts(INPUT, ids)
    result, original = read_json(INPUT, 'result.json'), read_json(INPUT, 'launch.json')
    assert result['status'] == 'SURFACE_MEMORY_GOAL_PROBE_COMPLETE' and result['cases'] == [list(c) for c in CASES]
    bindings = ids | result['artifact_sha256']; verify_artifacts(INPUT, bindings)
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_surface_memory_goal_probe_v1.py'), original['source_sha256'])
    launch = original | dict(source_sha256=sources, protocol=PROTOCOL, output_root=str(OUTPUT),
        input_probe_artifact_sha256=bindings, native_execution=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    try:
        summaries = [case_summary(r) for r in result['conditions']]
        pairs = [paired(t, summaries) for t in TRIALS]
        verify(launch); verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SURFACE_MEMORY_GOAL_PAIRED_READOUT_COMPLETE',
            conditions=summaries, pairs=pairs, probe_sha256=ids, source_sha256=sources,
            launch_sha256=digest(OUTPUT/'launch.json'), native_execution=False,
            original_outcomes_changed=False, navigation_qualified=False, goal_achieved=False))
        print('SURFACE_MEMORY_PAIRED_READOUT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_PAIRED_READOUT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
