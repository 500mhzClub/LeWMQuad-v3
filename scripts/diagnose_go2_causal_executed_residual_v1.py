"""Bounded prequential public residual diagnostic, with native evaluation only."""
import time
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.executed_prefix_motion_diagnosis_development import diagnose
from lewm.causal_executed_residual_diagnosis_development import replay_bias, WINDOW_TICKS
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from scripts.run_go2_executed_horizon_final_goal_probe_v1 import OUTPUT as INPUT, PREVIOUS, PREVIOUS_SHA, CASES
from scripts.read_go2_executed_horizon_final_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

INPUT_SHA = 'dabc90b19f1d8285d21fd81d265efae504fbe75b1868357f6bcd17f2e77fe062'
READOUT_SHA = 'edd050780b55f45956cb685d57b989cb748ed90227bc8937bec088589d755559'
OUTPUT = BASE/'go2_causal_executed_residual_diagnosis_v1_attempt_001'
PROTOCOL = 'docs/go2_causal_executed_residual_diagnosis_v1_2026-09-08.md'


def summary(rows):
    if not rows: return dict(count=0)
    return dict(count=len(rows),
        original_native_xy_error_mean_m=float(np.mean([r['targets'][0]['xy_error_m'] for r in rows])),
        original_native_xy_error_max_m=max(r['targets'][0]['xy_error_m'] for r in rows),
        corrected_native_xy_error_mean_m=float(np.mean([r['corrected_native_xy_error_m'] for r in rows])),
        corrected_native_xy_error_max_m=max(r['corrected_native_xy_error_m'] for r in rows),
        mean_original_native_xy_residual_m=np.mean([r['targets'][0]['xy_residual_predicted_minus_actual_m'] for r in rows], axis=0).tolist(),
        observed_native_displacement_error_mean_m=float(np.mean([r['observed_native_displacement_error_m'] for r in rows])),
        observed_native_displacement_error_max_m=max(r['observed_native_displacement_error_m'] for r in rows),
        corrected_error_smaller_steps=sum(r['corrected_native_xy_error_m'] < r['targets'][0]['xy_error_m'] for r in rows),
        corrected_error_larger_steps=sum(r['corrected_native_xy_error_m'] > r['targets'][0]['xy_error_m'] for r in rows))


def analyze(root, case):
    directory = root/case
    rows = read_json(directory, 'context_decisions.json'); tape = read_json(directory, 'command_tape.json')
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as z: poses = z['base_pose_world']
    records = []; public = []; missing_public = []
    for row in rows:
        tick = row['tick']; decision = row['decision']; s = decision['new_selection']
        if not s or 'prediction' not in s or tick >= len(tape): continue
        matched = []
        for i, action in enumerate(ACTIONS):
            targets = diagnose(s['prediction'][i], candidate_commands(action)[:8], tape, poses, tick=tick)
            if targets: matched.append((action, targets))
        if len(matched) != 1: raise ValueError('one actually executed candidate prefix required')
        action, targets = matched[0]
        next_evidence = rows[tick+1]['decision']['evidence'] if tick+1 < len(rows) else None
        if next_evidence is None:
            missing_public.append(dict(tick=tick, forecast_action=action, targets=targets)); continue
        now = 1_500_000_000+tick*100_000_000
        p, R, pose = current_joint_pose(decision['evidence'], identity=(0, 0, 0), now_ns=now)
        q, _, next_pose = current_joint_pose(next_evidence, identity=(0, 0, 0), now_ns=now+100_000_000)
        if pose['frame'] != tick or next_pose['frame'] != tick+1:
            raise ValueError('consecutive current measured pose pair required')
        observed = (R.T@(q-p))[:2]
        public.append(dict(tick=tick, available_tick=tick+1,
            predicted_body_xy_m=targets[0]['predicted_body_xy_m'], observed_body_xy_m=observed.tolist()))
        records.append(dict(tick=tick, forecast_action=action, selected_action=s['action'],
            terminal=decision['terminal'], exact_final_goal=s['intermediate_target_is_mission_goal'],
            targets=targets, observed_body_xy_m=observed.tolist(),
            observed_native_displacement_error_m=float(np.linalg.norm(observed-targets[0]['actual_body_xy_m']))))
    # No native fields cross into the causal correction helper.
    corrections = replay_bias(public)
    for r, c in zip(records, corrections, strict=True):
        assert r['tick'] == c['tick']
        r.update(causal_diagnostic=c, corrected_native_xy_error_m=float(np.linalg.norm(
            np.asarray(c['corrected_body_xy_m'])-r['targets'][0]['actual_body_xy_m'])))
    return dict(case=case, input_root=str(root), window_ticks=WINDOW_TICKS,
        all_steps=summary(records), exact_final_goal_steps=summary([r for r in records if r['exact_final_goal']]),
        other_steps=summary([r for r in records if not r['exact_final_goal']]),
        actions={a: summary([r for r in records if r['forecast_action'] == a]) for a in ACTIONS},
        final_goal_actions={a: summary([r for r in records if r['forecast_action'] == a and r['exact_final_goal']]) for a in ACTIONS},
        records=records, native_prefixes_without_public_endpoint=missing_public,
        unexecuted_outcomes_inferred=False, runtime_correction_adopted=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive causal diagnostic required')
    bound = []
    for root, sha, status in ((PREVIOUS, PREVIOUS_SHA, 'EXACT_MISSION_TARGET_GOAL_PROBE_COMPLETE'),
            (INPUT, INPUT_SHA, 'EXECUTED_HORIZON_FINAL_GOAL_PROBE_COMPLETE'),
            (READOUT, READOUT_SHA, 'EXECUTED_HORIZON_FINAL_GOAL_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); r = read_json(root, 'result.json')
        assert r['status'] == status
        ids = {'result.json': sha, **r.get('artifact_sha256', {})}
        if 'launch_sha256' in r: ids['launch.json'] = r['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, r))
    assert bound[2][2]['probe_result_sha256'] == INPUT_SHA
    assert all(r['all_measurement_gates_pass'] and r['cases'] == [list(c) for c in CASES] for _, _, r in bound[:2])
    old = read_json(READOUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_causal_executed_residual_v1.py',
        'lewm/tests/test_causal_executed_residual_diagnosis_development.py',
        'lewm/tests/test_executed_prefix_motion_diagnosis_development.py',
        'docs/go2_executed_horizon_final_goal_probe_result_2026-09-08.md'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+256*1024**2:
        raise ValueError('bounded diagnostic resources unavailable')
    launch = old | dict(source_sha256=sources, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        diagnostic_input_bindings={str(p): ids for p, ids, _ in bound},
        native_execution=False, model_training=False, workers=1, numerical_threads=1,
        causal_window_ticks=WINDOW_TICKS, runtime_correction_adopted=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('CAUSAL_EXECUTED_RESIDUAL_DIAGNOSIS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        reports = []
        for root in (PREVIOUS, INPUT):
            for case in CASES:
                r = analyze(root, case[0]); reports.append(r)
                print('CAUSAL_RESIDUAL_CASE', root.name, case[0], r['all_steps'], r['exact_final_goal_steps'], flush=True)
        write_json(OUTPUT/'motion.json', dict(conditions=reports))
        verify(launch)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        bindings = {n: digest(OUTPUT/n) for n in ('launch.json', 'motion.json')}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='CAUSAL_EXECUTED_RESIDUAL_DIAGNOSIS_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, window_ticks=WINDOW_TICKS,
            summaries=[{k: v for k, v in r.items() if k not in ('records', 'native_prefixes_without_public_endpoint')}
                for r in reports], wall_s=time.perf_counter()-started, hardware_after=hardware(),
            native_execution=False, model_training=False, runtime_correction_adopted=False,
            navigation_qualified=False, goal_achieved=False))
        print('CAUSAL_EXECUTED_RESIDUAL_DIAGNOSIS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CAUSAL_RESIDUAL_DIAGNOSIS_FAILURE', reason=repr(error))); raise


if __name__ == '__main__': main()
