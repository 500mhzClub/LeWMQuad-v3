"""All offsets of the existing training trials; no model or native execution."""
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
from lewm.all_phase_training_targets_development import expand_trial
from scripts.navigation_artifact_root_development import (
    BASE, validate_root, create_output, verify_artifacts)
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INPUT = BASE/'go2_observation_horizon_family_targets_v1_attempt_001'
OUTPUT = BASE/'go2_all_phase_training_targets_v1_attempt_001'
SOURCE = 'scripts/derive_go2_all_phase_training_targets_v1.py'
TEST = 'lewm/tests/test_all_phase_training_targets_development.py'
PROTOCOL = 'docs/go2_all_phase_training_targets_v1_2026-09-10.md'
INPUT_SHA = 'fe3ab252e6da0ebadba13927c0dad7410d2084145c3b50439f6848ea5b65a775'
WINDOWS_SHA = 'c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811'
ROOTS = dict(family=BASE/'go2_geometry_progress_family_v1_attempt_001',
    switch=BASE/'go2_moving_action_switch_family_v1_attempt_001')
RESERVE = 40*1024**3
ALLOWANCE = 64*1024**2


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive derivation; no retry')
    resources = hardware()
    if (resources['memory_available_bytes'] < 40*1024**3
            or resources['artifact_free_bytes'] < RESERVE+ALLOWANCE):
        raise ValueError('8GiB derivation plus32GiB concurrent native and storage reserve required')
    verify_artifacts(INPUT, {'result.json':INPUT_SHA, 'windows.json':WINDOWS_SHA})
    original_result = read_json(INPUT, 'result.json')
    if (original_result['status'] != 'OBSERVATION_HORIZON_FAMILY_TARGETS_COMPLETE'
            or original_result['context_slots'] != 912 or original_result['available_contexts'] != 828
            or original_result['original_context_availability_and_roles_preserved'] is not True):
        raise ValueError('complete unchanged original target population required')
    input_ids = original_result['artifact_sha256'] | {'result.json':INPUT_SHA}
    verify_artifacts(INPUT, input_ids)
    original_launch = read_json(INPUT, 'launch.json')
    verify(original_launch)
    if original_result['source_sha256'] != original_launch['source_sha256']:
        raise ValueError('same original source bindings required')
    groups = defaultdict(list)
    for row in read_json(INPUT, 'windows.json'):
        if row['data_role'] == 'train': groups[row['source'], row['trial']].append(row)
    if (len(groups) != 120 or Counter(s for s,t in groups) != {'family':48, 'switch':72}
            or sum(len(v) for v in groups.values()) != 456
            or sum(r['available'] for v in groups.values() for r in v) != 408):
        raise ValueError('complete original training assignment required')
    consumed = {str(root):{} for root in ROOTS.values()}
    for source,trial in groups:
        root = ROOTS[source]
        old_ids = original_launch['source_collection_artifact_sha256'][str(root)]
        for name in ('physics_trace.npz', 'camera_audit.json'):
            relative = trial+'/'+name
            consumed[str(root)][relative] = old_ids[relative]
    for root,ids in consumed.items(): verify_artifacts(Path(root), ids)
    sources = discover_sources((SOURCE, TEST, PROTOCOL), original_result['source_sha256'])
    launch = original_launch | dict(source_sha256=sources, protocol=PROTOCOL, output_root=str(OUTPUT),
        original_target_result_sha256=INPUT_SHA, original_target_artifact_sha256=input_ids,
        consumed_training_artifact_sha256=consumed, hardware=resources,
        memory_allowance_bytes=8*1024**3, concurrent_native_allowance_bytes=32*1024**3,
        minimum_free_bytes=RESERVE, output_allowance_bytes=ALLOWANCE,
        cpu_workers=1, native_scene_workers=0, model_training=False,
        source_role='train', offsets=list(range(40)), target_horizon_ticks=8,
        native_labels_are_target_only=True, future_rgb_materialized=False,
        previous_raw_collection_audits_reexecuted=False, sealed_material_accessed=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    started = time.perf_counter(); print('ALL_PHASE_TRAINING_TARGETS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        rows = []; summary = defaultdict(Counter); transitions = Counter()
        for source,trial in sorted(groups):
            root = ROOTS[source]
            ids = {trial+'/'+n:consumed[str(root)][trial+'/'+n]
                for n in ('physics_trace.npz', 'camera_audit.json')}
            verify_artifacts(root, ids)
            with np.load(root/trial/'physics_trace.npz', allow_pickle=False) as a:
                raw = {k:a[k] for k in ('timestamp_s','base_pose_world','physics_contact','requested_command')}
            cameras = read_json(root/trial, 'camera_audit.json')
            expanded = expand_trial(groups[source,trial], raw, cameras)
            for row in expanded:
                phase = row['control_phase_modulo_five']; s = summary[source,phase]
                s['slots'] += 1; s['available'] += row['available']
                if not row['available']: s[row['reason']] += 1; continue
                targets = row['targets']; s['first_motion_valid'] += targets[0]['motion_valid']
                s['eight_motion_valid'] += all(t['motion_valid'] for t in targets)
                command = row['known_commands'][0]; zero = command == [0.,0.,0.]
                s['first_command_zero'] += zero
                frame = row['observation_horizon_receipt']['departure_tick']
                previous = raw['requested_command'][749+50*frame]
                changed = not np.array_equal(previous, command)
                transitions[source,phase,'any_switch'] += int(changed)
                if zero and changed: transitions[source,phase,'zero_after_nonzero'] += 1
                if zero: s['zero_first_motion_valid'] += targets[0]['motion_valid']
            rows.extend(expanded); verify_artifacts(root, ids)
        counts = dict(training_trials=len(groups), context_slots=len(rows),
            available_contexts=sum(r['available'] for r in rows),
            original_contexts_exact=sum(r['original_sample_id'] is not None for r in rows),
            first_motion_valid=sum(s['first_motion_valid'] for s in summary.values()),
            eight_motion_valid=sum(s['eight_motion_valid'] for s in summary.values()))
        if counts != dict(training_trials=120, context_slots=4800, available_contexts=4010,
                original_contexts_exact=456, first_motion_valid=3974, eight_motion_valid=3140):
            raise ValueError('all fixed training census denominators must reproduce')
        write_json(OUTPUT/'windows.json', rows)
        report = dict(**counts,
            phase_summary=[dict(source=k[0], phase=k[1], **v) for k,v in sorted(summary.items())],
            command_transitions=[dict(source=k[0],phase=k[1],kind=k[2],count=v)
                for k,v in sorted(transitions.items()) if v],
            command_switch_phase_diversity_added=False, independent_episodes_added=0,
            data_role_changed=False, native_labels_are_target_only=True,
            future_rgb_materialized=False, inputs_materialized=False, model_training=False,
            inference_or_native_execution=False, navigation_qualified=False)
        write_json(OUTPUT/'coverage.json', report)
        if sum(p.stat().st_size for p in (OUTPUT/'launch.json', OUTPUT/'windows.json', OUTPUT/'coverage.json')) > ALLOWANCE-1024**2:
            raise ValueError('metadata allowance exceeded; evidence retained')
        verify(launch); verify_artifacts(INPUT, input_ids)
        for root,ids in consumed.items(): verify_artifacts(Path(root), ids)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','windows.json','coverage.json')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='ALL_PHASE_TRAINING_TARGETS_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report,
            original_target_result_sha256=INPUT_SHA, wall_s=time.perf_counter()-started,
            hardware_after=hardware(), model_training=False, native_execution=False,
            navigation_qualified=False, goal_achieved=False))
        print('ALL_PHASE_TRAINING_TARGETS_COMPLETE', digest(OUTPUT/'result.json'), counts, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_ALL_PHASE_TRAINING_TARGET_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
