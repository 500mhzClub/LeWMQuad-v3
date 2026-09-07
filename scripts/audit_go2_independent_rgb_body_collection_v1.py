"""Terminal-only raw audit, complete denominator and modality-scoped dataset."""
import argparse
import json
import shutil
import cv2
import torch
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.terminal_event_coverage_development import population_coverage
from scripts.independent_rgb_body_batch_development import (BATCHES, PROTOCOL, BATCH_BUDGET,
    RESERVE, load_inventory, output_root, validate_launch, episode_artifacts)
from scripts.independent_rgb_body_audit_development import audit_rgb_body_condition
from scripts.audit_go2_independent_layout_collection_v1 import prefix_comparisons
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.navigation_artifact_root_development import BASE, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json


def load_terminal_batch(output, inventory, batch):
    launch = read_json(output, 'launch.json'); verify_ordered_launch(launch); validate_launch(launch, inventory, batch)
    names = [n for n in ('result.json', 'failure.json') if (output / n).is_file()]
    if len(names) != 1: raise ValueError('exactly one terminal record required; never audit a live batch')
    name = names[0]; terminal = read_json(output, name); ids = inventory.episode_ids(batch)
    assert terminal['batch'] == batch and terminal['planned_trials'] == list(ids)
    count, checked = len(terminal['commits']), len(terminal['prechecks'])
    assert 0 <= count <= len(ids) and max(0, count - 1) <= checked <= count
    assert list(terminal['conditions']) == list(ids[:count])
    assert list(terminal['commits']) == [f'episode_{i:03d}_commit.json' for i in range(count)]
    assert list(terminal['prechecks']) == [f'episode_{i:03d}_raw_precheck.json' for i in range(checked)]
    if name == 'result.json':
        assert terminal['status'] == 'RGB_BODY_LAYOUT_COLLECTION_COMPLETE' and count == checked == len(ids)
        assert terminal['uncommitted_trial'] is None
    else:
        assert terminal['status'] == 'TERMINAL_RGB_BODY_LAYOUT_COLLECTION_FAILURE'
        if terminal['uncommitted_trial'] is not None:
            assert count < len(ids) and terminal['uncommitted_trial'] == ids[count]
    bindings = {p: digest(output / p) for p in ('launch.json', name)} | terminal['commits'] | terminal['prechecks']
    verify_artifacts(output, bindings); committed = {}; total = (output / 'launch.json').stat().st_size
    for i, trial in enumerate(ids[:count]):
        leaf = f'episode_{i:03d}_commit.json'; commit = read_json(output, leaf)
        assert commit['trial'] == trial and commit['result'] == terminal['conditions'][trial]
        expected = {trial + '/' + p for p in episode_artifacts(inventory.specification(trial), commit['result'])}
        present, missing = set(commit['artifact_sha256']), set(commit['absent_expected_artifacts'])
        assert not present & missing and present | missing == expected and len(missing) == len(commit['absent_expected_artifacts'])
        verify_artifacts(output, commit['artifact_sha256'])
        assert commit['artifact_bytes'] == sum((output / p).stat().st_size for p in present)
        total += commit['artifact_bytes'] + (output / leaf).stat().st_size
        if i < checked:
            precheck = f'episode_{i:03d}_raw_precheck.json'; total += (output / precheck).stat().st_size
            commit['raw_precheck'] = read_json(output, precheck)
        bindings.update(commit['artifact_sha256']); committed[trial] = commit
    assert terminal['committed_bytes'] == total
    return launch, terminal, committed, bindings


def run_audit(batch):
    output = output_root(batch); inventory = load_inventory()
    if (output / 'rgb_body_layout_audit_launch.json').exists(): raise ValueError('exclusive terminal audit; no retry/resume')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    launch, terminal, committed, bindings = load_terminal_batch(output, inventory, batch)
    outputs, reports, prefixes, coverages = {}, {}, {}, {}
    windows, targets, eligible_windows, eligible_targets, invalid, infrastructure = [], [], [], [], [], []
    used = terminal['committed_bytes'] + (output / ('result.json' if terminal['status'] == 'RGB_BODY_LAYOUT_COLLECTION_COMPLETE' else 'failure.json')).stat().st_size
    audit_bytes = 0
    def save(name, value):
        nonlocal used, audit_bytes
        encoded = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        if (used + len(encoded) > BATCH_BUDGET or audit_bytes + len(encoded) > 128 * 1024**2
                or shutil.disk_usage(BASE).free < RESERVE + len(encoded)):
            raise ValueError('bounded terminal audit metadata allowance')
        write_json(output / name, value); used += len(encoded); audit_bytes += len(encoded)
        outputs[name] = digest(output / name)
    save('rgb_body_layout_audit_launch.json', dict(batch=batch, source_sha256=launch['source_sha256'],
        artifact_sha256=bindings, complete_collection=terminal['status'] == 'RGB_BODY_LAYOUT_COLLECTION_COMPLETE', model_training=False))
    try:
        for trial in inventory.episode_ids(batch):
            if trial not in committed:
                status = 'ATTEMPTED_UNCOMMITTED' if trial == terminal['uncommitted_trial'] else 'NOT_ATTEMPTED'
                if status == 'ATTEMPTED_UNCOMMITTED': infrastructure.append(trial)
                reports[trial] = dict(status=status, recorded_sensor_reconstruction_pass=False, departure_present=False)
                continue
            commit = committed[trial]
            if commit['absent_expected_artifacts']:
                invalid.append(trial)
                reports[trial] = dict(status='INCOMPLETE_ARTIFACTS_NOT_AUDITED', missing_artifacts=commit['absent_expected_artifacts'],
                    recorded_sensor_reconstruction_pass=False, departure_present=False)
                continue
            row = audit_rgb_body_condition(output / trial, inventory.specification(trial), commit['result'],
                launch['source_sha256'][PROTOCOL], batch=batch)
            if 'raw_precheck' in commit: assert json.loads(json.dumps(row)) == commit['raw_precheck']
            reports[trial] = row['report'] | dict(status='RAW_RGB_BODY_EPISODE_AUDITED', eligibility=row['eligibility'])
            prefixes[trial] = row['prefix']; coverages[trial] = row['coverage']
            if row['window'] is not None:
                windows.append(row['window']); targets.append(row['targets'])
                if row['eligibility']['rgb_body_prediction_eligible']:
                    eligible_windows.append(row['window']); eligible_targets.append(row['targets'])
            save(trial + '_rgb_body_evaluation.json', row)
            print('RGB_BODY_AUDIT', batch, trial, row['coverage']['classification'], row['eligibility'], flush=True)
        pairs = prefix_comparisons(inventory, batch, prefixes)
        roles = {w['condition']: dict(layout_id=inventory.episodes[w['condition']]['layout_id'],
            role=inventory.episodes[w['condition']]['role']) for w in eligible_windows}
        dataset = PulseTimedDataset(eligible_windows, eligible_targets, roles) if eligible_windows else None
        integrated = []
        if dataset is not None:
            for i, window in enumerate(eligible_windows):
                sample = dataset.sample(i, {window['condition']: IntentReturnRGBDReplay(output / window['condition'])})
                integrated.append(dict(condition=window['condition'], input_fields=sorted(sample['inputs']),
                    motion_valid=int(sample['targets']['motion_valid'].sum()),
                    contact_positive=int(((sample['targets']['contact'] == 1) & sample['targets']['contact_valid']).sum())))
        for name, value in (('all_departure_windows.json', windows), ('all_departure_targets.json', targets),
            ('rgb_body_eligible_windows.json', eligible_windows), ('rgb_body_eligible_targets.json', eligible_targets),
            ('rgb_body_episode_roles.json', roles), ('rgb_body_prefix_witnesses.json', prefixes)):
            save(name, value)
        population = population_coverage(inventory.episode_ids(batch), coverages, invalid=invalid, infrastructure=infrastructure)
        audited = [r for r in reports.values() if r['recorded_sensor_reconstruction_pass']]
        verify_ordered_launch(launch); verify_artifacts(output, bindings | outputs)
        result = dict(status='RGB_BODY_LAYOUT_AVAILABLE_EVIDENCE_AUDITED', batch=batch,
            collection_complete=terminal['status'] == 'RGB_BODY_LAYOUT_COLLECTION_COMPLETE', expected_trials=120,
            committed_trials=len(committed), audited_trials=len(audited), conditions=reports, population=population,
            prefix_comparisons=pairs, departures=len(windows), eligible_departures=len(eligible_windows),
            setup_admitted=sum(r['setup_admitted'] for r in audited), schedule_completions=sum(r['schedule_complete'] for r in audited),
            materialized_samples=integrated, role=launch['role'], action_coverage=dataset.coverage(launch['role']) if dataset else {},
            exact_nonreference_prefix_matches=sum(p['matched'] for p in pairs.values() if not p['is_reference']),
            contact_positive_targets=sum(r['target_contact_positive'] for r in audited),
            strict_visibility_failed_trials=[c for c, r in reports.items() if r.get('physical_visibility_pass') is False],
            hard_measurement_failed_trials=[c for c, r in reports.items() if r.get('eligibility', {}).get('hard_measurement_failed_frames')],
            output_sha256=dict(outputs), independent_layouts=1, model_trained=False,
            final_evaluation=False, navigation_qualified=False, goal_achieved=False)
        save('rgb_body_layout_audit.json', result)
        print('RGB_BODY_LAYOUT_AUDIT_COMPLETE', batch, len(audited), len(eligible_windows), flush=True)
    except Exception as error:
        write_json(output / 'rgb_body_layout_audit_failure.json', dict(status='TERMINAL_RGB_BODY_LAYOUT_AUDIT_FAILURE',
            reason=repr(error), completed_reports=list(reports), planned_trials=list(inventory.episode_ids(batch)),
            output_sha256=dict(outputs), goal_achieved=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=BATCHES, required=True)
    run_audit(parser.parse_args().batch)


if __name__ == '__main__': main()
