"""Read-only, receipt-bound joins for the fresh twelve-layout prediction study.

No discovery, fitting, physics, retry or implicit receipt creation. A caller must
freeze explicit launch/audit hashes before loading. A completed audit is evidence
of its declared raw checks, not hardware qualification or a new raw-audit replay.
"""
from copy import deepcopy
from dataclasses import dataclass

from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.terminal_event_coverage_development import population_coverage
from scripts.independent_rgb_body_batch_development import BATCHES, load_inventory, output_root, eligibility
from scripts.audit_go2_independent_rgb_body_collection_v1 import load_terminal_batch
from scripts.audit_go2_independent_layout_collection_v1 import prefix_comparisons
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.startup_raw_sensor_audit_development import read_json


AUDIT = 'rgb_body_layout_audit.json'
AUDIT_LAUNCH = 'rgb_body_layout_audit_launch.json'
PRODUCTS = ('all_departure_windows.json', 'all_departure_targets.json',
    'rgb_body_eligible_windows.json', 'rgb_body_eligible_targets.json',
    'rgb_body_episode_roles.json', 'rgb_body_prefix_witnesses.json')
INPUT_FIELDS = ['known_action_blocks', 'known_action_valid', 'observation_history']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _join_audit(inventory, batch, audit, read):
    """Check already authenticated metadata; this helper alone verifies no bytes."""
    ids = list(inventory.episode_ids(batch))
    role = inventory.episodes[ids[0]]['role']
    require(audit['status'] == 'RGB_BODY_LAYOUT_AVAILABLE_EVIDENCE_AUDITED'
        and audit['batch'] == batch and audit['role'] == role
        and audit['collection_complete'] is True, 'completed exact batch audit required')
    require(len(ids) == 120 and all(audit[k] == 120 for k in
        ('expected_trials', 'committed_trials', 'audited_trials'))
        and list(audit['conditions']) == ids, 'complete planned and audited denominator required')
    require(audit['independent_layouts'] == 1 and all(audit[k] is False for k in
        ('model_trained', 'final_evaluation', 'navigation_qualified', 'goal_achieved')),
        'prediction-data audit cannot grant broader claims')
    expected_outputs = set(PRODUCTS) | {AUDIT_LAUNCH} | {c + '_rgb_body_evaluation.json' for c in ids}
    require(set(audit['output_sha256']) == expected_outputs, 'exact completed audit output roster required')
    windows, targets, eligible_windows, eligible_targets = [], [], [], []
    prefixes, coverages, reports = {}, {}, {}
    for c in ids:
        row = read(c + '_rgb_body_evaluation.json')
        report = row['report']; e = inventory.episodes[c]
        require(report['trial'] == c and report['layout_id'] == e['layout_id']
            and report['data_role'] == e['role'] and all(report[k] == e[k] for k in
                ('context_kind', 'history_kind', 'support'))
            and report['recorded_sensor_reconstruction_pass'] is True, 'exact audited episode identity required')
        recomputed = eligibility(row['coverage'], row['window'], row['footprint_diagnostics'])
        require(row['eligibility'] == recomputed, 'modality eligibility disagrees with audited evidence')
        require(not recomputed['hard_measurement_failed_frames'], 'hard measurement failure blocks study data')
        require(audit['conditions'][c] == report | dict(status='RAW_RGB_BODY_EPISODE_AUDITED', eligibility=recomputed),
            'summary must exactly match complete episode evidence')
        prefixes[c] = row['prefix']; coverages[c] = row['coverage']; reports[c] = report
        w, t = row['window'], row['targets']
        require((w is None) == (t is None), 'window/target presence must agree')
        if w is not None:
            require(w['condition'] == t['condition'] == c, 'episode cannot substitute another departure')
            windows.append(w); targets.append(t)
            if recomputed['rgb_body_prediction_eligible']:
                eligible_windows.append(w); eligible_targets.append(t)
    roles = {w['condition']: {k: inventory.episodes[w['condition']][k] for k in ('layout_id', 'role')}
        for w in eligible_windows}
    expected = (windows, targets, eligible_windows, eligible_targets, roles, prefixes)
    for name, value in zip(PRODUCTS, expected, strict=True):
        require(read(name) == value, 'derived product differs from audited full population: ' + name)
    # Decode all departures, including excluded histories, so corrupted labels
    # cannot disappear merely because a sample is ineligible for prediction.
    all_roles = {w['condition']: {k: inventory.episodes[w['condition']][k] for k in ('layout_id', 'role')}
        for w in windows}
    if windows:
        IndependentPulseEvaluation(inventory, PulseTimedDataset(windows, targets, all_roles))
    dataset = PulseTimedDataset(eligible_windows, eligible_targets, roles) if eligible_windows else None
    if dataset is not None:
        IndependentPulseEvaluation(inventory, dataset)
    pairs = prefix_comparisons(inventory, batch, prefixes)
    require(audit['prefix_comparisons'] == pairs and audit['exact_nonreference_prefix_matches'] ==
        sum(p['matched'] for p in pairs.values() if not p['is_reference']), 'full action-prefix accounting mismatch')
    require(audit['population'] == population_coverage(ids, coverages), 'full population classification mismatch')
    require(audit['departures'] == len(windows) and audit['eligible_departures'] == len(eligible_windows)
        and audit['action_coverage'] == (dataset.coverage(role) if dataset else {}), 'departure/action coverage mismatch')
    for field, report_key in (('setup_admitted', 'setup_admitted'), ('schedule_completions', 'schedule_complete'),
        ('contact_positive_targets', 'target_contact_positive')):
        require(audit[field] == sum(r[report_key] for r in reports.values()), 'outcome count mismatch: ' + field)
    require(audit['hard_measurement_failed_trials'] == [] and audit['strict_visibility_failed_trials'] ==
        [c for c, r in reports.items() if r.get('physical_visibility_pass') is False], 'measurement failure accounting mismatch')
    integrated = []
    if dataset is not None:
        for w, t in zip(dataset.windows, dataset._targets, strict=True):
            integrated.append(dict(condition=w['condition'], input_fields=INPUT_FIELDS,
                motion_valid=int(t['motion_valid'].sum()),
                contact_positive=int(((t['contact'] == 1) & t['contact_valid']).sum())))
    require(audit['materialized_samples'] == integrated, 'audited tensor materialization count/schema mismatch')
    return dict(batch=batch, role=role, windows=eligible_windows, targets=eligible_targets, roles=roles,
        prefixes=prefixes, population=audit['population'], planned_trials=ids,
        excluded_trials=[c for c in ids if c not in roles],
        strict_visibility_failed_trials=audit['strict_visibility_failed_trials'])


def load_batch(batch, receipt, inventory):
    """Authenticate one terminal batch; partial batches never become study inputs."""
    require(batch in BATCHES, 'declared fresh batch required')
    require(isinstance(receipt, dict) and set(receipt) == {'launch.json', AUDIT},
        'explicit frozen launch and terminal audit SHA-256 receipt required')
    output = output_root(batch)
    require(not any((output / n).exists() for n in ('failure.json', 'rgb_body_layout_audit_failure.json')),
        'failed collection/audit cannot enter this completed-study loader')
    # Verifies paths and hashes before parsing either supplied document.
    verify_artifacts(output, receipt)
    audit = read_json(output, AUDIT)
    expected = set(PRODUCTS) | {AUDIT_LAUNCH} | {c + '_rgb_body_evaluation.json' for c in inventory.episode_ids(batch)}
    require(set(audit['output_sha256']) == expected, 'exact audit roster required before artifact access')
    verify_artifacts(output, audit['output_sha256'])
    launch, terminal, committed, inputs = load_terminal_batch(output, inventory, batch)
    require(terminal['status'] == 'RGB_BODY_LAYOUT_COLLECTION_COMPLETE', 'complete collection required')
    witness = read_json(output, AUDIT_LAUNCH)
    require(witness == dict(batch=batch, source_sha256=launch['source_sha256'], artifact_sha256=inputs,
        complete_collection=True, model_training=False), 'audit must bind exact collection sources and raw inputs')
    for c in inventory.episode_ids(batch):
        require(c in committed and not committed[c]['absent_expected_artifacts']
            and read_json(output, c + '_rgb_body_evaluation.json') == committed[c]['raw_precheck'],
            'terminal evaluation must match bound raw precheck for every planned case')
    joined = _join_audit(inventory, batch, audit, lambda n: read_json(output, n))
    verify_ordered_launch(launch)
    verify_artifacts(output, inputs | audit['output_sha256'] | receipt)
    return joined | dict(receipt=deepcopy(receipt), output_root=str(output),
        artifact_sha256=inputs | audit['output_sha256'] | receipt,
        source_and_artifact_bindings_verified=True, raw_audit_reexecuted=False)


@dataclass
class StudyData:
    evaluation: IndependentPulseEvaluation
    prefixes: dict
    batches: dict
    receipts: dict

    def report(self):
        return dict(planned_layouts=12, planned_episodes=1440,
            roles={r: self.evaluation.population(r) for r in ('train', 'selection', 'development_eval')},
            batches={b: {k: row[k] for k in ('role', 'population', 'excluded_trials',
                'strict_visibility_failed_trials')} for b, row in self.batches.items()},
            receipts=deepcopy(self.receipts), all_batch_receipts_verified=True,
            raw_audit_reexecuted=False, model_training=False, final_evaluation=False,
            navigation_qualified=False, hardware_qualified=False, goal_achieved=False)


def load_study(receipts):
    """All twelve exact receipts required, including zero-eligible layouts.

    Exclusions remain explicit; no replacement, case selection or train-role
    reassignment. This constructs data, not permission to silently train on an
    incomplete role/action population. The prospective study must specify its gate.
    """
    require(isinstance(receipts, dict) and set(receipts) == set(BATCHES),
        'all twelve fixed batch receipts required; no partial study or selected layouts')
    inventory = load_inventory(); batches = {}; windows = []; targets = []; roles = {}; prefixes = {}
    for batch in BATCHES:
        row = load_batch(batch, receipts[batch], inventory); batches[batch] = row
        windows.extend(row['windows']); targets.extend(row['targets']); roles.update(row['roles']); prefixes.update(row['prefixes'])
    require(bool(windows), 'no eligible study samples; retain terminal audits, do not manufacture a dataset')
    evaluation = IndependentPulseEvaluation(inventory, PulseTimedDataset(windows, targets, roles))
    # No raw packets or image tensors are kept in this metadata join.
    for batch, row in batches.items():
        verify_artifacts(output_root(batch), row['artifact_sha256'])
    return StudyData(evaluation, prefixes, batches, deepcopy(receipts))
