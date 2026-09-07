"""Read only the completed study's six training-layout label products.

Post-hoc design diagnostic, not new training, raw auditing, a utility benchmark,
or execution authority. No image/checkpoint loading or runtime discovery.
The complete study was authenticated separately; this command reauthenticates
the exact terminal and the narrow metadata chain it consumes, not all raw bytes.
"""
import hashlib
import json
from collections import defaultdict

import numpy as np

from lewm.coupled_pulse_rollout_development import COMMANDS
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.matched_action_hazard_evaluation_development import HORIZON_NS, evaluate_matched_hazards
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.pulse_training_interaction_diagnostic_development import summarize
from scripts.independent_rgb_body_batch_development import load_inventory, output_root
from scripts.navigation_artifact_root_development import BASE, artifact_path, verify_artifacts

RESULT_SHA256 = '588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122'
DEFINITION_SHA256 = '8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8'
OUTPUT = BASE / 'go2_independent_pulse_parallel_study_v1_attempt_001'
PRODUCTS = ('rgb_body_eligible_windows.json', 'rgb_body_eligible_targets.json',
            'rgb_body_episode_roles.json', 'rgb_body_prefix_witnesses.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(root, name):
    p = artifact_path(root, name)
    require(p.stat().st_size <= 64 * 1024**2, 'bounded exact metadata file required')
    return json.loads(p.read_text())


def diagnostic():
    bindings = [(OUTPUT, {'result.json': RESULT_SHA256})]
    verify_artifacts(*bindings[0])
    terminal = read(OUTPUT, 'result.json')
    require(terminal['status'] == 'MATCHED_DEVELOPMENT_COMPARISON_COMPLETE'
        and terminal['fits'] == 36 and terminal['optimizer_updates'] == 43200
        and terminal['goal_achieved'] is False, 'complete unpromoted study required')
    names = ('launch.json', 'dataset.json', 'coverage.json')
    bindings.append((OUTPUT, {n: terminal['output_sha256'][n] for n in names}))
    verify_artifacts(*bindings[-1])
    launch, dataset, coverage = (read(OUTPUT, n) for n in names)
    require(launch['definition_sha256'] == DEFINITION_SHA256, 'exact completed definition required')
    inventory = load_inventory()
    batches = [b for b in inventory.batches
               if inventory.episodes[inventory.episode_ids(b)[0]]['role'] == 'train']
    require(batches == [f'l{i:02d}' for i in range(6)], 'all six training layouts required')
    windows, targets, roles, prefixes = [], [], {}, {}
    for batch in batches:
        root = output_root(batch)
        receipt = dataset['receipts'][batch]
        require(set(receipt) == {'launch.json', 'rgb_body_layout_audit.json'}, 'exact batch receipt required')
        bindings.append((root, receipt)); verify_artifacts(*bindings[-1])
        audit = read(root, 'rgb_body_layout_audit.json')
        require(audit['status'] == 'RGB_BODY_LAYOUT_AVAILABLE_EVIDENCE_AUDITED'
            and audit['batch'] == batch and audit['role'] == 'train'
            and audit['collection_complete'] is True, 'completed training audit required')
        bindings.append((root, {n: audit['output_sha256'][n] for n in PRODUCTS}))
        verify_artifacts(*bindings[-1])
        w, t, r, p = (read(root, n) for n in PRODUCTS)
        expected = set(inventory.episode_ids(batch))
        require(len(w) == len(t) == 120 and set(r) == set(p) == expected
            and {x['condition'] for x in w} == {x['condition'] for x in t} == expected,
            'complete original training batch, not selected episodes')
        require(all(v['role'] == 'train' for v in r.values()), 'no nontraining labels allowed')
        windows.extend(w); targets.extend(t); roles.update(r); prefixes.update(p)
    view = IndependentPulseEvaluation(inventory, PulseTimedDataset(windows, targets, roles))
    matched = evaluate_matched_hazards(view, prefixes, {}, role='train')
    require(matched == coverage['roles']['train']['matched_contact_coverage'],
            'reconstructed training contact coverage must equal original complete report')
    data = view.arrays('train')
    lookup = {m['condition']: i for i, m in enumerate(data['metadata'])}
    groups, motions = [], []
    for g in matched['groups']:
        truth = []
        for action, condition in enumerate(g['conditions']):
            i = lookup[condition]
            slots = np.flatnonzero(data['active'][i] & (data['offsets_ns'][i] == HORIZON_NS))
            require(len(slots) == 1, 'exact shared two-second horizon required')
            h = int(slots[0]); valid = data['targets']['contact_valid'][i, h]
            truth.append(float(data['targets']['contact'][i, h]) if valid else None)
            if data['targets']['motion_valid'][i, h]:
                motions.append({k: g[k] for k in ('layout_id', 'context_kind', 'history_kind', 'support')} |
                    dict(action=action, motion=data['targets']['motion'][i, h].tolist()))
        groups.append({k: g[k] for k in ('layout_id', 'context_kind', 'history_kind', 'support')} |
            dict(group_id=g['conditions'][0].rsplit('_a', 1)[0], role='train', contact=truth,
                 matched_prefix=not g['unavailable_prefix_actions'] and not g['unequal_prefix_actions']))
    result = summarize(groups, planned_group_ids=[g['group_id'] for g in groups])
    require(result['planned_groups'] == 120 and result['complete_population'], 'complete120-group training diagnostic required')
    action_rows = []
    for action in range(6):
        selected = np.asarray([r['motion'] for r in motions if r['action'] == action])
        action_rows.append(dict(action_index=action, requested_command=list(COMMANDS[action//2]),
            pulse_ticks=(2, 5)[action % 2], planned_rows=120, observed_motion_rows=len(selected),
            censored_motion_rows=120-len(selected),
            mean_body_frame_xy_m=selected[:, :2].mean(0).tolist() if len(selected) else None,
            body_frame_x_range_m=[float(selected[:, 0].min()), float(selected[:, 0].max())] if len(selected) else None,
            yaw_range_rad=[float(selected[:, 2].min()), float(selected[:, 2].max())] if len(selected) else None))
    cells = defaultdict(list)
    for r in motions:
        cells[(r['action'], r['context_kind'], r['history_kind'], r['support'])].append(r)
    dispersion = []
    planned_cells = {(a, g['context_kind'], g['history_kind'], g['support'])
                     for g in groups for a in range(6)}
    for key in sorted(planned_cells):
        rows = cells[key]
        xy = np.asarray([r['motion'][:2] for r in rows])
        dispersion.append(dict(action_index=key[0], context_kind=key[1], history_kind=key[2], support=key[3],
            observed_layouts=len(rows), planned_layouts=6,
            observed_xy_rms_about_cell_mean_m=(float(np.sqrt(np.mean(np.sum((xy-xy.mean(0))**2, axis=1))))
                                              if len(rows) else None),
            interpretation='descriptive observed-label dispersion, not prediction error or isolated geometry effect'))
    for root, named in bindings:
        verify_artifacts(root, named)
    result.update(horizon_ns=HORIZON_NS, action_motion=action_rows, cross_layout_motion_dispersion=dispersion,
        groups=groups, study_result_sha256=RESULT_SHA256,
        input_sha256={str(root): {n: h for r, named in bindings if r == root for n, h in named.items()}
                      for root, _ in bindings},
        source_sha256={n: hashlib.sha256(open(n, 'rb').read()).hexdigest() for n in
            ('scripts/read_go2_independent_pulse_training_interaction_v1.py',
             'lewm/pulse_training_interaction_diagnostic_development.py')},
        scope='post-hoc training-label diagnostic; narrow metadata authentication; no raw audit, RGB loading, fitting or progress metric')
    return result


if __name__ == '__main__':
    print(json.dumps(diagnostic(), sort_keys=True, allow_nan=False))
