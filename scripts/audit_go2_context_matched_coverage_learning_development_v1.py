#!/usr/bin/env python3
"""Audit the completed18 checkpoints on the same frozen development bytes.

Replays inference, not optimization. Scalar primary-score and action-choice
checks are independent of the vectorized reducers used by the training runner.
"""
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_go2_context_matched_coverage_learning_development_v1 import (
    OUTPUT, SEEDS, DATA_CONDITIONS, CONDITIONS, UPDATES, digest, write_json,
    source_bindings, input_bindings, verify, materialize, matched_schedule,
    take, simple_predictions, coverage_report, moving_decisions, predictions,
    comparisons, TemporalRGBBodyJEPA, active_parameters, state_identity)

LAUNCH_SHA = '1ddaaee56a42b78b1b8fb0425e15dabad7609b6fdc243820e6a2506fcd154ddf'
NEW_SOURCES = (str(Path(__file__).relative_to(ROOT)),
               'lewm/tests/test_context_matched_coverage_raw_audit.py')


def check(value, message):
    if not value:
        raise ValueError(message)


def equal_scalar(a, b, message):
    check(a is None and b is None or a is not None and b is not None
          and math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-12, message)


def check_primary(prediction, batch, report):
    """Independent per-layout switched first-half-second endpoints and masks."""
    expected = []
    for layout in sorted({m['layout_id'] for m in batch['metadata']}):
        motion, contact = [], []
        for i, row in enumerate(batch['metadata']):
            if row['layout_id'] != layout or row['coverage_source'] != 'switch':
                continue
            check(row['offset_ns'] == 1_000_000_000 and row['prefix_action_index'] != row['action_index'],
                  'new suffix is actually a moving action change')
            check(bool(batch['known_action_valid'][i, 0].all()), 'first action horizon known')
            t = batch['targets']
            if bool(t['motion_valid'][i, 0]):
                motion.append(math.hypot(float(prediction[i, 0, 0]) - float(t['motion'][i, 0, 0]),
                                         float(prediction[i, 0, 1]) - float(t['motion'][i, 0, 1])))
            if bool(t['contact_valid'][i, 0]):
                probability = 1 / (1 + math.exp(-max(-60., min(60., float(prediction[i, 0, 4])))))
                contact.append((probability - float(t['contact'][i, 0])) ** 2)
        expected.append({'layout_id': layout, 'motion_count': len(motion), 'contact_count': len(contact),
                         'position_error_m': sum(motion) / len(motion) if motion else None,
                         'contact_brier': sum(contact) / len(contact) if contact else None})
    actual = report['moving_switch']['first_half_second']
    check([r['layout_id'] for r in actual['layouts']] == [r['layout_id'] for r in expected], 'primary layout identities')
    for a, b in zip(actual['layouts'], expected, strict=True):
        check(a['motion_count'] == b['motion_count'] and a['contact_count'] == b['contact_count'], 'primary valid counts')
        for key in ('position_error_m', 'contact_brier'):
            equal_scalar(a[key], b[key], 'independent primary layout endpoint')
    for key in ('position_error_m', 'contact_brier'):
        values = [r[key] for r in expected if r[key] is not None]
        equal_scalar(actual['layout_macro'][key], sum(values) / len(values) if values else None,
                     'independent primary macro endpoint')


def check_choices(prediction, batch, result, horizon):
    metadata, t = batch['metadata'], batch['targets']
    for row in result['rows']:
        indices = sorted([i for i, m in enumerate(metadata) if m['layout_id'] == row['layout_id']
                          and m['offset_ns'] == 1_000_000_000
                          and m.get('prefix_action_index', m['action_index']) == row['past_action']],
                         key=lambda i: metadata[i]['action_index'])
        check(len(indices) == 5 and [metadata[i]['action_index'] for i in indices] == list(range(5)), 'choice candidates')
        cue = {'forward': (.8, 0), 'left': (0, .8), 'right': (0, -.8)}[row['cue']]
        estimated, actual = [], []
        for i in indices:
            check(bool(t['contact_valid'][i, horizon]), 'choice contact horizon known')
            p = 1 / (1 + math.exp(-max(-60., min(60., float(prediction[i, horizon, 4])))))
            estimated.append(10 * p + math.hypot(float(prediction[i, horizon, 0]) - cue[0],
                                                  float(prediction[i, horizon, 1]) - cue[1]))
            if bool(t['contact'][i, horizon]):
                actual.append(10.)
            else:
                check(bool(t['motion_valid'][i, horizon]), 'actual noncontact endpoint known')
                actual.append(math.hypot(float(t['motion'][i, horizon, 0]) - cue[0],
                                         float(t['motion'][i, horizon, 1]) - cue[1]))
        chosen = min(range(5), key=estimated.__getitem__)
        check(chosen == row['chosen_action'], 'independent chosen action')
        check(bool(t['contact'][indices[chosen], horizon]) == row['contact'], 'independent chosen contact')
        equal_scalar(row['realized_cost'], actual[chosen], 'independent realized cost')
        equal_scalar(row['regret'], actual[chosen] - min(actual), 'independent regret')
        equal_scalar(row['always_stop_cost'], actual[0], 'independent stop baseline')


def check_updates(entries, condition):
    check(len(entries) == UPDATES, 'fixed complete update count')
    for update, row in enumerate(entries, 1):
        keys = {'update', 'loss', 'gradient_norm_before_clip', 'direct_outcome', 'variance', 'covariance'}
        if condition != 'direct':
            keys.add('rollout_outcome')
        allowed = keys | ({'latent_prediction'} if condition == 'jepa' else set())
        check(keys <= set(row) <= allowed and row['update'] == update, 'objective/update identity')
        check(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                  for value in row.values()), 'finite update log')
        total = row['direct_outcome'] + .1 * row['variance'] + .01 * row['covariance']
        total += row.get('rollout_outcome', 0) + row.get('latent_prediction', 0)
        check(abs(total - row['loss']) <= 1e-5 * max(1., abs(total)), 'logged weighted loss')
        check(row['gradient_norm_before_clip'] >= 0, 'nonnegative gradient norm')


def main():
    check(len(sys.argv) == 1, 'fixed full audit, no interim/retry overrides')
    audit_path = OUTPUT / 'raw_artifact_audit.json'
    check(not audit_path.exists() and digest(OUTPUT / 'launch.json') == LAUNCH_SHA, 'fresh exact-launch audit')
    launch = json.loads((OUTPUT / 'launch.json').read_text())
    result = json.loads((OUTPUT / 'result.json').read_text())
    check(result['status'] == 'COMPLETE' and len(result['models']) == 18, 'full terminal18-model study required')
    sources = launch['source_sha256'] | {name: digest(ROOT / name) for name in NEW_SOURCES}
    verify(sources)
    audited = []
    try:
        check(launch['source_sha256'] == source_bindings() and launch['input_sha256'] == input_bindings(), 'full source/input closure')
        check(launch['seeds'] == list(SEEDS) and launch['data_conditions'] == list(DATA_CONDITIONS)
              and launch['conditions'] == list(CONDITIONS) and launch['updates'] == UPDATES, 'fixed study configuration')
        check(result['launch_sha256'] == LAUNCH_SHA, 'terminal launch identity')
        artifacts = {'schedules.json', 'validation_order.json', 'baseline_result.json', 'baseline_predictions.npz'}
        check(set(result['artifact_sha256']) == artifacts, 'root artifact population')
        for name, sha in result['artifact_sha256'].items():
            check(digest(OUTPUT / name) == sha, 'root artifact identity')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        train_rows, train = materialize('train')
        _, validation = materialize('validation')
        expected = {str(seed): matched_schedule(train_rows, updates=UPDATES, seed=seed) for seed in SEEDS}
        check(json.loads((OUTPUT / 'schedules.json').read_text()) == {'train_order': train_rows, 'schedules': expected}, 'all exact matched schedules')
        check(json.loads((OUTPUT / 'validation_order.json').read_text()) == validation['metadata'], 'validation order')
        schedule_sha = digest(OUTPUT / 'schedules.json')
        check(result['schedule_sha256'] == schedule_sha, 'terminal schedule binding')
        for schedule in expected.values():
            for step in schedule:
                check(len(step['batch']) == 16 and len({r['layout_id'] for r in step['batch']}) == 16, 'one context per training layout')
                for pair in step['batch']:
                    limited = train_rows[pair['coverage_limited']['dataset_index']]
                    expanded = train_rows[pair['expanded']['dataset_index']]
                    check(limited['source_kind'] == 'old' and limited['data_role'] == expanded['data_role'] == 'train', 'training coverage/role only')
                    check(limited['context_id'] == expanded['context_id'] == pair['context_id'], 'paired current context')
        baselines = json.loads((OUTPUT / 'baseline_result.json').read_text())
        check(baselines == result['baselines'], 'root baseline result matches')
        with np.load(OUTPUT / 'baseline_predictions.npz', allow_pickle=False) as saved:
            expected_keys = set()
            for data in DATA_CONDITIONS:
                indices = [i for i, row in enumerate(train_rows) if data == 'expanded' or row['source_kind'] == 'old']
                values, fallback = simple_predictions(take(train, indices), validation)
                check(baselines[data]['fallback'] == fallback, 'training-only fallback accounting')
                for name, p in values.items():
                    key = data + '__' + name
                    expected_keys.add(key)
                    check(np.array_equal(p, saved[key]), 'exact baseline inference')
                    report = coverage_report(p, validation)
                    choices = {label: moving_decisions(p, validation, h) for label, h in (('first_half_second', 0), ('three_seconds', 5))}
                    check(baselines[data]['controls'][name] == {'prediction': report, 'moving_decisions': choices}, 'baseline reduction')
                    check_primary(p, validation, report)
                    for label, h in (('first_half_second', 0), ('three_seconds', 5)):
                        check_choices(p, validation, choices[label], h)
            check(set(saved.files) == expected_keys, 'baseline population')
        all_rows = []
        for seed in SEEDS:
            for data in DATA_CONDITIONS:
                for condition in CONDITIONS:
                    directory = OUTPUT / f'{seed}-{data}-{condition}'
                    row = json.loads((directory / 'result.json').read_text())
                    check(row == result['models'][len(all_rows)], 'terminal model order/result equality')
                    check((row['seed'], row['data_condition'], row['condition'], row['updates']) == (seed, data, condition, UPDATES), 'model identity')
                    check(row['schedule_sha256'] == schedule_sha, 'model schedule identity')
                    check(set(row['artifact_sha256']) == {'final.pt', 'updates.jsonl', 'validation_predictions.npz'}, 'model artifact population')
                    for name, sha in row['artifact_sha256'].items():
                        check(digest(directory / name) == sha, 'model artifact digest')
                    check_updates([json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()], condition)
                    torch.manual_seed(seed)
                    model = TemporalRGBBodyJEPA()
                    initial = state_identity(model)
                    checkpoint = torch.load(io.BytesIO((directory / 'final.pt').read_bytes()), map_location='cpu', weights_only=True)
                    for key, value in (('seed', seed), ('condition', condition), ('data_condition', data), ('updates', UPDATES),
                                       ('launch_sha256', LAUNCH_SHA), ('schedule_sha256', schedule_sha), ('initial_state_sha256', initial)):
                        check(checkpoint[key] == value, 'checkpoint configuration/initial identity')
                    check(row['initial_state_sha256'] == initial, 'result initial identity')
                    check(row['active_trainable_parameters'] == sum(p.numel() for p in active_parameters(model, condition)), 'active parameter count')
                    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    model.eval()
                    with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved:
                        expected_keys = {'intact__context_latents', 'shuffle_eligible'}
                        for control in ('intact', 'rgb_shuffle', 'body_shuffle'):
                            values, z, eligible = predictions(model, validation, condition, control)
                            reports = {head: coverage_report(p, validation, None if control == 'intact' else eligible) for head, p in values.items()}
                            check(reports == row['validation'][control], 'replayed stratified metrics')
                            for head, p in values.items():
                                key = control + '__' + head
                                expected_keys.add(key)
                                check(np.array_equal(p, saved[key]), 'exact checkpoint prediction replay')
                                if control == 'intact':
                                    check_primary(p, validation, reports[head])
                                    for label, h in (('first_half_second', 0), ('three_seconds', 5)):
                                        choices = moving_decisions(p, validation, h)
                                        check(choices == row['moving_decisions'][head][label], 'replayed offline choices')
                                        check_choices(p, validation, choices, h)
                            if control == 'intact':
                                check(np.array_equal(z, saved['intact__context_latents']) and np.array_equal(eligible, saved['shuffle_eligible']), 'context latent and donor eligibility')
                                check(row['validation']['intact_matched_shuffle'] == {head: coverage_report(p, validation, eligible) for head, p in values.items()}, 'matched intact shuffle population')
                        check(set(saved.files) == expected_keys, 'prediction artifact population')
                    all_rows.append(row)
                    audited.append({'seed': seed, 'data_condition': data, 'condition': condition, 'maximum_prediction_difference': 0.})
                    print(json.dumps({'event': 'model_audited', 'completed': len(audited), 'total': 18}), flush=True)
        check(result['paired_comparisons'] == comparisons(all_rows), 'paired layout/seed contrasts')
        verify(sources)
        check(launch['input_sha256'] == input_bindings(), 'unchanged end-of-audit inputs')
        write_json(audit_path, {'status': 'PASS', 'audited_models': 18, 'models': audited,
                   'study_result_sha256': digest(OUTPUT / 'result.json'), 'launch_sha256': LAUNCH_SHA,
                   'audit_source_sha256': sources, 'schedule_sha256': schedule_sha,
                   'scope': 'source/artifact, schedule, checkpoint-inference and independent scalar endpoint audit; not optimizer retraining or navigation'})
        print(json.dumps({'status': 'PASS', 'audited_models': 18}), flush=True)
    except Exception as error:
        write_json(audit_path, {'status': 'FAIL', 'error': repr(error), 'audited_models': len(audited),
                   'models': audited, 'audit_source_sha256': sources, 'launch_sha256': LAUNCH_SHA,
                   'study_result_sha256': digest(OUTPUT / 'result.json')})
        raise


if __name__ == '__main__':
    main()
