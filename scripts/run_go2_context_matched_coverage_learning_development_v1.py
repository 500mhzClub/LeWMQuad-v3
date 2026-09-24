#!/usr/bin/env python3
"""Fixed 18-model comparison of observed action coverage, not navigation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lewm.context_matched_coverage_development import AugmentedCausalDataset, matched_schedule, TENSOR_CHECK
from lewm.coverage_prediction_metrics_development import (
    coverage_report, coverage_shuffle, moving_decisions, paired_layout_delta)
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.temporal_rgb_body_learning_development import CONDITIONS, active_parameters, training_loss
from lewm.temporal_prediction_metrics_development import simple_predictions
from scripts.run_go2_temporal_rgb_body_learning_comparison_development_v1 import (
    digest, write_json, safe_path, stack, take, state_identity, input_bindings as old_inputs)

OUTPUT = ROOT / '.generated/go2_context_matched_coverage_learning_development_v1_attempt_001'
PROTOCOL = 'docs/go2_context_matched_coverage_learning_development_v1_2026-09-05.md'
SEEDS = (2026092400, 2026092401, 2026092402)
DATA_CONDITIONS = ('coverage_limited', 'expanded')
UPDATES = 1200
TESTS = (
    'lewm/tests/test_context_matched_coverage_development.py',
    'lewm/tests/test_coverage_prediction_metrics_development.py',
    'lewm/tests/test_context_matched_coverage_runner_development.py',
)


def source_bindings():
    # Ignore-aware discovery is only of source names; no export or raw corpus scan.
    available = set(subprocess.run(
        ['rg', '--files', '-g', '*.py', 'lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines())
    pending = [str(Path(__file__).relative_to(ROOT)), *TESTS]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        if name not in available:
            raise ValueError('source absent from ignore-aware discovery: ' + name)
        tree = ast.parse(safe_path(name).read_text())
        visited.add(name)
        parts = Path(name).parts[:-1]
        for end in range(1, len(parts) + 1):
            init = str(Path(*parts[:end]) / '__init__.py')
            if init in available:
                pending.append(init)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = '.'.join([*parts[:len(parts) - node.level + 1], *([module] if module else [])])
                modules = [module, *[module + '.' + a.name for a in node.names]]
            else:
                continue
            for module in modules:
                if module.split('.')[0] not in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'):
                    continue
                rel = module.replace('.', '/')
                pending.extend(p for p in (rel + '.py', rel + '/__init__.py') if p in available)
    return {name: digest(safe_path(name)) for name in sorted(visited | {PROTOCOL})}


def verify(bindings):
    for name, sha in bindings.items():
        if digest(safe_path(name)) != sha:
            raise ValueError('bound source/input changed: ' + name)


def input_bindings():
    launch = json.loads((TENSOR_CHECK / 'launch.json').read_text())
    result = json.loads((TENSOR_CHECK / 'result.json').read_text())
    if (result['status'] != 'PASS' or result['checked_cells'] != 600
            or result['launch_sha256'] != digest(TENSOR_CHECK / 'launch.json')):
        raise ValueError('full composite qualification required')
    bound = old_inputs()
    for section in ('source_sha256', 'input_sha256', 'gait_sha256'):
        for name, sha in launch[section].items():
            if name in bound and bound[name] != sha:
                raise ValueError('conflicting prerequisite binding')
            bound[name] = sha
    for name in ('launch.json', 'result.json'):
        path = TENSOR_CHECK / name
        bound[str(path.relative_to(ROOT))] = digest(path)
    verify(bound)
    return bound


def materialize(role):
    dataset = AugmentedCausalDataset(role)
    rows, metadata = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        metadata.append(item.pop('metadata'))
        rows.append(item)
        if (i + 1) % 100 == 0:
            print(json.dumps({'event': 'window_loaded', 'role': role, 'completed': i + 1,
                              'total': len(dataset)}), flush=True)
    batch = stack(rows)
    batch['metadata'] = metadata
    groups = {}
    for i, row in enumerate(metadata):
        first = groups.setdefault(row['context_id'], i)
        if any(not torch.equal(value[first], value[i]) for value in batch['observation_history'].values()):
            raise ValueError('paired current context tensor differs')
    if len(groups) != (546 if role == 'train' else 272):
        raise ValueError('current-context population changed')
    return dataset.rows, batch


@torch.no_grad()
def predictions(model, batch, condition, control='intact'):
    if condition not in CONDITIONS or control not in ('intact', 'rgb_shuffle', 'body_shuffle'):
        raise ValueError('fixed condition/control required')
    donors, eligible = coverage_shuffle(batch['metadata'])
    output = {'direct': []}
    if condition != 'direct':
        output['rollout'] = []
    latents = []
    for start in range(0, len(batch['metadata']), 16):
        indices = list(range(start, min(start + 16, len(batch['metadata']))))
        part = take(batch, indices)
        history = part['observation_history']
        if control != 'intact':
            name = 'rgb' if control == 'rgb_shuffle' else 'body'
            history[name] = batch['observation_history'][name][donors[indices]]
        plans, valid = part['known_action_blocks'], part['known_action_valid']
        if condition == 'direct':
            z, _ = model.encode_history(history)
            p = model.direct(z, plans)
            result = {'latent': z, 'direct_outcomes': torch.where(valid.all(-1)[:, :, None], p, torch.zeros_like(p))}
        else:
            result = model(history, plans, valid)
        latents.append(result['latent'].numpy())
        for head in output:
            output[head].append(result[head + '_outcomes'].numpy())
    return {k: np.concatenate(v) for k, v in output.items()}, np.concatenate(latents), eligible


def comparisons(runs):
    lookup = {(r['data_condition'], r['condition'], r['seed']): r for r in runs}
    if len(lookup) != 18 or len(runs) != 18:
        raise ValueError('all fixed models required; no best-seed selection')
    heads = [('direct', 'direct'), ('supervised_rollout', 'direct'),
             ('supervised_rollout', 'rollout'), ('jepa', 'direct'), ('jepa', 'rollout')]
    contrasts = [(("expanded", c, h), ("coverage_limited", c, h)) for c, h in heads]
    for data in DATA_CONDITIONS:
        contrasts.extend([((data, 'jepa', h), (data, 'supervised_rollout', h)) for h in ('direct', 'rollout')])
        contrasts.extend([((data, c, 'rollout'), (data, c, 'direct')) for c in ('supervised_rollout', 'jepa')])
        contrasts.append(((data, 'jepa', 'direct'), (data, 'direct', 'direct')))
    result = {}
    for a, b in contrasts:
        rows = {}
        for stratum in ('moving_switch', 'moving_continuation', 'old_later', 'old_initial'):
            for horizon in ('first_half_second', 'three_seconds', 'all_known'):
                for metric in ('position_error_m', 'contact_brier'):
                    def get(which):
                        return [lookup[(*which[:2], seed)]['validation']['intact'][which[2]][stratum][horizon]['layouts'] for seed in SEEDS]
                    rows[f'{stratum}/{horizon}/{metric}'] = paired_layout_delta(get(a), get(b), metric)
        for horizon in ('first_half_second', 'three_seconds'):
            for metric in ('regret', 'contact', 'realized_cost'):
                def get_choice(which):
                    return [lookup[(*which[:2], seed)]['moving_decisions'][which[2]][horizon]['layouts'] for seed in SEEDS]
                rows[f'choice/{horizon}/{metric}'] = paired_layout_delta(get_choice(a), get_choice(b), metric)
        result['/'.join(a) + '_minus_' + '/'.join(b)] = rows
    return result


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh output, no overrides/retry/resume')
    sources, inputs = source_bindings(), input_bindings()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUTPUT.mkdir()
    started = time.monotonic()
    runs, initial = [], {}
    launch = {'schema': 'context_matched_coverage_learning_development.v1',
              'source_sha256': sources, 'input_sha256': inputs, 'seeds': SEEDS,
              'data_conditions': DATA_CONDITIONS, 'conditions': CONDITIONS, 'updates': UPDATES,
              'batch': 'one shared current context per each of sixteen training layouts',
              'device': 'cpu', 'threads': 1, 'torch': torch.__version__, 'numpy': np.__version__,
              'optimizer': {'name': 'AdamW', 'lr': .0003, 'weight_decay': .0001, 'clip': 5.},
              'ema': .99, 'loss_weights': {'direct': 1., 'rollout': 1., 'jepa_latent': 1., 'variance': .1, 'covariance': .01},
              'selection': 'final fixed update, all seeds, no tuning', 'scope': 'offline development, not online or maze qualification'}
    write_json(OUTPUT / 'launch.json', launch)
    try:
        train_rows, train = materialize('train')
        _, validation = materialize('validation')
        if {r['layout_id'] for r in train['metadata']} & {r['layout_id'] for r in validation['metadata']}:
            raise ValueError('layout role overlap')
        schedules = {str(seed): matched_schedule(train_rows, updates=UPDATES, seed=seed) for seed in SEEDS}
        write_json(OUTPUT / 'schedules.json', {'train_order': train_rows, 'schedules': schedules})
        write_json(OUTPUT / 'validation_order.json', validation['metadata'])
        schedule_sha = digest(OUTPUT / 'schedules.json')
        baselines, baseline_predictions = {}, {}
        for data_condition in DATA_CONDITIONS:
            indices = [i for i, r in enumerate(train_rows) if data_condition == 'expanded' or r['source_kind'] == 'old']
            values, fallback = simple_predictions(take(train, indices), validation)
            baselines[data_condition] = {'fallback': fallback, 'controls': {
                name: {'prediction': coverage_report(p, validation),
                       'moving_decisions': {label: moving_decisions(p, validation, h) for label, h in (('first_half_second', 0), ('three_seconds', 5))}}
                for name, p in values.items()}}
            baseline_predictions.update({data_condition + '__' + name: p for name, p in values.items()})
        write_json(OUTPUT / 'baseline_result.json', baselines)
        np.savez_compressed(OUTPUT / 'baseline_predictions.npz', **baseline_predictions)
        for seed in SEEDS:
            for data_condition in DATA_CONDITIONS:
                for condition in CONDITIONS:
                    directory = OUTPUT / f'{seed}-{data_condition}-{condition}'
                    directory.mkdir()
                    torch.manual_seed(seed)
                    model = TemporalRGBBodyJEPA().train()
                    identity = state_identity(model)
                    if initial.setdefault(seed, identity) != identity:
                        raise ValueError('initial model pairing changed')
                    parameters = active_parameters(model, condition)
                    optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001)
                    fit_start = time.monotonic()
                    with (directory / 'updates.jsonl').open('x') as log:
                        for step in schedules[str(seed)]:
                            indices = [r[data_condition]['dataset_index'] for r in step['batch']]
                            optimizer.zero_grad(set_to_none=True)
                            loss, parts = training_loss(model, take(train, indices), condition)
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                            optimizer.step()
                            model.update_target(.99)
                            row = {'update': step['update'] + 1, 'loss': float(loss.detach()),
                                   'gradient_norm_before_clip': float(norm), **parts}
                            log.write(json.dumps(row, allow_nan=False) + '\n')
                            log.flush()
                            if row['update'] % 100 == 0:
                                print(json.dumps({'event': 'training_update', 'seed': seed, 'data_condition': data_condition,
                                                  'condition': condition, **row}), flush=True)
                    fit_seconds = time.monotonic() - fit_start
                    model.eval()
                    torch.save({'model_state_dict': model.state_dict(), 'seed': seed, 'data_condition': data_condition,
                                'condition': condition, 'updates': UPDATES, 'launch_sha256': digest(OUTPUT / 'launch.json'),
                                'schedule_sha256': schedule_sha, 'initial_state_sha256': identity}, directory / 'final.pt')
                    metrics, stored, choices = {}, {}, {}
                    for control in ('intact', 'rgb_shuffle', 'body_shuffle'):
                        values, z, eligible = predictions(model, validation, condition, control)
                        metrics[control] = {head: coverage_report(p, validation, None if control == 'intact' else eligible) for head, p in values.items()}
                        stored.update({control + '__' + head: p for head, p in values.items()})
                        if control == 'intact':
                            metrics['intact_matched_shuffle'] = {head: coverage_report(p, validation, eligible) for head, p in values.items()}
                            choices = {head: {label: moving_decisions(p, validation, h) for label, h in (('first_half_second', 0), ('three_seconds', 5))} for head, p in values.items()}
                            stored.update(intact__context_latents=z, shuffle_eligible=eligible)
                    np.savez_compressed(directory / 'validation_predictions.npz', **stored)
                    row = {'seed': seed, 'data_condition': data_condition, 'condition': condition,
                           'updates': UPDATES, 'initial_state_sha256': identity, 'schedule_sha256': schedule_sha,
                           'active_trainable_parameters': sum(p.numel() for p in parameters),
                           'fit_seconds': fit_seconds, 'validation': metrics, 'moving_decisions': choices,
                           'artifact_sha256': {name: digest(directory / name) for name in ('final.pt', 'updates.jsonl', 'validation_predictions.npz')}}
                    write_json(directory / 'result.json', row)
                    runs.append(row)
                    print(json.dumps({'event': 'model_finished', 'seed': seed, 'data_condition': data_condition,
                                      'condition': condition, 'completed': len(runs), 'total': 18}), flush=True)
        if sources != source_bindings() or inputs != input_bindings() or digest(OUTPUT / 'schedules.json') != schedule_sha:
            raise ValueError('source/input/schedule changed during study')
        write_json(OUTPUT / 'result.json', {
            'status': 'COMPLETE', 'models': runs, 'baselines': baselines, 'paired_comparisons': comparisons(runs),
            'launch_sha256': digest(OUTPUT / 'launch.json'), 'schedule_sha256': schedule_sha,
            'artifact_sha256': {name: digest(OUTPUT / name) for name in ('schedules.json', 'validation_order.json', 'baseline_result.json', 'baseline_predictions.npz')},
            'elapsed_seconds': time.monotonic() - started,
            'scope': 'offline reused development layouts; no executed proposals, generalization or hardware claim'})
        print(json.dumps({'status': 'COMPLETE', 'models': len(runs)}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'INFRASTRUCTURE_FAILURE', 'error': repr(error),
                   'models_completed': len(runs), 'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
