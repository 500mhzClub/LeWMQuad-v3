"""Describe new-maze command coverage on completed navigation windows; CPU only."""
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
OUTPUT = Path('docs/go2_maze_view_command_coverage_2026-09-23.json')


def history(directory):
    with np.load(directory / 'policy_histories.npz', allow_pickle=False) as data:
        return data['applied_command_values'].astype(np.float64), data['applied_command_valid'].copy()


def sequences(commands, valid, frames):
    frames = np.asarray(frames)
    assert valid[frames].all() and valid[frames + 7, -7:].all()
    past, future = commands[frames], commands[frames + 7, -7:]
    assert (past[:, :, 1] == 0).all() and (future[:, :, 1] == 0).all()
    return np.concatenate((past[:, :, [0, 2]], future[:, :, [0, 2]]), axis=1)


def evaluate():
    training = BASE / 'go2_maze_view_training_v1_attempt_001'
    sample_path = training / 'samples.json'
    samples = json.loads(sample_path.read_text())
    train = []
    for case in range(16):
        selected = [r for r in samples if r['case'] == case and r['horizon_ms'] == 700]
        assert all(r['data_role'] == 'train' for r in selected)
        train.append(sequences(*history(training / f'case_{case:02d}'), [r['frame'] for r in selected]))
    train = np.concatenate(train)
    unique = np.unique(train, axis=0)
    scale = np.max(np.abs(train), axis=(0, 1))
    assert np.all(scale > 0)
    results = {}
    for arm in ('old_data', 'maze_data'):
        root = BASE / f'go2_dense_world_model_maze_layout00_action_maze_view_{arm}_v1_attempt_001'
        report = json.loads((root / 'dense_navigation_readout.json').read_text())
        assert report['physical']['clean_result_available']
        windows = report['executed_windows']['rows']
        query = sequences(*history(root / 'native'), [r['frame'] for r in windows])
        rows = []
        for window, q in zip(windows, query):
            distances, matches = {}, {}
            for label, selection in (('past', slice(0, 15)), ('future', slice(15, 22)), ('joint', slice(0, 22))):
                difference = unique[:, selection] - q[selection]
                distances[label] = float(np.sqrt(np.mean((difference / scale) ** 2, axis=(1, 2))).min())
                matches[label] = bool((np.max(np.abs(difference), axis=(1, 2)) <= 1e-6).any())
            rows.append(dict(frame=window['frame'], group=window['group'],
                all_zero_past=bool(np.max(np.abs(q[:15])) <= 1e-6),
                all_zero_joint=bool(np.max(np.abs(q)) <= 1e-6),
                exact_training_match=matches, nearest_normalized_rms=distances))
        summary = {}
        for group in ('all', 'hold', 'translation', 'turn'):
            selected = [r for r in rows if group == 'all' or r['group'] == group]
            summary[group] = dict(windows=len(selected),
                all_zero_past=sum(r['all_zero_past'] for r in selected),
                all_zero_joint=sum(r['all_zero_joint'] for r in selected),
                exact_match_counts={k:sum(r['exact_training_match'][k] for r in selected) for k in ('past', 'future', 'joint')},
                nearest_normalized_rms={k:dict(median=float(np.median([r['nearest_normalized_rms'][k] for r in selected])),
                    p90=float(np.percentile([r['nearest_normalized_rms'][k] for r in selected], 90))) for k in ('past', 'future', 'joint')})
        results[arm] = dict(root=str(root), summary=summary, rows=rows)
    result = dict(status='COMPLETE', source=__file__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_samples_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        training=dict(contexts=len(train), unique_joint_sequences=len(unique),
            all_zero_past=int((np.max(np.abs(train[:, :15]), axis=(1, 2)) <= 1e-6).sum()),
            all_zero_joint=int((np.max(np.abs(train), axis=(1, 2)) <= 1e-6).sum())),
        definition='15 past applied forward/yaw pairs and seven executed future pairs at 100-ms intervals; nearest RMS scaled by training maximum absolute forward/yaw command.',
        scales=scale.tolist(), exact_match_tolerance=1e-6,
        population='Physical reader windows whose full 700-ms executed request sequence matched the selected forecast.',
        limitations=['Only the newly collected four-maze training population, not the older training pool.',
            'Command coverage is not visual or physical-state coverage, nor a causal explanation of failure.',
            'Future measured commands are post-hoc descriptive data, not proposed model inputs.',
            'Overlapping windows from one exposed layout per arm; not independent replications.',
            'No training, sensor replay, simulator execution or live controller change.'],
        navigation=results)
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'navigation'}, indent=2))
    print(json.dumps({a:r['summary'] for a,r in results.items()}, indent=2))


if __name__ == '__main__':
    evaluate()
