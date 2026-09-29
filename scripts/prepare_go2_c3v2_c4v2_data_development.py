"""Matched training data for the C3-v2 readout and the C4-v2 refit (pre-declared 29 Sep 2026).

train rows: the C4-v1 prepared population (the v1 readout's 5,966 old + 2,448 maze contexts)
plus every new fit-split rest/turn context, added to the 'maze' pool. Held-out rest/turn
contexts go to a separate evaluation file and are never used for fitting.
Rows use the C4-v1 format (causal frame paths, normalised command history, executed applied
tape, eight true targets) plus the recording directory and frame for C3's native path.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
RECORDINGS = BASE/'c3v2_rest_turn_recordings_v1'
OUT = BASE/'c3v2_data_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prereg = json.loads(Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())
    stats = json.loads(Path(prereg['harness_v0']['shared_model_and_sensor_bindings']['normalization']['path']).read_text())
    mean, std = [np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std')]
    v1 = BASE/'c4_preparation'
    rows = json.loads((v1/'samples.json').read_text())
    paths = json.loads((v1/'frame_paths.json').read_text())
    lookup = {p: i for i, p in enumerate(paths)}
    for row in rows:
        current = Path(paths[row['frame_indices'][-1]])
        row.update(directory=str(current.parent), frame=int(current.stem.split('_')[-1]), source='c4_v1_population')
    # Consistency: C3's native path reproduces the v1 rows' command history on a sample.
    for row in rows[::400]:
        recording = Recording.training_case(row['directory'])
        past = recording.values[row['frame']][:, [0, 2]].reshape(3, 5, 2)
        np.testing.assert_allclose((past-mean)/std, np.asarray(row['control'], np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(recording.executed_tape(row['frame'])[:, [0, 2]], np.asarray(row['future_actions'], np.float32), rtol=0, atol=0)
    plan = json.loads((RECORDINGS/'plan.json').read_text())
    split = plan['splits']
    windows = json.loads((RECORDINGS/'samples.json').read_text())
    contexts = {}
    for w in windows:
        contexts.setdefault((w['case'], w['frame']), {})[w['horizon_ms']] = w['motion']
    fit, heldout = [], []
    for (case, frame), targets in sorted(contexts.items()):
        assert sorted(targets) == list(range(100, 801, 100))
        directory = RECORDINGS/f'case_{case:02d}'
        recording = Recording.training_case(directory)
        recording.native(frame)  # enforces the run-time causal-context checks
        indices = []
        for i in (frame-10, frame-5, frame):
            key = str((directory/f'rgb_{i:04d}.png').resolve())
            if key not in lookup:
                lookup[key] = len(paths)
                paths.append(key)
            indices.append(lookup[key])
        past = recording.values[frame][:, [0, 2]].reshape(3, 5, 2)
        row = dict(group='maze', context_index=None, frame_indices=indices, control=((past-mean)/std).tolist(),
                   future_actions=recording.executed_tape(frame)[:, [0, 2]].tolist(),
                   targets=[targets[h] for h in range(100, 801, 100)], directory=str(directory), frame=frame,
                   source=f'c3v2_rest_turn_{split[str(case)]}', case=case,
                   past_applied=recording.values[frame].tolist(), applied_tape=recording.executed_tape(frame).tolist())
        (fit if split[str(case)] == 'fit' else heldout).append(row)
    OUT.mkdir(exist_ok=False)
    output.install(BASE)
    train = rows+fit
    result = dict(schema='c3v2_matched_training_data.v1', contexts=len(train), old_contexts=sum(r['group'] == 'old' for r in train),
        maze_contexts_v1=sum(r['source'] == 'c4_v1_population' and r['group'] == 'maze' for r in train),
        new_fit_contexts=len(fit), heldout_contexts=len(heldout), frames=len(paths),
        v1_population_sha256={n: digest(v1/n) for n in ('samples.json', 'frame_paths.json', 'result.json')},
        recordings_samples_sha256=digest(RECORDINGS/'samples.json'), all_roles_train=True,
        heldout_used_for_fitting=False, validation_or_sealed_used=False)
    for name, value in (('train_samples.json', train), ('heldout_samples.json', heldout), ('frame_paths.json', paths), ('result.json', result)):
        owner.save(OUT/name, value)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
