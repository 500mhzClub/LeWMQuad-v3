"""Post-hoc actual-future readout check on explicitly trained large motions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import evaluate_go2_all_motion_horizon_readout_development as fixed

OUTPUT = fixed.fit.OUTPUT / 'training_translation_fit_diagnostic_v1'


@torch.inference_mode()
def main():
    base = fixed.fit.OUTPUT
    training_plan = json.loads((base / 'plan.json').read_text())
    for name, sha in training_plan['input_sha256'].items():
        assert fixed.fit.digest(base / name) == sha
    trained = json.loads((base / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['steps'] == 440
    data = json.loads((base / 'samples.json').read_text())
    paths = json.loads((base / 'frame_paths.json').read_text())
    schedule = json.loads((base / 'schedule.json').read_text())
    seen = Counter((i, h) for batch, horizons in zip(schedule['original'], schedule['original_horizons'])
                   for i, h in zip(batch, horizons))
    fixed_seen = Counter(i for batch in schedule['original'] for i in batch)
    by_recording = defaultdict(list)
    for i, (origin, targets) in enumerate(zip(data['original_origins'], data['original_targets'])):
        assert origin['data_role'] == 'train'
        if np.linalg.norm(targets[6][:2]) >= .05 and seen[i, 6] > 0:
            by_recording[origin['source'], origin['trial']].append(i)
    recordings = sorted(by_recording)
    assert len(recordings) >= 32
    selected_recordings = [recordings[int(i)] for i in np.linspace(0, len(recordings)-1, 32, dtype=int)]
    selected = []
    for key in selected_recordings:
        candidates = sorted(by_recording[key], key=lambda i: data['original_origins'][i]['frame'])
        selected.append(candidates[len(candidates)//2])
    assert len(set(selected)) == 32
    rows = []
    for i in selected:
        for h in (4, 6):
            rows.append(dict(sample_index=i, origin=data['original_origins'][i], horizon_ms=100*(h+1),
                pair=data['original_pairs'][i][h], actual=data['original_targets'][i][h],
                presentations=dict(fixed_500ms=fixed_seen[i] if h == 4 else 0,
                                   multi_100_800ms=seen[i, h])))
    needed = sorted({i for row in rows for i in row['pair']})
    receipts = {}
    for i in needed:
        path = Path(paths[i]).resolve()
        assert not any(p == 'sealed' or p.startswith('sealed_') for p in path.parts)
        assert path.is_file()
        marker = path.parent / 'depth_retention.json'
        if str(marker) not in receipts:
            receipts[str(marker)] = json.loads(marker.read_text()) if marker.exists() else None
    heads = {'starting_mixed': fixed.fit.prior.previous.load('mixed_data')}
    assert fixed.fit.digest(training_plan['initial_checkpoint']) == training_plan['initial_checkpoint_sha256']
    for arm in fixed.fit.ARMS:
        path = base / f'{arm}_final.pt'
        assert fixed.fit.digest(path) == trained['checkpoint_sha256'][arm]
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['updates'] == 440 and state['plan_sha256'] == fixed.fit.digest(base / 'plan.json')
        head = fixed.fit.prior.previous.load('mixed_data')
        head.load_state_dict(state['model_state_dict'])
        heads[arm] = head
    for head in heads.values():
        head.eval().requires_grad_(False)
    OUTPUT.mkdir(exist_ok=False)
    fixed.save(OUTPUT / 'plan.json', dict(source_sha256=fixed.fit.digest(__file__),
        training_plan_sha256=fixed.fit.digest(base / 'plan.json'),
        checkpoint_sha256=trained['checkpoint_sha256'],
        starting_checkpoint_sha256=training_plan['initial_checkpoint_sha256'],
        selection='700-ms XY >=50 mm and at least one actual multi-horizon training presentation; 32 evenly spaced sorted recordings; median eligible frame in each',
        eligible_recordings=len(recordings), eligible_contexts=sum(map(len, by_recording.values())),
        selected_rows=rows, image_paths=[paths[i] for i in needed], depth_retention_receipts=receipts,
        heads=list(heads), cpu_cores=[4, 5, 6, 7], no_fitting=True, no_predictor_inference=True,
        no_navigation=True, training_examples_not_generalization=True,
        limitations=['Post-hoc training-fit diagnostic; selection uses targets and training exposure, never errors.',
                    'Different images and motion distribution from maze evaluation; not an isolated domain-shift cause.',
                    'Fixed head saw the selected departures at 500 ms only; exact presentation counts retained.',
                    'Full-float pooled features at inference; training stored pooled features in FP16 RAM.']))
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        encoder = fixed.VJepa21Arm()
        encoder.build(torch.device('cpu'), torch.float32)
        pooled = {}
        for j, i in enumerate(needed):
            pixels = encoder.preprocess(paths[i])[None]
            pooled[i] = pool_tokens(F.layer_norm(encoder.tokens(pixels).float(), (1024,)))
            if (j+1) % 16 == 0 or j+1 == len(needed):
                print('TRAIN_TRANSLATION_FEATURES', j+1, len(needed), flush=True)
        del encoder
        for row in rows:
            a, b = row['pair']
            predictions = {name: head(pooled[a], pooled[b])[0].numpy() for name, head in heads.items()}
            predictions['zero_motion'] = np.zeros(3)
            errors = {}
            for name, value in predictions.items():
                delta = value - np.asarray(row['actual'])
                delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                errors[name] = delta.tolist()
            row.update(predictions={k: v.tolist() for k, v in predictions.items()}, errors=errors)
        summary = {str(h): fixed.metrics([r for r in rows if r['horizon_ms'] == h]) for h in (500, 700)}
        fixed.save(OUTPUT / 'result.json', dict(status='COMPLETE', rows=rows, by_horizon=summary,
            plan_sha256=fixed.fit.digest(OUTPUT / 'plan.json'), wall_s=time.monotonic()-started,
            training_examples_not_generalization=True, automatically_promoted=False))
        print('TRAIN_TRANSLATION_COMPLETE', json.dumps(summary), flush=True)
    except BaseException as error:
        fixed.save(OUTPUT / 'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
