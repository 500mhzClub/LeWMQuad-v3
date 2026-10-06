"""Marker-visibility probe: can frozen V-JEPA features tell tinted floor from plain floor? (development; Andrew, 5 October 2026).

Andrew: before the refits, "a linear classifier on frozen V-JEPA features of tinted versus untinted floor frames
(held-out mazes for test). Report accuracy. If it's near chance, stop and tell me before the refits."

**Frames.** The verified-replay egocentric frames of the stage-2 recordings (stage2_recording_replays/*/ego_frames, marked
strips), every second frame. Train on stage2_fit mazes, test on stage2_heldout mazes.

**Labels, from the rendered pixels.** The tint scales each floor quad's colour by (0.45, 0.70, 1.00), a signature that
grey floor and walls (r = g = b) and the dark sky do not have. A pixel is tinted when:
- b >= 60;
- |r/b - 0.45| <= 0.07;
- |g/b - 0.70| <= 0.07.
A frame is "tinted" if at least 2% of its pixels are tinted, "untinted" if none are; frames in between are excluded.

**Features.** The frozen V-JEPA 2.1 encoder, exactly as the C3 feature cache uses it:
- 512x384 bicubic resize, the cache's normalisation;
- encoder.tokens, then layer norm, then pool_tokens to 192 x 1024;
- then the mean and the max over the 192 tokens (2,048 values), standardised on the training set.

**Classifier.** L2-regularised logistic regression (lambda = 1e-3, fixed in advance, nothing tuned on the test set),
full-batch L-BFGS.

**Reported** on the held-out mazes: accuracy, balanced accuracy (chance 0.5), AUC, per-maze accuracy, and class counts.
A pixel-space reference (the same classifier on the frame's mean colour ratios) shows how separable the marker is in raw
pixels.

Usage: probe_go2_marker_visibility_development.py --out DIR [--stride 2]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from lewm.c3v2_offline_pipeline_development import _normalise, _to_chw
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
REPLAYS = BASE/'stage2_recording_replays'
REGISTRY = BASE/'stage2_sets_v1_registry.json'
LAMBDA = 1e-3
TINTED_MIN, PIXEL_B_MIN, RATIO_TOL = .02, 60, .07


def tint_fraction(rgb):
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (b >= PIXEL_B_MIN) & (np.abs(r/np.maximum(b, 1)-.45) <= RATIO_TOL) & (np.abs(g/np.maximum(b, 1)-.70) <= RATIO_TOL)
    return float(mask.mean())


def collect(stride):
    roles = {e['maze_id']: e['role'] for e in json.loads(REGISTRY.read_text())['entries']}
    rows = []
    for replay in sorted(REPLAYS.glob('dev_*')):
        verification = replay/'replay_verification.json'
        if not replay.is_dir() or not verification.exists() or not json.loads(verification.read_text()).get('passed'):
            continue
        maze = int(replay.name.split('stage2_')[1].split('_')[0].lstrip('fitheldout'))
        for path in sorted((replay/'ego_frames').glob('*.png'))[::stride]:
            rgb = np.asarray(Image.open(path).convert('RGB'))
            f = tint_fraction(rgb)
            label = 1 if f >= TINTED_MIN else 0 if f == 0 else None
            if label is None:
                continue
            rows.append(dict(path=str(path), maze=maze, role=roles[maze], label=label, tint_fraction=f,
                             pixel=[*(rgb.reshape(-1, 3).mean(0)/255), *(rgb[240:].reshape(-1, 3).mean(0)/255)]))
    return rows


@torch.inference_mode()
def encode(rows, batch=16):
    model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
    device = next(model.predictor.parameters()).device
    out = np.zeros((len(rows), 2048), np.float32)
    for k in range(0, len(rows), batch):
        chunk = rows[k:k+batch]
        pixels = torch.stack([_normalise(_to_chw(Image.open(r['path']).convert('RGB').resize((512, 384), Image.Resampling.BICUBIC)))
                              for r in chunk]).to(device)
        tokens = F.layer_norm(model.encoder.tokens(pixels).float(), (1024,))
        pooled = pool_tokens(tokens).float()
        out[k:k+len(chunk)] = torch.cat((pooled.mean(1), pooled.amax(1)), dim=1).cpu().numpy()
    return out


def fit_logistic(x, y):
    x, y = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x@w+b, y)+LAMBDA*(w@w)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score)+1)
    pos = y == 1
    return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*(~pos).sum()))


def evaluate(train_x, train_y, test_x, test_y, test_maze):
    mean, std = train_x.mean(0), train_x.std(0)+1e-6
    w, b = fit_logistic((train_x-mean)/std, train_y)
    score = ((test_x-mean)/std)@w+b
    pred = (score > 0).astype(int)
    bal = .5*(np.mean(pred[test_y == 1] == 1)+np.mean(pred[test_y == 0] == 0))
    per_maze = {int(m): dict(n=int((test_maze == m).sum()), accuracy=float(np.mean(pred[test_maze == m] == test_y[test_maze == m])))
                for m in np.unique(test_maze)}
    train_pred = ((((train_x-mean)/std)@w+b) > 0).astype(int)
    return dict(test_accuracy=float(np.mean(pred == test_y)), test_balanced_accuracy=float(bal), test_auc=auc(score, test_y),
                train_accuracy=float(np.mean(train_pred == train_y)), per_heldout_maze=per_maze)


def main(out, stride):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = collect(stride)
    train = np.array([r['role'] == 'stage2_fit' for r in rows])
    y = np.array([r['label'] for r in rows])
    maze = np.array([r['maze'] for r in rows])
    counts = dict(train=dict(tinted=int(y[train].sum()), untinted=int((y[train] == 0).sum())),
                  test=dict(tinted=int(y[~train].sum()), untinted=int((y[~train] == 0).sum())))
    features = encode(rows)
    pixel = np.array([r['pixel'] for r in rows], np.float32)
    result = dict(
        frames=len(rows), stride=stride, counts=counts, lambda_l2=LAMBDA, label_rule=dict(tinted_min_fraction=TINTED_MIN,
        pixel_b_min=PIXEL_B_MIN, ratio_tolerance=RATIO_TOL, tint=[.45, .70, 1.0]),
        vjepa=evaluate(features[train], y[train], features[~train], y[~train], maze[~train]),
        pixel_reference=evaluate(pixel[train], y[train], pixel[~train], y[~train], maze[~train]),
        heldout_mazes=sorted({int(m) for m in maze[~train]}), wall_s=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--stride', type=int, default=2)
    a = p.parse_args()
    main(a.out, a.stride)
