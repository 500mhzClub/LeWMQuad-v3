"""Stage-2 feature cache: the decoder fix's contexts plus the stage-2 patch recordings (development; Andrew, 3 and 5 October 2026).

Plan (docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md, stage 2): "The feature cache is rebuilt: the earlier
pools (34,745 contexts; deleted after the decoder choice) plus up to 12,000 patch and 4,000 off-patch contexts from the
new missions." Andrew, 5 October: "build the training set only from decisions with meaningful motion (exclude holds and
latched-veto ticks)".

**Earlier contexts.** `scripts/build_go2_dev_c3_feature_cache_development.py`'s `collect()` is rerun unchanged (same seed,
same sources). Its items must equal the surviving `dev_c3_cache_v1/items.json` exactly, or the build stops. These are the
train set (old, maze and on-policy groups) and the eval_onpolicy, eval_offline, eval_transfer and eval_fresh_c3 sets.

**Stage-2 contexts** come from the `s2rec2` C1 recordings (marked strips, patches3, mu_p = 0.3):
- missions with a disallowed contact contribute nothing (the on-policy rule), and every mission's frame replay must have
  passed verification;
- a context is a C1 decision (a planning record with a selection) at frame t = (measured_ns - 1.5 s) / 100 ms;
- it is usable under the counting script's rule (scripts/count_go2_stage2_recording_contexts_development.py): over
  [t, t + 700 ms) the requested commands are not all hold, and no dispatch tick has reason CURRENT_OBSERVED_OBSTACLE_VETO or
  COMMAND_WINDOW_VETO_LATCHED;
- it has a complete causal context and executed tape, as for the C3-v3 on-policy contexts
  (`causal_context_complete`, `executed_tape`), and frames up to t + 8;
- it is binned by the body centre's signed distance to the nearest strip edge at t and t + 700 ms (`edge_bin`: approach,
  entry, on_patch, exit or off_patch).

| Mazes | Set | Group | Horizons |
|---|---|---|---|
| stage2_fit (0-23) | train | `patch` (approach, entry, on_patch, exit) | two random of 1-8, as the train set |
| stage2_fit (0-23) | train | `patch_off` (off_patch), at most 4,000 (seeded subset) | two random of 1-8 |
| stage2_heldout (24-29) | eval_patch | all bins | 500 and 800 ms |

Each stage-2 item also carries run, maze, role, edge_bin and the two signed edge distances.

**Encoding** is the decoder fix's `main()` unchanged (frozen V-JEPA 2.1 encoder; frames.f16 and pred.f16 as float16), run
with this module's `collect` and output directory.

Usage: build_go2_stage2_feature_cache_development.py [--check]
  --check: collect, verify the reproduction, count and confirm every frame file exists; no encoding.
"""
import argparse
from collections import Counter
import json
import os

import numpy as np

from lewm.c3v3_onpolicy_data_development import camera_frames, causal_context_complete
from lewm.dev_dynamics_patches_v2_development import signed_edge_distance
from scripts import build_go2_dev_c3_feature_cache_development as v1
from scripts.count_go2_stage2_recording_contexts_development import HORIZON, STEP_NS, VETOES
from scripts.score_go2_dev_closed_loop_prediction_development import cohort_runs
from scripts.score_go2_dev_patch_edge_prediction_development import edge_bin

OUT = v1.BASE/'stage2_feature_cache_v1'
REPLAYS = v1.BASE/'stage2_recording_replays'
COHORT = 's2rec2'
SEED = 2026100517
OFF_PATCH_CAP = 4000
DECODER_FIX_COLLECT = v1.collect  # bound before main() swaps in this module's collect


def contributing(run):
    failure = run/'failure.json'
    if failure.exists() and 'DISALLOWED_CONTACT' in json.loads(failure.read_text()).get('reason', ''):
        return False
    evaluation = run/'episode_evaluation.json'
    return not (evaluation.exists() and json.loads(evaluation.read_text()).get('disallowed_contact_samples'))


def stage2_contexts(run):
    """Usable C1 decisions of one recording: (frame, edge bin, edge distance at t and t + 700 ms)."""
    with np.load(run/'native/physics_trace.npz', allow_pickle=False) as z:
        stamps = np.rint(z['timestamp_s']*1e9).astype(np.int64)
        pose, requested = z['base_pose_world'].copy(), z['requested_command'].copy()
    rects = [p['rect'] for p in json.loads((run/'native/dynamics_patches.json').read_text())['placement']['patches']]
    vetoed = np.array([r['now_ns'] for r in json.loads((run/'requests.json').read_text()) if r.get('reason') in VETOES], np.int64)
    index = {t: i for i, t in enumerate(stamps)}
    out = []
    for row in json.loads((run/'planning.json').read_text()):
        if 'selection' not in row:
            continue
        t = row['measured_ns']
        steps = [index.get(t+k*STEP_NS+STEP_NS//2) for k in range(HORIZON)]
        start, end = index.get(t), index.get(t+HORIZON*STEP_NS)
        if None in steps or start is None or end is None:
            continue
        if not np.any(np.abs(requested[steps]) > 1e-9):
            continue
        if np.any((vetoed >= t) & (vetoed < t+HORIZON*STEP_NS)):
            continue
        d0, d1 = signed_edge_distance(pose[start, :2], rects), signed_edge_distance(pose[end, :2], rects)
        out.append(((t-1_500_000_000)//100_000_000, t, edge_bin(d0, d1), float(d0), float(d1)))
    return out


def collect():
    items, sources = DECODER_FIX_COLLECT()
    saved = json.loads((v1.OUT/'items.json').read_text())
    if len(saved) != len(items) or any({k: v for k, v in s.items() if k != 'frame_rows'} != json.loads(json.dumps(i))
                                       for s, i in zip(saved, items)):
        raise ValueError('the decoder-fix contexts were not reproduced exactly')
    rng = np.random.default_rng(SEED)
    fit_off, report = [], Counter()
    for _arm, run in cohort_runs(COHORT):
        replay = REPLAYS/run.name
        verification = replay/'replay_verification.json'
        if not contributing(run):
            report['excluded: contact'] += 1
            continue
        if not verification.exists() or not json.loads(verification.read_text()).get('passed'):
            raise ValueError(f'unverified replay: {run.name}')
        packet = json.loads((run/'episode.json').read_text())
        role, maze = packet['role'], int(packet['maze_id'])
        source = v1.training_source(str(run), str(replay))
        sources[source.key] = source
        recording, meta = source.recording, camera_frames(run)
        for frame, t, bin_, d0, d1 in stage2_contexts(run):
            if frame < 10 or frame+8 >= len(meta) or meta[frame]['measured_ns'] != t or not causal_context_complete(recording, frame):
                report['skipped: causal context'] += 1
                continue
            try:
                tape = recording.executed_tape(frame)
            except ValueError:
                report['skipped: executed tape'] += 1
                continue
            past = recording.values[frame]
            item = dict(set='train' if role == 'stage2_fit' else 'eval_patch', source=source.key, frame=int(frame),
                        tape=np.asarray(tape, np.float32).tolist(), past=np.asarray(past, np.float32).tolist(),
                        targets=np.asarray(recording.targets(frame, meta), np.float32).tolist(),
                        horizons=sorted(int(x) for x in rng.choice(np.arange(1, 9), size=2, replace=False)) if role == 'stage2_fit' else [5, 8],
                        category=v1.category(tape, past), run=run.name, maze=maze, role=role, edge_bin=bin_,
                        edge_distance_start_m=d0, edge_distance_end_m=d1)
            if role == 'stage2_fit':
                item['group'] = 'patch_off' if bin_ == 'off_patch' else 'patch'
                if bin_ == 'off_patch':
                    fit_off.append(item)
                    continue
            items.append(item)
    keep = rng.permutation(len(fit_off))[:OFF_PATCH_CAP]
    items.extend(fit_off[k] for k in sorted(keep))
    report['fit off-patch available'] = len(fit_off)
    print(json.dumps(dict(stage2_report=report)), flush=True)
    return items, sources


def check():
    items, sources = collect()
    counts = Counter((it['set'], it.get('group', ''), it.get('edge_bin', '')) for it in items)
    missing = [path for it in items for f in (it['frame']-10, it['frame']-5, it['frame'])
               for path in [sources[it['source']].frame_file(f)] if not os.path.exists(path)]
    stage2 = [it for it in items if 'edge_bin' in it]
    print(json.dumps(dict(items=len(items), stage2_items=len(stage2), missing_frame_files=len(missing), first_missing=missing[:3],
                          counts={' / '.join(k): v for k, v in sorted(counts.items())},
                          stage2_categories={f"{k[0]} / {k[1]}": v for k, v in Counter((it['set'], it['category']) for it in stage2).most_common()}), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        check()
    else:
        v1.OUT, v1.collect = OUT, collect
        v1.main()
