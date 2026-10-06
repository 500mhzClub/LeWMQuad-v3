"""Matched training data and held-out decisions for the C3-v3 round (pre-declared 30 Sep 2026, §3–§5).

**Training rows:** the C3-v2 matched training data (5,966 old + 5,216 maze-pool contexts),
unchanged, plus every on-policy context from the 16 `onpolicy_fit` C1 missions, in group
'onpolicy'. A frame t becomes a context if all of these hold:
- t >= 10, with a complete causal native context (the `Recording.native` checks);
- the executed applied tape for the eight 100-ms steps after t is complete;
- frames up to t+8 exist;
- the mission has zero disallowed contacts (otherwise the whole mission is excluded).
Targets are the physics-true motion in the decision body frame at 100–800 ms. Frames come
from verified replay.

**Held-out decisions (acceptance P):** every C1 decision of the 6 `onpolicy_heldout` missions.
Each carries the forward candidate's applied tape, whether it was executed, its forward
step count, from-rest status and the true motion.

The rows use the C4-v1 and C3-v2 format, plus `replay` and `run`. The data comes from C1, not
C3. Validation and sealed sets are untouched.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v3_onpolicy_data_development import (BASE, FIT, HELDOUT, REPLAYS, camera_frames, causal_context_complete,
                                                  run_name, run_recording)
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

PREREG = Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json')
OUT = BASE/'c3v3_data_v1'
FORWARD = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prereg = json.loads(PREREG.read_text())
    stats = json.loads(Path(prereg['harness_v0']['shared_model_and_sensor_bindings']['normalization']['path']).read_text())
    mean, std = [np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std')]
    previous = BASE/'c3v2_data_v1'
    rows = json.loads((previous/'train_samples.json').read_text())
    paths = json.loads((previous/'frame_paths.json').read_text())
    lookup = {p: i for i, p in enumerate(paths)}
    missions, fit, decisions = [], [], []
    for maze in list(FIT)+list(HELDOUT):
        run, replay = BASE/'runs'/run_name(maze), REPLAYS/run_name(maze)
        evaluation = json.loads((run/'episode_evaluation.json').read_text())
        verified = json.loads((replay/'replay_verification.json').read_text())['passed']
        assert verified, 'unverified replay: '+run.name
        split = 'fit' if maze in FIT else 'heldout'
        info = dict(run=run.name, maze=maze, split=split, round_trip=evaluation['round_trip_success'],
                    disallowed_contacts=evaluation['disallowed_contact_samples'], source_error=evaluation['source_error'])
        missions.append(info)
        if evaluation['disallowed_contact_samples']:
            info['excluded'] = 'disallowed contact'
            continue
        recording, meta = run_recording(run, replay), camera_frames(run)
        if split == 'fit':
            count = 0
            for frame in range(10, len(meta)-8):
                if not causal_context_complete(recording, frame):
                    continue
                try:
                    executed = recording.executed_tape(frame)
                except ValueError:
                    continue
                indices = []
                for i in (frame-10, frame-5, frame):
                    key = str((replay/'ego_frames'/f'{i:04d}.png').resolve())
                    if key not in lookup:
                        lookup[key] = len(paths)
                        paths.append(key)
                    indices.append(lookup[key])
                past = recording.values[frame][:, [0, 2]].reshape(3, 5, 2)
                fit.append(dict(group='onpolicy', context_index=None, frame_indices=indices, control=((past-mean)/std).tolist(),
                                future_actions=executed[:, [0, 2]].tolist(), targets=recording.targets(frame, meta).tolist(),
                                directory=str(run), replay=str(replay), run=run.name, frame=frame, source='c3v3_onpolicy_fit'))
                count += 1
            info['contexts'] = count
        else:
            calls = json.loads((run/'model_calls.json').read_text())
            count = 0
            for call in calls:
                frame = (call['observed_ns']-1_500_000_000)//100_000_000
                if frame < 10 or frame+8 >= len(meta) or not causal_context_complete(recording, frame):
                    continue
                try:
                    executed = recording.executed_tape(frame)
                except ValueError:
                    continue
                tape = np.asarray(call['applied_commands'][FORWARD], np.float32)
                decisions.append(dict(run=run.name, directory=str(run), replay=str(replay), frame=int(frame), observed_ns=call['observed_ns'],
                                      forward_tape=tape.tolist(), forward_executed=bool(np.array_equal(executed, tape)),
                                      forward_steps=int((tape[:, 0] > 0).sum()), from_rest=bool(not recording.values[frame][-10:].any()),
                                      executed_tape=executed.tolist(), targets=recording.targets(frame, meta).tolist()))
                count += 1
            info['decisions'] = count
    OUT.mkdir(exist_ok=False)
    output.install(BASE)
    train = rows+fit
    qualifying = [d for d in decisions if d['forward_executed'] and d['forward_steps'] >= 4 and not d['from_rest']]
    result = dict(schema='c3v3_matched_training_data.v1', predeclaration_commit='ec2e34c9', contexts=len(train),
                  old_contexts=sum(r['group'] == 'old' for r in train), maze_pool_contexts=sum(r['group'] == 'maze' for r in train),
                  onpolicy_fit_contexts=len(fit), frames=len(paths), missions=missions,
                  heldout_decisions=len(decisions), p_qualifying_moving_full_forward=len(qualifying),
                  p_decidable=len(qualifying) >= 30, onpolicy_source_controller='C1',
                  c3v2_data_sha256={n: digest(previous/n) for n in ('train_samples.json', 'frame_paths.json')},
                  heldout_used_for_fitting=False, validation_or_sealed_used=False)
    for name, value in (('train_samples.json', train), ('frame_paths.json', paths), ('heldout_onpolicy_decisions.json', decisions),
                        ('result.json', result)):
        owner.save(OUT/name, value)
    print(json.dumps({k: v for k, v in result.items() if k != 'missions'}))


if __name__ == '__main__':
    main()
