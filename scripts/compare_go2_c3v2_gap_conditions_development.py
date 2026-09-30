"""Gap diagnosis, conditions: closed-loop decision inputs versus the offline held-out recordings.

Diagnosis on the check mazes only; validation and sealed sets are untouched. Read-only over
preserved records; no model runs.

Closed loop: every decision (model call) of the C3-v2 and C4-v2 fresh-check missions, plus
the subset whose executed tape equals the forward candidate's tape. Offline: the 1,384
held-out rest/turn contexts used by the acceptance evaluator.

The same features are computed for both, from the same kinds of record (the policy
histories, the physics trace and the camera transforms):
- speed and command history in the second before the decision;
- time since the last commanded movement and since the last body movement;
- camera pitch and roll;
- the committed prefix. In closed loop this is the first three 100-ms steps of the candidate
  tapes, fixed by the previous decision. Offline, the same three steps of the executed tape.

Continuous features are compared with the two-sample Kolmogorov-Smirnov statistic, and
categorical ones by the absolute difference in proportions. Both are on [0, 1], so the
features are ranked by that value.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
RUNS = ('c3v2_check_C3_chk*_ep0_attempt001', 'c3v2_check_C4_chk*_ep0_attempt001')
FORWARD = 1
MOVING_M_S = .02


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Source:
    def __init__(self, directory, histories, observations, trace, meta):
        self.recording = Recording(histories, observations, None, trace)
        with np.load(trace, allow_pickle=False) as a:
            self.pose = a['base_pose_world'].copy()
            self.time = a['timestamp_s'].copy()
        # Older training recordings keep no camera metadata: the sample index then comes from the image time
        # (verified identical to the recorded index where both exist) and camera pitch/roll are unavailable.
        self.meta = json.loads(Path(meta).read_text())['frames'] if Path(meta).exists() else None
        self.directory = directory

    def features(self, frame, tape):
        i = (int(self.meta[frame]['physical_sample_index']) if self.meta is not None
             else int(np.searchsorted(self.time, self.recording.stamps[frame]/1e9-1e-9)))
        t = self.time[i]
        at = lambda s: int(np.clip(np.searchsorted(self.time, t-s), 0, i))
        xy = self.pose[:, :2]
        speed_now = float(np.linalg.norm(xy[i]-xy[at(.1)])/max(t-self.time[at(.1)], 1e-9))
        speed_1s = float(np.linalg.norm(xy[i]-xy[at(1.)])/max(t-self.time[at(1.)], 1e-9))
        # Time since the body last moved faster than 2 cm/s (0.1-s windows, 5-s lookback).
        since_body = 5.
        for k in range(1, 51):
            a, b = at(.1*k), at(.1*(k-1))
            if b > a and np.linalg.norm(xy[b]-xy[a])/max(self.time[b]-self.time[a], 1e-9) > MOVING_M_S:
                since_body = .1*(k-1)
                break
        past = self.recording.values[frame]
        moving = np.any(past != 0, axis=1)
        since_command = 1.5 if not moving.any() else .1*(len(moving)-1-int(np.flatnonzero(moving)[-1]))
        if self.meta is not None:
            R = np.asarray(self.meta[frame]['transforms'][0], float)[:3, :3]
            pitch = float(np.degrees(np.arcsin(np.clip(R[2, 2], -1, 1))))
            roll = float(np.degrees(np.arctan2(R[2, 0], -R[2, 1])))
        else:
            pitch = roll = float('nan')
        tape = np.asarray(tape, float)
        prefix, rest = tape[:3], tape[3:]
        return dict(speed_now_m_s=speed_now, speed_mean_1s_m_s=speed_1s,
                    command_forward_mean_1s=float(past[-10:, 0].mean()), command_moving_fraction_1s=float(moving[-10:].mean()),
                    command_turning_fraction_1s=float((past[-10:, 2] != 0).mean()),
                    time_since_commanded_motion_s=since_command, time_since_body_motion_s=since_body,
                    camera_pitch_deg=pitch, camera_roll_deg=roll,
                    prefix_is_hold=bool(not prefix.any()), prefix_continues_last_command=bool(np.all(prefix == past[-1])),
                    prefix_differs_from_rest_of_tape=bool(np.any(prefix != rest[0])),
                    from_rest=bool(not past[-10:].any()))


def closed_loop():
    rows = []
    for pattern in RUNS:
        for run in sorted((BASE/'runs').glob(pattern)):
            if not (run/'episode_evaluation.json').exists():
                continue
            n = run/'native'
            source = Source(run, n/'policy_histories.npz', n/'policy_observations.json', n/'physics_trace.npz', n/'in_memory_camera_observations.json')
            for call in json.loads((run/'model_calls.json').read_text()):
                frame = (call['observed_ns']-1_500_000_000)//100_000_000
                if frame+8 >= len(source.meta) or not source.recording.valid[frame].all():
                    continue
                tapes = np.asarray(call['applied_commands'], np.float32)
                try:
                    executed = source.recording.executed_tape(frame)
                    forward_executed = bool(np.array_equal(executed, tapes[FORWARD]))
                except ValueError:
                    forward_executed = False
                rows.append(source.features(frame, tapes[FORWARD]) | dict(run=run.name, frame=int(frame), forward_executed=forward_executed))
    return rows


def offline(name='heldout_samples.json'):
    rows, cache = [], {}
    for r in json.loads((BASE/'c3v2_data_v1'/name).read_text()):
        d = Path(r['directory'])
        if d not in cache:
            cache[d] = Source(d, d/'policy_histories.npz', d/'policy_observations.json', d/'physics_trace.npz', d/'in_memory_camera_observations.json')
        rows.append(cache[d].features(r['frame'], cache[d].recording.executed_tape(r['frame'])) | dict(run=d.name, frame=r['frame'],
                                                                                                    source=r.get('source')))
    return rows


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, 'right')/len(a)-np.searchsorted(b, grid, 'right')/len(b))))


def compare(closed, reference):
    out = {}
    for key, value in closed[0].items():
        if key in ('run', 'frame', 'forward_executed'):
            continue
        a = np.asarray([r[key] for r in closed], float)
        b = np.asarray([r[key] for r in reference], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b):
            continue
        if isinstance(value, bool):
            out[key] = dict(kind='proportion', closed_loop=float(a.mean()), offline=float(b.mean()), distance=float(abs(a.mean()-b.mean())))
        else:
            q = lambda x: [float(np.percentile(x, p)) for p in (25, 50, 75)]
            out[key] = dict(kind='ks', closed_loop_quartiles=q(a), offline_quartiles=q(b), distance=ks(a, b))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['distance']))


def main():
    output.install(BASE)
    closed, reference = closed_loop(), offline()
    forward = [r for r in closed if r['forward_executed']]
    subsets = dict(all_decisions=closed, forward_executed=forward,
                   forward_executed_from_rest=[r for r in forward if r['from_rest']],
                   forward_executed_moving=[r for r in forward if not r['from_rest']])
    result = dict(schema='c3v2_gap_conditions.v1', label='Diagnosis on check mazes; validation and sealed sets untouched',
                  counts=dict({k: len(v) for k, v in subsets.items()}, offline_heldout=len(reference)),
                  offline_from_rest_fraction=float(np.mean([r['from_rest'] for r in reference])),
                  comparisons={k: compare(v, reference) for k, v in subsets.items() if v},
                  analyser_sha256=sha(__file__))
    root = BASE/'c3v2_gap_diagnosis_v1'
    root.mkdir(exist_ok=True)
    # Second reference: the matched training contexts of C3-v2 and C4-v2 (does training cover what closed loop visits?).
    training = offline('train_samples.json')
    result['training_reference'] = dict(contexts=len(training), comparisons={k: compare(v, training) for k, v in subsets.items() if v},
        speed_now_above_0_1_m_s=dict(closed_loop_forward_executed=float(np.mean([r['speed_now_m_s'] > .1 for r in forward])),
                                     offline_heldout=float(np.mean([r['speed_now_m_s'] > .1 for r in reference])),
                                     training=float(np.mean([r['speed_now_m_s'] > .1 for r in training])),
                                     training_by_source={s: float(np.mean([r['speed_now_m_s'] > .1 for r in training if r['source'] == s]))
                                                         for s in sorted({r['source'] for r in training})}))
    owner.save(root/f"conditions_{len({r['run'] for r in closed})}runs_with_training.json", result)
    for name, comparison in result['comparisons'].items():
        print(name, result['counts'][name])
        for key, value in list(comparison.items())[:14]:
            print(f"  {key:36s} {value['distance']:.3f}", value.get('closed_loop_quartiles', value.get('closed_loop')), '|', value.get('offline_quartiles', value.get('offline')))


if __name__ == '__main__':
    main()
