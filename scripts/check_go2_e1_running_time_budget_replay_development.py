"""Bitwise replay check of the E1 running-time budget (Andrew's ruling of 29 Sep 2026, item 3).

1. Stop behaviour, on a scratch root whose calendar origin is past the 160-h window: the
   owner's Budget stops; the running-time budget does not, and stops once E1 running time
   reaches its cap.
2. Replay: one recorded development episode (V4 C1 screen, development maze 00, episode 0; not
   validation, not sealed) re-runs through the unchanged frozen owner with only `Budget`
   swapped, then the frozen reader runs on it.
3. Every record is compared with the original run. Declared before the run: only the identity
   and wall-clock fields in ALLOWED may differ, after replacing the assignment name in paths.
   Everything else must be bitwise identical: decisions, dispatch commands, model calls, native
   trace arrays, consumed packet hashes, mission rows and the episode evaluation.

Attempt 1 (`e1_budget_replay_check_v1`) failed as declared: its allow-list missed four
wall-clock measurement fields (routing compute time, camera acquisition wall time, decision
latency, and the evaluator's hash of the planning file that carries the routing time) and the
log order of parallel service stages finishing at the same simulated time. It is preserved.
Attempt 2 adds exactly those, declared before its run; stage timings are compared as a
multiset without wall time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm import e1_running_time_budget_development as e1_budget
from lewm import navigation_capability_active_wall_development as wall
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT, closeout

SOURCE = 'v4_completed_support_screen_C1_dev00_ep0_attempt001'
ASSIGNMENT = 'e1_budget_replay_C1_dev00_ep0_attempt002'
CHECK = 'e1_budget_replay_check_v2'
ARM, MAZE, EPISODE = 'C1', 0, 0
REPLAY_CAP_HOURS = 2.
ALLOWED = {
    'config.json': {('assignment',)},
    'process.json': {('pid',), ('created',)},
    'pose_worker_identity.json': {('pid',)},
    'result.json': {('wall_s',), ('peak_process_tree_rss_bytes',), ('peak_device_used_bytes',)},
    'native/in_memory_camera_observations.json': {('frames', '*', 'acquisition_wall_ms')},
    'planning.json': {('*', 'clearance_preferred_route', 'added_routing_s')},
    'progress.jsonl': {('*', 'wall_s')},
    'episode_evaluation.json': {('wall_s',), ('wall_seconds_per_simulated_second',), ('articulated_reader_wall_s',),
                                ('decision_latency_s', 'median'), ('decision_latency_s', 'p95'),
                                ('input_sha256', 'config.json'), ('input_sha256', 'result.json'), ('input_sha256', 'planning.json')},
}
UNORDERED = {'stage_timings.json': 'wall_ns'}  # parallel service stages; compared as a multiset without wall time
DECISION_RECORDS = ('requests.json', 'model_calls.json', 'mission.json', 'poses.json', 'acquisitions.json', 'planning.json')
NOT_COMPARED = {'worker.log'}  # free-text log of the owner process


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield path+(key,)
            else:
                yield from differences(a[key], b[key], path+(key,))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path+('<length>',)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, path+(i,))
    elif a != b or type(a) is not type(b):
        yield path


def allowed(name, path):
    """A rule's '*' matches any list index."""
    return any(len(rule) == len(path) and all(r == p or (r == '*' and isinstance(p, int)) for r, p in zip(rule, path))
               for rule in ALLOWED.get(name, ()))


def load(path):
    text = path.read_text().replace(SOURCE, ASSIGNMENT)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def compare(original, replay):
    files = sorted(p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file())
    replayed = sorted(p.relative_to(replay).as_posix() for p in replay.rglob('*') if p.is_file())
    rows, failures = [], []
    for name in sorted(set(files) | set(replayed)):
        if name in NOT_COMPARED:
            rows.append(dict(file=name, compared=False))
            continue
        if name not in files or name not in replayed:
            failures.append(dict(file=name, reason='present in only one run'))
            continue
        a, b = original/name, replay/name
        if name in UNORDERED:
            def strip(rows):
                return sorted(json.dumps({k: v for k, v in r.items() if k != UNORDERED[name]}, sort_keys=True) for r in rows)
            x, y = load(a), load(b)
            same = strip(x) == strip(y)
            order = sum(json.dumps({k: v for k, v in r.items() if k != UNORDERED[name]}, sort_keys=True) !=
                        json.dumps({k: v for k, v in q.items() if k != UNORDERED[name]}, sort_keys=True) for r, q in zip(x, y))
            rows.append(dict(file=name, multiset_identical_without=UNORDERED[name], entries=len(x), order_differences=order))
            if not same:
                failures.append(dict(file=name, reason='entries differ beyond order and wall time'))
        elif a.suffix in ('.json', '.jsonl'):
            diffs = [list(p) for p in differences(load(a), load(b))]
            bad = [d for d in diffs if not allowed(name, tuple(d))]
            rows.append(dict(file=name, identical=not diffs, allowed_differences=len(diffs)-len(bad), disallowed=bad[:20], disallowed_count=len(bad)))
            if bad:
                failures.append(dict(file=name, disallowed=bad[:20], count=len(bad)))
        elif a.suffix == '.npz':
            x, y = np.load(a), np.load(b)
            bad = sorted(set(x.files) ^ set(y.files)) + [k for k in sorted(set(x.files) & set(y.files))
                                                          if x[k].dtype != y[k].dtype or x[k].shape != y[k].shape or x[k].tobytes() != y[k].tobytes()]
            rows.append(dict(file=name, arrays=len(x.files), bitwise_identical=not bad, differing=bad))
            if bad:
                failures.append(dict(file=name, differing=bad))
        else:
            same = sha(a) == sha(b)
            rows.append(dict(file=name, sha256_identical=same))
            if not same:
                failures.append(dict(file=name, reason='sha256 differs'))
    return rows, failures


def stop_behaviour(base, scratch):
    """The swap disables only the calendar condition and enforces the running-time cap."""
    protocol = json.loads(owner.PROTOCOL.read_text())
    scratch.mkdir()
    owner.save(scratch/'wall_budget_origin.json', dict(started_unix_s=0., cap_hours=160, basis='scratch root: calendar window long past'))
    now = time.time()
    with (scratch/wall.LEDGER).open('x') as stream:
        stream.write(json.dumps(dict(job='E1 scratch mission', id='a', event='start', unix_s=now-3600))+'\n')
    rows = {}
    for name, factory in (('owner_budget', lambda: owner.Budget(scratch, protocol)),
                          ('running_time_cap_2h', lambda: e1_budget.running_time_budget(2.)(scratch, protocol)),
                          ('running_time_cap_0.5h', lambda: e1_budget.running_time_budget(.5)(scratch, protocol))):
        try:
            factory()
            rows[name] = 'no stop'
        except owner.ResourceStop as stop:
            rows[name] = 'ResourceStop: '+str(stop)
    passed = (rows['owner_budget'].startswith('ResourceStop: 160-hour') and rows['running_time_cap_2h'] == 'no stop'
              and rows['running_time_cap_0.5h'].startswith('ResourceStop: E1 running-time cap'))
    return dict(passed=passed, cases=rows, e1_running_hours_on_scratch=e1_budget.running_hours(scratch))


def run_replay():
    """Child process: the frozen owner with only Budget swapped."""
    bind(owner.run, Budget=e1_budget.running_time_budget(REPLAY_CAP_HOURS))(ARM, MAZE, EPISODE, ASSIGNMENT)


def main():
    assert Path.cwd().resolve() == owner.REPO and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
    base = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    output.install(base)
    root = base/CHECK
    root.mkdir(exist_ok=False)
    original = base/'runs'/SOURCE
    assert json.loads((original/'config.json').read_text())['harness_sha256'] == owner.sha(owner.FREEZE)
    owner.save(root/'plan.json', dict(schema='e1_budget_replay_check.v1', source=SOURCE, assignment=ASSIGNMENT,
        source_role='dev_tune (V4 C1 screen)', swapped=['Budget'], replay_cap_hours=REPLAY_CAP_HOURS,
        allowed_differences={k: sorted(map(list, v)) for k, v in ALLOWED.items()}, not_compared=sorted(NOT_COMPARED),
        unordered=UNORDERED, previous_attempt='e1_budget_replay_check_v1 (failed as declared; preserved)',
        script_sha256=sha(__file__), budget_module_sha256=sha('lewm/e1_running_time_budget_development.py'),
        harness_sha256=owner.sha(owner.FREEZE), original_result_sha256=sha(original/'result.json')))
    behaviour = stop_behaviour(base, root/'scratch_root')
    owner.save(root/'stop_behaviour.json', behaviour)
    with wall.job(base, 'E1 budget replay check'):
        with (root/'replay.log').open('x') as log:
            code = subprocess.run([sys.executable, __file__, '--replay'], stdout=log, stderr=subprocess.STDOUT,
                                  env=os.environ | ENVIRONMENT).returncode
        row = closeout(base, root, ASSIGNMENT, code)
        rows, failures = compare(original, base/'runs'/ASSIGNMENT)
    decisions = {name: dict(entries=len(load(original/name)), allowed_differences=next(r['allowed_differences'] for r in rows if r['file'] == name),
                            disallowed=next(r['disallowed_count'] for r in rows if r['file'] == name)) for name in DECISION_RECORDS}
    result = dict(passed=behaviour['passed'] and code == 0 and not failures, stop_behaviour=behaviour, replay_exit_code=code, decision_records=decisions,
                  replay_row=row, files=rows, failures=failures, compared_files=sum(r.get('compared', True) for r in rows))
    owner.save(root/'result.json', result)
    print(json.dumps(dict(passed=result['passed'], stop_behaviour=behaviour['passed'], failures=failures[:5],
                          compared_files=result['compared_files'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', action='store_true')
    if p.parse_args().replay:
        run_replay()
    else:
        main()
