"""Run a development cohort of missions and report contacts and wall clearance on every run.

Development mode (30 Sep 2026). Missions go through `run_go2_dev_mission_development.py`. The
frozen readers classify each mission; nothing gates or stops the cohort except storage (the
12-GiB reserve). Each row reports round trip, disallowed contacts, hard and operating clearance
violations, minimum wall separation, hold rates, the frozen failure taxonomy and wall time.

Plan format (JSON list): [[controller, set, maze, episode], ...].
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT, closeout

GIB = 1024**3
MEMORY_GIB = dict(C0=10, C1=9, C2=9, C3=13, C4=12)
ENTRY = 'scripts/run_go2_dev_mission_development.py'


def row_for(base, assignment, arm, code):
    destination = base/'runs'/assignment
    ev = json.loads((destination/'episode_evaluation.json').read_text())
    s = ev['safety']
    stall = ev['stall_by_phase']
    return dict(assignment=assignment, controller=arm, round_trip=ev['round_trip_success'], beacon=ev['beacon_success'], home=ev['home_success'],
                contacts=ev['disallowed_contact_samples'], hard=s['hard']['confirmed_violation_samples'], hard_unresolved=s['hard']['unresolved_sampled_samples'],
                operating=s['operating']['confirmed_violation_samples'], min_clearance_m=s['hard']['minimum_separation_lower_m'],
                outbound_hold_rate=stall.get('OUTBOUND', {}).get('rate'), return_hold_rate=stall.get('RETURN', {}).get('rate'),
                source_error=ev['source_error'], taxonomy=ev['failure_and_stall_taxonomy'], wall_s=ev['wall_s'], exit=code)


def main(name, plan, fixes, c3_decoder, c4_weights, workers, c3_lanes, allow_final_round):
    assert Path.cwd().resolve() == owner.REPO
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    root = base/'dev_cohorts'/name
    root.mkdir(parents=True, exist_ok=False)
    jobs = [(arm, set_name, maze, episode, f'dev_{name}_{arm}_{set_name}{maze:02d}_ep{episode}') for arm, set_name, maze, episode in plan]
    owner.save(root/'config.json', dict(mode='development', plan=jobs, fixes=fixes, c3_decoder=c3_decoder, c4_weights=c4_weights,
                                        workers=workers, c3_lanes=c3_lanes))
    rows, running, lock = {}, {}, threading.Lock()
    pending = list(jobs)

    def attempt(arm, set_name, maze, episode, assignment):
        try:
            command = [sys.executable, ENTRY, '--controller', arm, '--set', set_name, '--maze', str(maze), '--episode', str(episode),
                       '--assignment', assignment, '--fixes', ','.join(fixes)]
            command += ['--c3-decoder', c3_decoder] if c3_decoder else []
            command += ['--c4-weights', c4_weights] if c4_weights else []
            command += ['--allow-final-round'] if allow_final_round else []
            with (root/f'{assignment}.log').open('x') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT).returncode
            try:
                closeout(base, root, assignment, code)
                row = row_for(base, assignment, arm, code)
            except Exception as exc:  # development: record and continue
                row = dict(assignment=assignment, controller=arm, error=repr(exc)[:500], exit=code)
            with lock:
                rows[assignment] = row
            print(json.dumps(row), flush=True)
        finally:
            with lock:
                running.pop(assignment, None)

    with wall.job(base, f'dev cohort {name}'):
        while True:
            with lock:
                if not pending and not running:
                    break
                lanes = sum(1 for a in running.values() if a == 'C3')
                candidate = next((j for j in pending if j[0] != 'C3' or lanes < c3_lanes), None)
                free_ok = shutil.disk_usage(base).free > 14*GIB
                mem_ok = candidate and psutil.virtual_memory().available >= (MEMORY_GIB[candidate[0]]+8)*GIB
                if candidate and len(running) < workers and free_ok and mem_ok:
                    pending.remove(candidate)
                    running[candidate[4]] = candidate[0]
                    threading.Thread(target=attempt, args=candidate).start()
                elif not free_ok and not running:
                    print(json.dumps(dict(stop='storage reserve')), flush=True)
                    break
            time.sleep(10)
    ordered = [rows[j[4]] for j in jobs if j[4] in rows]
    owner.save(root/'result.json', dict(mode='development', rows=ordered))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--plan', required=True, help='JSON list of [controller, set, maze, episode]')
    p.add_argument('--fixes', default='')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--c3-lanes', type=int, default=2)
    p.add_argument('--allow-final-round', action='store_true')
    a = p.parse_args()
    main(a.name, json.loads(a.plan), [f for f in a.fixes.split(',') if f], a.c3_decoder, a.c4_weights, a.workers, a.c3_lanes,
         a.allow_final_round)
