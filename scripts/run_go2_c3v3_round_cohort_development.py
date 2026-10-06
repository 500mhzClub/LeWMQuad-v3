"""C3-v3 round cohorts (pre-declared 30 Sep 2026, commit ec2e34c9).

  --stage onpolicy: C1 on the 22 on-policy episodes (fit 0-15, held-out 16-21).
  --stage safety:   C3-v3 and C4-v3 on the 10 safety-check episodes (22-31), only after
                    C3-v3 passes offline acceptance.
One fresh attempt per assignment, frozen V4 harness and readers, memory-admitted concurrency
with at most two C3 owners. Controller failures are results; any disallowed contact or hard
violation, technical failure or closeout defect stops new launches.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT, GateStop, closeout

GIB = 1024**3
MEMORY_GIB = dict(C1=9, C3=13, C4=12)
HEADROOM_GIB, WORKERS, LANE_LIMITS = 8, 5, dict(C3=2)
ENTRY = 'scripts/run_go2_c3v3_round_development.py'


def main(stage):
    assert Path.cwd().resolve() == owner.REPO and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    if stage == 'onpolicy':
        root = base/'cohorts/c3v3_onpolicy_c1'
        plan = [('C1', maze, f'c3v3_onpolicy_C1_m{maze:02d}_ep0_attempt001') for maze in range(22)]
        versions = dict(C1='unchanged')
    else:
        assert json.loads((base/'c3v3_acceptance_v1/result.json').read_text())['passed'], 'C3-v3 did not pass offline acceptance'
        root = base/'cohorts/c3v3_safety_check'
        plan = [(arm, maze, f'c3v3_safety_{arm}_m{maze:02d}_ep0_attempt001') for arm in ('C3', 'C4') for maze in range(22, 32)]
        versions = dict(C3='C3-v3', C4='C4-v3')
    root.mkdir(parents=True, exist_ok=False)
    owner.save(root/'config.json', dict(stage=stage, assignments=plan, harness_sha256=owner.sha(owner.FREEZE), entry_sha256=owner.sha(ENTRY),
        owner_sha256=owner.sha(__file__), predeclaration_commit='ec2e34c9',
        registry_sha256=owner.sha(base/'c3v3_sets_v1/registry.json'), versions=versions,
        concurrent_owners=WORKERS, lane_limits=LANE_LIMITS, automatic_retry=False,
        label='C3-v3 round (on-policy C1 data or safety check); not validation, not E1'))
    rows, stops, running, lock = {}, [], {}, threading.Lock()
    pending = list(plan)

    def attempt(arm, maze, assignment):
        try:
            with (root/f'{assignment}.log').open('x') as log:
                code = subprocess.run([sys.executable, ENTRY, '--controller', arm, '--maze', str(maze), '--assignment', assignment],
                                      stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT).returncode
            row = closeout(base, root, assignment, code) | dict(controller=arm)
            with lock:
                rows[assignment] = row
                if row['disallowed_contacts'] or row['hard_violations']:
                    stops.append('Safety violation: '+assignment)
            print(json.dumps(dict(assignment=assignment, round_trip=row['round_trip_success'], contacts=row['disallowed_contacts'],
                                  hard=row['hard_violations'], taxonomy=row['failure_and_stall_taxonomy'])), flush=True)
        except BaseException as exc:
            with lock:
                stops.append(f'{assignment}: {exc!r}')
        finally:
            with lock:
                running.pop(assignment, None)

    with wall.job(base, f'C3-v3 round: {stage} cohort'):
        started = time.monotonic()
        while True:
            launched = False
            with lock:
                if stops or not pending:
                    if not running:
                        break
                else:
                    lanes = {}
                    for arm in running.values():
                        lanes[arm] = lanes.get(arm, 0)+1
                    candidate = next((p for p in pending if lanes.get(p[0], 0) < LANE_LIMITS.get(p[0], WORKERS)), None)
                    if candidate and len(running) < WORKERS and psutil.virtual_memory().available >= (MEMORY_GIB[candidate[0]]+HEADROOM_GIB)*GIB:
                        owner.Budget(base, protocol).check(force=True)
                        pending.remove(candidate)
                        running[candidate[2]] = candidate[0]
                        threading.Thread(target=attempt, args=candidate).start()
                        launched = True
            time.sleep(20 if launched else 5)
        ordered = [rows[a] for _, _, a in plan if a in rows]
        summary = {arm: dict(episodes=sum(r['controller'] == arm for r in ordered), round_trips=sum(r['round_trip_success'] for r in ordered if r['controller'] == arm),
                             contacts=sum(r['disallowed_contacts'] for r in ordered if r['controller'] == arm),
                             hard=sum(r['hard_violations'] for r in ordered if r['controller'] == arm)) for arm in versions}
        owner.save(root/'result.json', dict(stage=stage, complete=len(ordered) == len(plan), summary=summary, rows=ordered, stops=stops,
                                            wall_s=time.monotonic()-started, versions=versions))
        print(json.dumps(dict(complete=len(ordered) == len(plan), summary=summary, stops=stops, hours_used=round(wall.hours_used(base), 2))), flush=True)
    if stops:
        raise GateStop('; '.join(stops))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('onpolicy', 'safety'), required=True)
    main(p.parse_args().stage)
