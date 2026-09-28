"""Capability qualification on the frozen V4 harness after its C0 gate passed.

Fixed reduced design (fixed before any validation outcome): C1-C4 on 10/0-29/0,
C0 on 10/0-19/0, one fresh attempt each, the unchanged run owner and readers,
the C0 prefix erratum at closeout. Controller failures are results; a C0
validity problem, technical failure or safety violation stops new launches.
Concurrency is output-preserving (handoff 5.7) and admitted by free memory.
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
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import (
    ENVIRONMENT, RUN_OWNER, GateStop, closeout)

GIB = 1024**3
# Admission estimates from measured peaks (pilot and V4 runs) plus margin.
MEMORY_GIB = dict(C0=10, C1=9, C2=9, C3=13, C4=12)
HEADROOM_GIB = 8
LANE_LIMITS = dict(C3=2)


def assignments(protocol):
    fixed = protocol['fixed_validation']
    rows = [(arm, int(i.split('/')[0]), f'v4_completed_support_validation_{arm}_val{int(i.split("/")[0]):02d}_ep0_attempt001')
            for arm in ('C3', 'C4', 'C0', 'C2', 'C1') for i in (fixed['C0_ids'] if arm == 'C0' else fixed['ids'])]
    assert all(i.endswith('/0') for i in fixed['ids']) and len(rows) == 90
    return rows


def run(workers, resume):
    assert Path.cwd().resolve() == owner.REPO and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    gate = json.loads((base/'cohorts/v4_completed_support_C0_gate/result.json').read_text())
    assert gate['passed'] and gate['harness_sha256'] == owner.sha(owner.FREEZE)
    root = base/'cohorts/v4_completed_support_validation'
    plan = assignments(protocol)
    if resume:
        assert (root/'config.json').exists()
        owner.save(root/f'resume_{int(time.time())}.json', dict(owner_sha256=owner.sha(__file__),
            closed=[a for _, _, a in plan if (root/f'{a}_result.json').exists()], automatic_retry=False))
    else:
        root.mkdir(parents=True, exist_ok=False)
        owner.save(root/'config.json', dict(assignments=plan, harness_sha256=owner.sha(owner.FREEZE),
            gate_result_sha256=owner.sha(base/'cohorts/v4_completed_support_C0_gate/result.json'),
            owner_sha256=owner.sha(__file__), protocol_sha256=owner.sha(owner.PROTOCOL), run_owner_sha256=owner.sha(RUN_OWNER),
            concurrent_owners=workers, lane_limits=LANE_LIMITS, memory_admission_gib=MEMORY_GIB, headroom_gib=HEADROOM_GIB,
            label='Capability qualification, not paper benchmark results', automatic_retry=False))
    rows, stops, running, lock = {}, [], {}, threading.Lock()
    pending = []
    for arm, maze, assignment in plan:
        if (root/f'{assignment}_result.json').exists():
            rows[assignment] = json.loads((root/f'{assignment}_result.json').read_text()) | dict(controller=arm)
        elif (base/'runs'/assignment).exists():
            code = 0 if json.loads((base/'runs'/assignment/'result.json').read_text())['error'] is None else 1
            rows[assignment] = closeout(base, root, assignment, code) | dict(controller=arm)
        else:
            pending.append((arm, maze, assignment))

    def attempt(arm, maze, assignment):
        try:
            with (root/f'{assignment}.log').open('x') as log:
                code = subprocess.run([sys.executable, RUN_OWNER, '--controller', arm, '--maze', str(maze), '--episode', '0',
                    '--assignment', assignment], stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT).returncode
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

    with wall.job(base, 'validation qualification'):
        started = time.monotonic()
        while True:
            launched = False
            with lock:
                if stops or not pending:
                    if not running:
                        break
                    candidate = None
                else:
                    lanes = {}
                    for arm in running.values():
                        lanes[arm] = lanes.get(arm, 0)+1
                    candidate = next((p for p in pending if lanes.get(p[0], 0) < LANE_LIMITS.get(p[0], workers)), None)
                    if candidate and len(running) < workers:
                        need = (MEMORY_GIB[candidate[0]]+HEADROOM_GIB)*GIB
                        if psutil.virtual_memory().available >= need:
                            owner.Budget(base, protocol).check(force=True)
                            pending.remove(candidate)
                            running[candidate[2]] = candidate[0]
                            threading.Thread(target=attempt, args=candidate, daemon=False).start()
                            launched = True
            # After a launch, let the new owner allocate before the next admission.
            time.sleep(20 if launched else 5)
        ordered = [rows[a] for _, _, a in plan if a in rows]
        summary = {}
        for arm in ('C0', 'C1', 'C2', 'C3', 'C4'):
            arm_rows = [r for r in ordered if r['controller'] == arm]
            summary[arm] = dict(episodes=len(arm_rows), round_trips=sum(r['round_trip_success'] for r in arm_rows),
                contacts=sum(r['disallowed_contacts'] for r in arm_rows), hard=sum(r['hard_violations'] for r in arm_rows),
                hard_unresolved=sum(r['hard_unresolved'] for r in arm_rows))
        owner.save(root/f'result_{int(time.time())}.json' if resume else root/'result.json', dict(
            harness='v4_completed_support', harness_sha256=owner.sha(owner.FREEZE), complete=len(ordered) == len(plan),
            episodes=len(ordered), summary=summary, rows=ordered, stops=stops, wall_s=time.monotonic()-started,
            label='Capability qualification, not paper benchmark results'))
        print(json.dumps(dict(complete=len(ordered) == len(plan), summary=summary, stops=stops,
            hours_used=round(wall.hours_used(base), 2))), flush=True)
    if stops:
        raise GateStop('; '.join(stops))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.workers, args.resume)
