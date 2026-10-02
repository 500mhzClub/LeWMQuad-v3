"""Re-run missions that crashed at import before simulating (development incident, 2 October 2026).

Editing lewm/dev_harness_fixes_development.py while cohort lanes were launching missions broke
the mission entry's import for about a minute (commit 6be6d83a records it). Three sensitivity
missions crashed with ImportError before any simulation, leaving only a 502-byte log, no run
directory and no per-mission result. They are infrastructure failures, not results, so each is
re-run under its original assignment with the cohort's own settings (from its config.json),
then read and closed out exactly as the cohort runner does.

`run`: precondition (no run directory, the cohort log shows the ImportError), memory and disk
gates as in the cohort runner, mission log to <assignment>.repair.log, row to
<assignment>_repair_row.json, receipt in repair_import_failures.json.
`patch`: once the cohort has written result.json, the original is kept as
result.import_failure_original.json and the error row is replaced by the repaired row with a
`repair` note.

Usage: repair_go2_dev_cohort_import_failures_development.py run|patch
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

from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT, closeout
from scripts.run_go2_dev_cohort_development import ENTRY, GIB, MEMORY_GIB, read_dev, row_for
from lewm import decision_headroom_json_v42_development as output

CRASHED = (('sens_scale1p25', 'dev_sens_scale1p25_C1_prelim_test34_ep0'),
           ('sens_scale1p5', 'dev_sens_scale1p5_C1_prelim_test42_ep0'),
           ('sens_turnscale1p25', 'dev_sens_turnscale1p25_C1_prelim_test46_ep0'))
REPO = Path(__file__).resolve().parents[1]


def base():
    root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    output.install(root)
    return root


def command_for(config, job):
    arm, set_name, maze, episode, assignment = job
    command = [sys.executable, ENTRY, '--controller', arm, '--set', set_name, '--maze', str(maze), '--episode', str(episode), '--assignment', assignment]
    command += ['--recovery', config['recovery']] if config.get('recovery') else ['--fixes', ','.join(config['fixes'])]
    command += ['--degrade', config['forecast_degradation']] if config.get('forecast_degradation') else []
    command += ['--margin', config['clearance_margin']] if config.get('clearance_margin') else []
    command += ['--pessimistic-unknown'] if config.get('pessimistic_unknown') else []
    command += ['--c3-decoder', config['c3_decoder']] if config.get('c3_decoder') else []
    command += ['--c4-weights', config['c4_weights']] if config.get('c4_weights') else []
    return command


def repair(b, cohort, assignment):
    root = b/'dev_cohorts'/cohort
    config = json.loads((root/'config.json').read_text())
    job = next(j for j in config['plan'] if j[4] == assignment)
    original = (root/f'{assignment}.log').read_text(errors='replace')
    if (b/'runs'/assignment).exists() or 'ImportError' not in original:
        raise ValueError(f'precondition failed for {assignment}: run exists or no import failure')
    while psutil.virtual_memory().available < (MEMORY_GIB[job[0]]+8)*GIB or os.statvfs(b).f_bavail*os.statvfs(b).f_frsize < 14*GIB:
        time.sleep(10)
    command = command_for(config, job)
    began = time.time()
    with (root/f'{assignment}.repair.log').open('x') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT, cwd=REPO).returncode
    try:
        read_dev(b, root, assignment, job[0], code)
        closeout(b, root, assignment, code)
        row = row_for(b, assignment, job[0], code)
    except Exception as exc:
        row = dict(assignment=assignment, controller=job[0], error=repr(exc)[:500], exit=code)
    row['repair'] = dict(original_failure=original.strip().splitlines()[-1][:300], reason='crashed at import before simulating (incident 6be6d83a)',
                         command=command[1:], wall_s=round(time.time()-began, 1))
    (root/f'{assignment}_repair_row.json').write_text(json.dumps(row, indent=1)+'\n')
    print(json.dumps(dict(cohort=cohort, assignment=assignment, exit=code, round_trip=row.get('round_trip'), error=row.get('error'))), flush=True)


def run():
    b = base()
    threads = [threading.Thread(target=repair, args=(b, c, a)) for c, a in CRASHED]
    for t in threads:
        t.start()
        time.sleep(15)
    for t in threads:
        t.join()
    print('REPAIR_RUN_DONE', flush=True)


def patch():
    b = base()
    for cohort, assignment in CRASHED:
        root = b/'dev_cohorts'/cohort
        repaired = root/f'{assignment}_repair_row.json'
        if not (root/'result.json').exists() or not repaired.exists():
            print(cohort, 'not ready (cohort result or repair row missing)')
            continue
        result = json.loads((root/'result.json').read_text())
        backup = root/'result.import_failure_original.json'
        if not backup.exists():
            backup.write_text(json.dumps(result, indent=1)+'\n')
        row = json.loads(repaired.read_text())
        rows = [row if r['assignment'] == assignment and 'error' in r else r for r in result['rows']]
        if assignment not in {r['assignment'] for r in result['rows']}:
            rows.append(row)
        result['rows'] = rows
        (root/'result.json').write_text(json.dumps(result, indent=1)+'\n')
        print(cohort, assignment, 'patched')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('run', 'patch'))
    a = p.parse_args()
    run() if a.mode == 'run' else patch()
