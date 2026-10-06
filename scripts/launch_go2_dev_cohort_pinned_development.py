"""Launch a development cohort from a pinned commit (development process, Andrew 2 October 2026).

Process rule (after the 2 October import incident, commit 6be6d83a): live harness code is never
edited while missions can launch; new behaviour is developed in new files, and every batch
launches through this script, which
1. refuses unless each runtime file is tracked and identical to HEAD (a git diff on exactly these
   paths, never a recursive scan of the tree);
2. records the commit and the sha256 of every runtime file in dev_cohorts/<name>_launch_pin.json
   and in the cohort's config.json and result.json;
3. runs every mission through scripts/run_go2_dev_mission_pinned_development.py, which re-verifies
   the hashes before importing anything from the repository;
4. serialises appends to the shared wall-clock ledger with a file lock (two cohorts launched at the
   same instant raced on it on 2 October and one failed to start).
A git worktree or clone is not used: AGENTS.md forbids worktree or checkout copies while legacy
sealed blobs remain tracked.

Usage: the cohort runner's arguments (--name, --plan, --recovery, --degrade, --margin,
--pessimistic-unknown, --c3-decoder, --c4-weights, --workers, --c3-lanes) plus
--seeded-pessimistic (requires --pessimistic-unknown).
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ('lewm/dev_harness_fixes_development.py', 'lewm/dev_pessimistic_unknown_seeded_development.py',
                 'lewm/dev_readout_variants_development.py', 'lewm/navigation_capability_active_wall_development.py',
                 'scripts/run_go2_dev_mission_development.py', 'scripts/run_go2_dev_mission_pinned_development.py',
                 'scripts/run_go2_dev_cohort_development.py', 'scripts/read_go2_dev_mission_development.py',
                 'scripts/summarise_go2_dev_cohorts_development.py', 'scripts/launch_go2_dev_cohort_pinned_development.py',
                 'docs/go2_navigation_calibrated_margins_2026-10-02.json')


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)


def pin():
    tracked = git('ls-files', '--error-unmatch', '--', *RUNTIME_FILES)
    if tracked.returncode:
        raise SystemExit('runtime files must be tracked: '+tracked.stderr.strip())
    if git('diff', '--quiet', 'HEAD', '--', *RUNTIME_FILES).returncode:
        raise SystemExit('runtime files differ from HEAD; commit before launching')
    return dict(commit=git('rev-parse', 'HEAD').stdout.strip(),
                files={p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in RUNTIME_FILES})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--recovery', choices=('on', 'off'), default='off')
    p.add_argument('--degrade')
    p.add_argument('--margin', choices=('p95', 'p99'))
    p.add_argument('--pessimistic-unknown', action='store_true')
    p.add_argument('--seeded-pessimistic', action='store_true')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--c3-lanes', type=int, default=2)
    a = p.parse_args()
    if a.seeded_pessimistic and not a.pessimistic_unknown:
        p.error('--seeded-pessimistic requires --pessimistic-unknown')
    launch = pin()
    os.environ['LEWM_LAUNCH_PIN'] = json.dumps(launch)
    if a.seeded_pessimistic:
        os.environ['LEWM_SEEDED_PESSIMISTIC'] = '1'
    from lewm import decision_headroom_json_v42_development as output
    from lewm import navigation_capability_active_wall_development as wall
    from lewm.dev_harness_fixes_development import fixes_for
    from scripts import run_go2_dev_cohort_development as cohort
    base = Path(json.loads(cohort.owner.PROTOCOL.read_text())['output_root'])
    output.install(base)
    record = dict(launch, seeded_pessimistic_start=a.seeded_pessimistic, arguments=vars(a))
    cohort.owner.save(base/'dev_cohorts'/f'{a.name}_launch_pin.json', record)
    original_append, original_save = wall._append, cohort.owner.save

    def locked_append(root, row):
        with open(Path(root)/'wall_active_ledger.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return original_append(root, row)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save(path, value, *args, **kwargs):
        if Path(path).parent.name == a.name and Path(path).name in ('config.json', 'result.json') and isinstance(value, dict):
            value = dict(value, launch_pin=record)
        return original_save(path, value, *args, **kwargs)
    wall._append, cohort.owner.save = locked_append, save
    cohort.ENTRY = 'scripts/run_go2_dev_mission_pinned_development.py'
    cohort.main(a.name, json.loads(a.plan), fixes_for(a.recovery), a.c3_decoder, a.c4_weights, a.workers, a.c3_lanes, False,
                a.recovery, a.degrade, a.margin, a.pessimistic_unknown)


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt):
        main()
