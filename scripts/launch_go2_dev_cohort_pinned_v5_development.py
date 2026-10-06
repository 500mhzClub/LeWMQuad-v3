"""Launch a development cohort from a pinned commit, v5: harness versions up to reserve_exit_v2 plus a dynamics perturbation
(Andrew, 4 October 2026).

A copy of scripts/launch_go2_dev_cohort_pinned_v4_development.py (unchanged) that runs missions through
scripts/run_go2_dev_mission_pinned_v5_development.py and adds --dynamics friction:MU (stage 1 of the
dynamics experiment: uniform floor friction on the collision floor and all robot geometries,
lewm/dev_dynamics_friction_development.py).

Process rule (after the 2 October import incident, commit 6be6d83a): live harness code is never edited while missions
can launch; new behaviour is developed in new files, and every batch launches through a pinned launcher, which
1. refuses unless each runtime file is tracked and identical to HEAD (a git diff on exactly these paths, never a
   recursive scan of the tree);
2. records the commit and the sha256 of every runtime file in dev_cohorts/<name>_launch_pin.json and in the cohort's
   config.json and result.json;
3. runs every mission through the pinned entry, which re-verifies the hashes before importing anything from the
   repository;
4. serialises appends to the shared wall-clock ledger with a file lock.
A git worktree or clone is not used: AGENTS.md forbids worktree or checkout copies while legacy sealed blobs remain
tracked.

Usage: --name, --plan (JSON list of [controller, set, maze, episode]), --harness, --recovery, --c3-decoder,
--c4-weights, --workers, --c3-lanes.
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
                 'docs/go2_navigation_calibrated_margins_2026-10-02.json', 'lewm/dev_pessimistic_unknown_lookaround_development.py',
                 'scripts/run_go2_dev_mission_pinned_v2_development.py', 'scripts/launch_go2_dev_cohort_pinned_v2_development.py',
                 'lewm/dev_harness_reserve_exit_development.py', 'scripts/run_go2_dev_mission_pinned_v3_development.py',
                 'scripts/launch_go2_dev_cohort_pinned_v3_development.py', 'lewm/dev_dynamics_friction_development.py',
                 'scripts/run_go2_dev_mission_pinned_v4_development.py', 'scripts/launch_go2_dev_cohort_pinned_v4_development.py',
                 'lewm/dev_harness_reserve_exit_v1_1_development.py', 'lewm/dev_dynamics_patches_development.py',
                 'lewm/dev_harness_reserve_exit_v2_development.py', 'scripts/run_go2_dev_mission_pinned_v5_development.py',
                 'scripts/launch_go2_dev_cohort_pinned_v5_development.py')


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
    p.add_argument('--harness', choices=('reserve_exit_v1', 'reserve_exit_v1_1', 'reserve_exit_v2'), required=True)
    p.add_argument('--dynamics', help='friction:MU (uniform floor friction, stage 1) or patches:MU:marked|unmarked (stage 2)')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--c3-lanes', type=int, default=2)
    a = p.parse_args()
    launch = pin()
    os.environ['LEWM_LAUNCH_PIN'] = json.dumps(launch)
    os.environ['LEWM_HARNESS'] = a.harness
    if a.dynamics:
        kind, _, rest = a.dynamics.partition(':')
        mu, _, mark = rest.partition(':')
        valid = (kind == 'friction' and not mark) or (kind == 'patches' and mark in ('marked', 'unmarked'))
        if not valid or not .05 <= float(mu) <= 1.:
            p.error('--dynamics friction:MU or patches:MU:marked|unmarked, with 0.05 <= MU <= 1')
        os.environ['LEWM_DYNAMICS'] = a.dynamics
    from lewm import decision_headroom_json_v42_development as output
    from lewm import navigation_capability_active_wall_development as wall
    from lewm.dev_harness_fixes_development import fixes_for
    from scripts import run_go2_dev_cohort_development as cohort
    base = Path(json.loads(cohort.owner.PROTOCOL.read_text())['output_root'])
    output.install(base)
    record = dict(launch, harness=a.harness, dynamics=a.dynamics, arguments=vars(a))
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
    cohort.ENTRY = 'scripts/run_go2_dev_mission_pinned_v5_development.py'
    cohort.main(a.name, json.loads(a.plan), fixes_for(a.recovery), a.c3_decoder, a.c4_weights, a.workers, a.c3_lanes, False,
                a.recovery)


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt):
        main()
