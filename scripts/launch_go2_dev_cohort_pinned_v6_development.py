"""Launch a development cohort from a pinned commit, v6: v5 plus stage 2 of the dynamics experiment (Andrew, 5 October 2026).

As scripts/launch_go2_dev_cohort_pinned_v5_development.py (unchanged), with entry
scripts/run_go2_dev_mission_pinned_v6_development.py:
- --dynamics also accepts patches2:MU:marked|unmarked (straight-segment strips, lewm/dev_dynamics_patches_v2_development.py);
- the plan may name controller C1A (adaptive-travel C1, lewm/dev_c1_adaptive_travel_development.py), a CPU arm with C1's
  memory budget.
The new modules and the v6 entry and launcher are added to the pinned runtime files.
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
                 'scripts/launch_go2_dev_cohort_pinned_v5_development.py', 'lewm/dev_dynamics_patches_v2_development.py',
                 'lewm/dev_c1_adaptive_travel_development.py', 'scripts/run_go2_dev_mission_pinned_v6_development.py',
                 'scripts/launch_go2_dev_cohort_pinned_v6_development.py')


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
    p.add_argument('--dynamics', help='friction:MU, patches:MU:marked|unmarked or patches2:MU:marked|unmarked (stage 2, v2 placement)')
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
        valid = (kind == 'friction' and not mark) or (kind in ('patches', 'patches2') and mark in ('marked', 'unmarked'))
        if not valid or not .05 <= float(mu) <= 1.:
            p.error('--dynamics friction:MU or patches[2]:MU:marked|unmarked, with 0.05 <= MU <= 1')
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
    cohort.ENTRY = 'scripts/run_go2_dev_mission_pinned_v6_development.py'
    cohort.MEMORY_GIB.setdefault('C1A', cohort.MEMORY_GIB['C1'])  # this process only
    cohort.main(a.name, json.loads(a.plan), fixes_for(a.recovery), a.c3_decoder, a.c4_weights, a.workers, a.c3_lanes, False,
                a.recovery)


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt):
        main()
