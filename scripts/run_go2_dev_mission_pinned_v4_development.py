"""Mission entry for pinned launches, v4: the reserve_exit_v1 harness plus a dynamics perturbation (Andrew, 3 October 2026).

As scripts/run_go2_dev_mission_pinned_v3_development.py (unchanged while batches pinned to it run): re-hashes every pinned
runtime file before importing anything from the repository, then adds the harness version's mixins outermost
(reserve_exit_v1, or reserve_exit_v1_1 with C2's path as dispatched). With
LEWM_DYNAMICS=friction:MU it also wraps the owner's make_session so that the uniform floor friction of stage 1 of the
dynamics experiment is installed on the collision floor and all 27 robot geometries before any physics step
(lewm/dev_dynamics_friction_development.py). The pin, harness and dynamics are recorded in runs/<assignment>/launch_pin.json.
"""
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from scripts.run_go2_dev_mission_pinned_development import ENTRY, REPO, verify

HARNESSES = {'reserve_exit_v1': 'lewm.dev_harness_reserve_exit_development',
             'reserve_exit_v1_1': 'lewm.dev_harness_reserve_exit_v1_1_development'}


def parse_dynamics(text):
    if not text:
        return None
    kind, _, value = text.partition(':')
    if kind != 'friction':
        raise SystemExit(f'unknown dynamics perturbation: {text!r}')
    return dict(perturbation='uniform_floor_friction', mu=float(value))


def main():
    pin = json.loads(os.environ['LEWM_LAUNCH_PIN'])
    verify(pin)
    harness = os.environ.get('LEWM_HARNESS')
    if harness not in HARNESSES:
        raise SystemExit(f'unknown harness version: {harness!r}')
    dynamics = parse_dynamics(os.environ.get('LEWM_DYNAMICS'))
    arm = sys.argv[sys.argv.index('--controller')+1]
    from lewm import dev_harness_fixes_development as fixes
    version = importlib.import_module(HARNESSES[harness])
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    if version.HARNESS != harness:
        raise SystemExit(f'harness module is {version.HARNESS}, launch asked for {harness}')
    original = fixes.compose

    def compose(fix_names, base, extra=()):
        return original(fix_names, base, extra=tuple(version.mixins_for(arm))+tuple(extra))
    fixes.compose = compose  # this process only; the entry imports it by name
    if dynamics:
        from lewm.dev_dynamics_friction_development import friction_session
        owner.make_session = friction_session(owner.make_session, dynamics['mu'])  # read when the entry binds owner.run
    assignment = sys.argv[sys.argv.index('--assignment')+1]
    sys.argv = [str(REPO/ENTRY)]+sys.argv[1:]
    try:
        runpy.run_path(str(REPO/ENTRY), run_name='__main__')
    finally:
        try:  # never mask the mission's own outcome
            run = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])/'runs'/assignment
            if run.exists() and not (run/'launch_pin.json').exists():
                owner.save(run/'launch_pin.json', dict(pin, harness=harness, dynamics=dynamics,
                                                       mixins=[m.__name__ for m in version.mixins_for(arm)], entry=ENTRY,
                                                       pinned_entry='scripts/run_go2_dev_mission_pinned_v4_development.py'))
        except Exception as error:
            print(f'launch pin record failed: {error!r}', flush=True)


if __name__ == '__main__':
    main()
