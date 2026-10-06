"""Mission entry for pinned launches, v3: the next harness version (reserve exit and C2's aligned clearance).

Started by scripts/launch_go2_dev_cohort_pinned_v3_development.py with the launch pin in LEWM_LAUNCH_PIN and
LEWM_HARNESS=reserve_exit_v1. Like the v1/v2 entries (unchanged), it re-hashes every pinned runtime file before
importing anything from the repository and refuses on mismatch. It then adds this version's mixins
(lewm/dev_harness_reserve_exit_development.py) outermost to the composed runtime: the reserve exit for every
controller, and the nominal-path clearance check for C2. It runs the ordinary mission entry
(scripts/run_go2_dev_mission_development.py) unchanged and records the pin and harness version in
runs/<assignment>/launch_pin.json.
"""
import json
import os
from pathlib import Path
import runpy
import sys

from scripts.run_go2_dev_mission_pinned_development import ENTRY, REPO, verify

HARNESSES = ('reserve_exit_v1',)


def main():
    pin = json.loads(os.environ['LEWM_LAUNCH_PIN'])
    verify(pin)
    harness = os.environ.get('LEWM_HARNESS')
    if harness not in HARNESSES:
        raise SystemExit(f'unknown harness version: {harness!r}')
    arm = sys.argv[sys.argv.index('--controller')+1]
    from lewm import dev_harness_fixes_development as fixes
    from lewm import dev_harness_reserve_exit_development as version
    if version.HARNESS != harness:
        raise SystemExit(f'harness module is {version.HARNESS}, launch asked for {harness}')
    original = fixes.compose

    def compose(fix_names, base, extra=()):
        return original(fix_names, base, extra=tuple(version.mixins_for(arm))+tuple(extra))
    fixes.compose = compose  # this process only; the entry imports it by name
    assignment = sys.argv[sys.argv.index('--assignment')+1]
    sys.argv = [str(REPO/ENTRY)]+sys.argv[1:]
    try:
        runpy.run_path(str(REPO/ENTRY), run_name='__main__')
    finally:
        try:  # never mask the mission's own outcome
            from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
            run = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])/'runs'/assignment
            if run.exists() and not (run/'launch_pin.json').exists():
                owner.save(run/'launch_pin.json', dict(pin, harness=harness, mixins=[m.__name__ for m in version.mixins_for(arm)],
                                                       entry=ENTRY, pinned_entry='scripts/run_go2_dev_mission_pinned_v3_development.py'))
        except Exception as error:
            print(f'launch pin record failed: {error!r}', flush=True)


if __name__ == '__main__':
    main()
