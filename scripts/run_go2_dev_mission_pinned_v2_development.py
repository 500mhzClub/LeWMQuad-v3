"""Mission entry for pinned launches, v2 (development process, Andrew 2 October 2026).

As scripts/run_go2_dev_mission_pinned_development.py (which stays unchanged while batches pinned
to it run): verifies the launch pin (LEWM_LAUNCH_PIN) before importing anything from the
repository and refuses on mismatch. Adds the pessimistic-unknown variant chosen by the launcher:
LEWM_PESSIMISTIC_VARIANT = 'seeded' (start disc at the 0.5-m precondition) or 'lookaround' (that,
with the scripted look-around exempt; lewm/dev_pessimistic_unknown_lookaround_development.py),
and LEWM_UNKNOWN_LEVEL = p95 | p99, the calibrated e_f bound for unseen cells (otherwise the
run's margin level, p99 if none). Records the pin in runs/<assignment>/launch_pin.json.
"""
import json
import os
from pathlib import Path
import runpy
import sys

from scripts.run_go2_dev_mission_pinned_development import ENTRY, REPO, verify

BOUNDS = 'docs/go2_navigation_calibrated_margins_2026-10-02.json'


def main():
    pin = json.loads(os.environ['LEWM_LAUNCH_PIN'])
    verify(pin)
    variant, level = os.environ.get('LEWM_PESSIMISTIC_VARIANT'), os.environ.get('LEWM_UNKNOWN_LEVEL')
    arm = sys.argv[sys.argv.index('--controller')+1]
    if variant:
        from lewm import dev_harness_fixes_development as fixes
        from lewm.dev_pessimistic_unknown_lookaround_development import lookaround_exempt_pessimistic_unknown_mixin
        from lewm.dev_pessimistic_unknown_seeded_development import seeded_pessimistic_unknown_mixin
        chosen = dict(seeded=seeded_pessimistic_unknown_mixin, lookaround=lookaround_exempt_pessimistic_unknown_mixin)[variant]
        bounds = json.loads((REPO/BOUNDS).read_text())['controllers'][arm]

        def factory(bound_m, run_level):
            return chosen(bounds[level]['margin_m'], level) if level else chosen(bound_m, run_level)
        fixes.pessimistic_unknown_mixin = factory  # this process only; the entry imports it by name
    assignment = sys.argv[sys.argv.index('--assignment')+1]
    sys.argv = [str(REPO/ENTRY)]+sys.argv[1:]
    try:
        runpy.run_path(str(REPO/ENTRY), run_name='__main__')
    finally:
        try:  # never mask the mission's own outcome
            from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
            run = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])/'runs'/assignment
            if run.exists() and not (run/'launch_pin.json').exists():
                owner.save(run/'launch_pin.json', dict(pin, pessimistic_variant=variant, unknown_level=level, entry=ENTRY,
                                                       pinned_entry='scripts/run_go2_dev_mission_pinned_v2_development.py'))
        except Exception as error:
            print(f'launch pin record failed: {error!r}', flush=True)


if __name__ == '__main__':
    main()
