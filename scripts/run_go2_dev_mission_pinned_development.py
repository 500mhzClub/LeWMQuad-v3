"""Mission entry for pinned launches (development process, Andrew 2 October 2026).

Started by scripts/launch_go2_dev_cohort_pinned_development.py, which passes the launch pin in
LEWM_LAUNCH_PIN: the commit and the sha256 of every runtime file, recorded from a clean committed
tree. Before importing anything from the repository it re-hashes those files and refuses to run
on any mismatch, so a batch cannot mix code. With LEWM_SEEDED_PESSIMISTIC=1 the pessimistic-
unknown rule uses the start disc seeded at the operating precondition
(lewm/dev_pessimistic_unknown_seeded_development.py). It then runs the ordinary mission entry
(scripts/run_go2_dev_mission_development.py) unchanged and records the pin in
runs/<assignment>/launch_pin.json.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/run_go2_dev_mission_development.py'


def verify(pin):
    bad = [path for path, digest in pin['files'].items() if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != digest]
    if bad:
        raise SystemExit(f"launch pin mismatch against commit {pin['commit']}: {bad}")


def main():
    pin = json.loads(os.environ['LEWM_LAUNCH_PIN'])
    verify(pin)
    seeded = os.environ.get('LEWM_SEEDED_PESSIMISTIC') == '1'
    if seeded:
        from lewm import dev_harness_fixes_development as fixes
        from lewm.dev_pessimistic_unknown_seeded_development import seeded_pessimistic_unknown_mixin
        fixes.pessimistic_unknown_mixin = seeded_pessimistic_unknown_mixin  # this process only; the entry imports it by name
    assignment = sys.argv[sys.argv.index('--assignment')+1]
    sys.argv = [str(REPO/ENTRY)]+sys.argv[1:]
    try:
        runpy.run_path(str(REPO/ENTRY), run_name='__main__')
    finally:
        try:  # never mask the mission's own outcome
            from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
            run = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])/'runs'/assignment
            if run.exists() and not (run/'launch_pin.json').exists():
                owner.save(run/'launch_pin.json', dict(pin, seeded_pessimistic_start=seeded, entry=ENTRY))
        except Exception as error:
            print(f'launch pin record failed: {error!r}', flush=True)


if __name__ == '__main__':
    main()
