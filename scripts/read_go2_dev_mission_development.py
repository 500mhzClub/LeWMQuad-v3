"""Read a development mission with the frozen episode readers, minus the programme window.

The frozen readers bind the owner's `Budget`, whose 160-hour programme window (counted in wall
time from a stored origin) would stop every read once it closes, about 03:00 on 2 October.
Development mode dropped formal budget stops, so this entry swaps in `DevBudget` (the owner's
filesystem and VRAM reserve checks unchanged, no window) and then calls the unchanged reader:
- `v4`: the V4 episode reader (C0, C1, C3, C4);
- `reactive`: the reactive-hold reader erratum (C2);
- `failure`: the controller-failure reader (for example, pose loss).
No arrival, safety, SPL or stall computation changes.
"""
import argparse
import importlib
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.run_go2_dev_mission_development import DevBudget

READERS = dict(v4='scripts.read_go2_navigation_capability_completed_support_v4_development',
               reactive='scripts.read_go2_capability_v4_reactive_holds_development',
               failure='scripts.read_go2_capability_completed_support_v4_failure_development')


def main(root, kind):
    base = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    assert root.resolve().is_relative_to((base/'runs').resolve())
    output.install(base)
    owner.Budget = DevBudget
    record = importlib.import_module(READERS[kind]).report(root)
    print({k: record.get(k) for k in ('controller', 'episode_id', 'round_trip_success', 'failure_and_stall_taxonomy')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reader', choices=tuple(READERS), required=True)
    a = p.parse_args()
    main(a.root, a.reader)
