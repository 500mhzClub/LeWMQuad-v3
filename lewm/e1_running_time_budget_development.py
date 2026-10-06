"""E1 running-time budget (Andrew's ruling of 29 September 2026, item 3).

The frozen run owner's `Budget` stops every mission 160 calendar hours after
`wall_budget_origin.json`. That cap covers the capability brief, not E1. For E1 the calendar
condition is replaced by E1's own running-time cap: the union of E1 job intervals in the
active-wall ledger (job names starting with 'E1 '), open jobs included.

Every other check is the owner's code, unchanged: filesystem reserves, peak process-tree RSS,
the VRAM reserve and closeout admission. The class is swapped into the frozen owner's `run`
with `bind`; no harness file is edited. Budget raises stops and records peaks; no decision
reads it.
"""
import json
import math
import time

from lewm import navigation_capability_active_wall_development as wall
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

PREFIX = 'E1 '
CLOSEOUT_RESERVE_S = 120
LEDGER_PERIOD_S = 30.


def running_hours(root, prefix=PREFIX, now=None):
    """Union of the ledger's intervals for jobs named with `prefix`, open jobs included."""
    total, end = 0., None
    for start, stop, name in wall.intervals(root, now):
        if not name.startswith(prefix):
            continue
        if end is None or start > end:
            total += max(0., stop-start)
            end = stop
        elif stop > end:
            total += stop-end
            end = stop
    return total/3600


def running_time_budget(cap_hours, prefix=PREFIX):
    """A drop-in `Budget` for the frozen owner with the calendar stop replaced."""

    class RunningTimeBudget(owner.Budget):
        running_time_cap_hours = cap_hours

        def __init__(self, root, protocol):
            self.ledger_checked = -math.inf
            super().__init__(root, protocol)
            self.calendar_origin_unix_s = json.loads((root/'wall_budget_origin.json').read_text())['started_unix_s']

        def check(self, force=False):
            now = time.monotonic()
            if force or now-self.ledger_checked >= LEDGER_PERIOD_S:
                self.ledger_checked = now
                if running_hours(self.root, prefix)*3600 >= cap_hours*3600-CLOSEOUT_RESERVE_S:
                    raise owner.ResourceStop(f'E1 running-time cap ({cap_hours} h) closeout reserve reached')
            # The owner's calendar condition is time.time()-origin >= 160 h; an infinite origin
            # disables only that line. Everything after it in the owner's check runs unchanged.
            self.origin = math.inf
            super().check(force)

    return RunningTimeBudget
