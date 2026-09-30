"""Synthetic check of the scripted back-up (development).

A wall 0.42 m ahead, observed floor behind: a requested back-up must run for
BACKUP_DECISIONS decisions as `hold` with a reverse command, and the stored plan must carry
the reverse command. With no observed floor behind, or a wall close behind, it must abort.
"""
import math
import types

import numpy as np

from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.dev_harness_fixes_development import BACKUP_DECISIONS, BACKUP_SPEED_MPS, ScriptedBackupMixin
from lewm.fine_stored_obstacle_routing_development import FINE_CELL_M
from lewm.observed_floor_waypoint_development import CELL_M


class Base:
    def __init__(self):
        self.stored = []

    def _select_clear_prediction(self, *args):
        return dict(action='left_turn', action_index=4, requested_command=[0., 0., .45], command_duration_ns=400_000_000)

    def _store_plan(self, plan, completed, prefix):
        self.stored.append(plan)


class T(ScriptedBackupMixin, Base):
    pass


def world(behind_floor=True, wall_behind=None):
    ys = np.arange(-1., 1., FINE_CELL_M)
    cells = {(int(math.floor(.42/FINE_CELL_M)), int(math.floor(y/FINE_CELL_M))) for y in ys}
    if wall_behind is not None:
        cells |= {(int(math.floor(-wall_behind/FINE_CELL_M))-1, int(math.floor(y/FINE_CELL_M))) for y in ys}
    xs = range(int(-1/CELL_M) if behind_floor else 0, int(.4/CELL_M))
    floor = {(i, j) for i in xs for j in range(int(-.5/CELL_M), int(.5/CELL_M))}
    return types.SimpleNamespace(fine_occupied=frozenset(cells), floor=frozenset(floor))


def run(snapshot, steps):
    t = T()
    t._request_backup('test')
    out = []
    for k in range(steps):
        r = t._select_clear_prediction(None, None, snapshot, np.array([-.02*k, 0., 0.]), np.eye(3))
        if t._backup_store:
            t._store_plan(ScheduledCommand(1000*k, 1000*k+300, 1000*k+700, (0., 0., 0.)), 0, None)
        out.append(r)
    return t, out


def main():
    t, out = run(world(), BACKUP_DECISIONS+1)
    assert all(r['action'] == 'hold' and r['requested_command'][0] == -BACKUP_SPEED_MPS for r in out[:BACKUP_DECISIONS]), out
    assert out[BACKUP_DECISIONS]['action'] == 'left_turn' and 'dev_backup' not in out[BACKUP_DECISIONS]
    assert [p.command for p in t.stored] == [(-BACKUP_SPEED_MPS, 0., 0.)]*BACKUP_DECISIONS
    _, out = run(world(behind_floor=False), 2)
    assert 'dev_backup_aborted' in out[0] and out[0]['action'] == 'left_turn', out[0]
    _, out = run(world(wall_behind=.41), 2)
    assert 'dev_backup_aborted' in out[0], out[0]
    print('back-up runs', BACKUP_DECISIONS, 'steps on observed floor; aborts with unobserved floor or a wall behind')


if __name__ == '__main__':
    main()
