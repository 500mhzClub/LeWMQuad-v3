"""Synthetic check of the C2 reactive deadlock escape (development): a wall 0.42 m away, three layouts."""
import math, types
import numpy as np
from lewm.dev_harness_fixes_development import DeadlockEscapeMixin, ESCAPE_AFTER
from lewm.persistent_visual_baselines_development import select_reactive_terminal
from lewm.fine_stored_obstacle_routing_development import cached_clearance, FINE_CELL_M

class Base:
    def __init__(self, result): self._r = result
    def _select_clear_prediction(self, *a): return dict(self._r)
class T(DeadlockEscapeMixin, Base): pass

def run(wall_side, yaw, goal):
    # wall: a line of fine cells 0.42 m from origin on the given side (map frame)
    xs = np.arange(-1.0, 1.0, FINE_CELL_M)
    d = 0.42 + FINE_CELL_M*0.0
    if wall_side == 'front': cells = {(int(math.floor(d/FINE_CELL_M)), int(math.floor(x/FINE_CELL_M))) for x in xs}
    if wall_side == 'left': cells = {(int(math.floor(x/FINE_CELL_M)), int(math.floor(d/FINE_CELL_M))) for x in xs}
    if wall_side == 'back': cells = {(int(math.floor(-d/FINE_CELL_M))-1, int(math.floor(x/FINE_CELL_M))) for x in xs}
    snap = types.SimpleNamespace(fine_occupied=frozenset(cells))
    c = cached_clearance(snap.fine_occupied).minimum(np.zeros(2), np.zeros(2))
    res = select_reactive_terminal(goal, scan_error=None, clearance_m=c, pulse=False, arrival_radius_m=.02)
    t = T(res)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    for k in range(ESCAPE_AFTER):
        out = t._select_clear_prediction(None, None, snap, np.zeros(3), R)
    print(wall_side, 'yaw', round(yaw, 2), 'clear', round(c, 3), 'frozen', res['action'], '->', out['action'], out.get('dev_deadlock_escape'))

run('front', 0.0, [1.0, 0.0])      # facing the wall, goal beyond it
run('left', 0.0, [1.0, 0.0])       # wall on the left, goal ahead
run('back', 0.0, [1.0, 0.2])       # wall behind, goal ahead
