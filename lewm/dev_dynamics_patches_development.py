"""Dynamics perturbation, stage 2: low-friction floor patches, marked or unmarked (development; Andrew, 3 October 2026).

docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md, "Stage 2". Three parts:

Placement (maze geometry and the episode's home and beacon only). Mazes are cell graphs with pitch 1.3 m and walls on
cell boundaries. The route is the shortest cell path from the home cell to the beacon cell over the layout's edges.
Patches are runs of at least two consecutive interior route cells (2.6 m along the route; the forward camera sees the
floor from about 0.9 m ahead, so the marker stays in view while the feet are on it), covering whole corridor cells so
the robot cannot avoid them by keeping to a wall. Start and beacon cells are never patched. Runs are drawn with a seed
until they cover 20-40 % of the route's cells (at least one run); the return trip crosses the same patches.

Friction field (no geometry). Genesis multiplies each geometry's base friction by a runtime per-geometry ratio
(solver.set_geoms_friction_ratio) and takes the maximum over a contact pair. The floor's ratio is set to mu_p, and each
leg's four calf geometries (the foot sphere is one of them) get ratio 1.0 off a patch and mu_p when the foot sphere is
over a patch; both are written before every 20-ms policy step (settling resets ratios, so nothing is set at build time). So floor contact is nominal off-patch and mu_p on-patch; body,
thigh and hip geometries keep 1.0 against the floor, and walls stay nominal.

Marker (visual only, geometry unchanged). The renderer draws exactly two static surfaces (floor, wall union) in a fixed
order, so the marker cannot be an extra mesh. In the marked condition the floor mesh's 12.5-cm quads whose centres lie in
a patch cell are recoloured a uniform slick_patch colour (lewm_worlds randomization palette, RGB 0.30/0.40/0.55) instead
of the floor's random greys. Vertices, faces, the identity witnesses (geometry only), depth and the draw order are
unchanged. The unmarked control has the same friction field and the floor's own colours.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

PITCH_M = 1.3
MARKER_RGB = (.30, .40, .55)
COVERAGE = (.20, .40)
MIN_RUN_CELLS = 2
_MARKERS = []  # this process's marked patch cells; read by the wrapped appearance builder
_MARKED_QUADS = []


def placement_seed(set_name, maze, episode):
    """Fixed seed rule: sha256 of "patches-v1:<set>:<maze>:<episode>", first 8 bytes."""
    return int.from_bytes(hashlib.sha256(f'patches-v1:{set_name}:{int(maze)}:{int(episode)}'.encode()).digest()[:8], 'big')


def route_cells(spec, packet):
    """Shortest cell path from the home cell to the beacon cell over the layout's edges (breadth-first, fixed order)."""
    layout = spec['evaluation_layout']
    if abs(float(layout['pitch_m'])-PITCH_M) > 1e-9:
        raise ValueError('pitch 1.3 m layouts only')
    cells = {tuple(c) for c in layout['cells']}
    neighbours = {c: [] for c in cells}
    for a, b in layout['edges']:
        neighbours[tuple(a)].append(tuple(b))
        neighbours[tuple(b)].append(tuple(a))
    to_cell = lambda xy: tuple(int(round(v/PITCH_M)) for v in xy[:2])
    start, goal = to_cell(packet['home_se2_world']), to_cell(packet['beacon_xy_world'])
    if start not in cells or goal not in cells:
        raise ValueError('home and beacon must lie in layout cells')
    previous, queue = {start: None}, deque([start])
    while queue:
        c = queue.popleft()
        for n in sorted(neighbours[c]):
            if n not in previous:
                previous[n] = c
                queue.append(n)
    if goal not in previous:
        raise ValueError('beacon cell unreachable')
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = previous[c]
    return path[::-1]


def place_patches(spec, packet, seed):
    """Seeded runs of >= 2 consecutive interior route cells covering 20-40 % of the route's cells."""
    route = route_cells(spec, packet)
    interior = list(range(1, len(route)-1))
    if len(interior) < MIN_RUN_CELLS:
        return dict(route=[list(c) for c in route], patch_cells=[], coverage=0., note='route too short for a patch run')
    rng = np.random.default_rng(seed)
    target_low, target_high = COVERAGE[0]*len(route), COVERAGE[1]*len(route)
    chosen = set()
    free = lambda i, length: all(j not in chosen and j-1 not in chosen and j+1 not in chosen for j in range(i, i+length))
    for _ in range(100):
        if len(chosen) >= max(target_low, MIN_RUN_CELLS):
            break
        fits = [n for n in (MIN_RUN_CELLS, MIN_RUN_CELLS+1) if len(chosen)+n <= max(target_high, MIN_RUN_CELLS)
                and any(i+n-1 <= interior[-1] and free(i, n) for i in interior)]
        if not fits:
            break
        length = int(rng.choice(fits))
        i = int(rng.choice([i for i in interior if i+length-1 <= interior[-1] and free(i, length)]))
        chosen.update(range(i, i+length))
    cells = [route[i] for i in sorted(chosen)]
    return dict(route=[list(c) for c in route], patch_cells=[list(c) for c in cells], coverage=len(cells)/len(route),
                seed=int(seed), pitch_m=PITCH_M, rule='runs of >= 2 consecutive interior route cells; 20-40 % of route cells')


def on_patch(xy, patch_cells):
    """Boolean per point: inside any patch cell's 1.3-m square."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not patch_cells:
        return np.zeros(len(xy), bool)
    centres = np.asarray(patch_cells, float)*PITCH_M
    return np.any(np.all(np.abs(xy[:, None, :]-centres[None]) <= PITCH_M/2, axis=2), axis=1)


def mark_floor(mesh, patch_cells):
    """Copy of the floor mesh with every quad whose centre lies in a patch cell recoloured; returns (mesh, quads)."""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) % 4 or not np.allclose(vertices[:, 2], 0.):
        raise ValueError('flat quad-per-four-vertices floor mesh required')
    centres = vertices.reshape(-1, 4, 3).mean(axis=1)[:, :2]
    inside = on_patch(centres, patch_cells)
    colours = np.asarray(mesh.visual.vertex_colors).copy()
    colours[np.repeat(inside, 4)] = np.asarray([round(255*c) for c in MARKER_RGB]+[255], np.uint8)
    marked = mesh.copy()
    marked.visual.vertex_colors = colours
    return marked, int(inside.sum())


def _install_marker_hook():
    from lewm_genesis import visible_robot_union_rgbd_scene_development as builder
    if getattr(builder.independently_seeded_union_surfaces, 'dev_patch_markers', False):
        return
    original = builder.independently_seeded_union_surfaces

    def surfaces(boxes, arm, seed):
        result = list(original(boxes, arm, seed))
        if _MARKERS:
            name, floor = result[0]
            if name != 'ground_visual':
                raise ValueError('floor-first surfaces required')
            floor, quads = mark_floor(floor, _MARKERS)
            _MARKED_QUADS[:] = [quads]
            result[0] = (name, floor)
        return result
    surfaces.dev_patch_markers = True
    builder.independently_seeded_union_surfaces = surfaces


class FrictionField:
    def __init__(self, build, patch_cells, mu):
        if not (np.isfinite(mu) and .05 <= mu <= 1.):
            raise ValueError('patch friction must lie in [0.05, 1]')
        self.build, self.patch_cells, self.mu = build, patch_cells, float(mu)
        robot = build.robot
        self.solver = robot._solver
        self.feet = [g for g in robot.geoms if g.link.name.endswith('_calf') and 'SPHERE' in repr(g.type)]
        self.legs = [[int(h.idx) for h in robot.geoms if h.link.name == g.link.name] for g in self.feet]
        self.floor = [int(g.idx) for g in build.collision_floor.geoms]
        if len(self.feet) != 4 or any(len(leg) != 4 for leg in self.legs) or len(self.floor) != 1:
            raise ValueError('four foot spheres on four calf links and one collision floor required')
        self.ids = [i for leg in self.legs for i in leg]+self.floor
        self.updates = 0
        self.on_counts = np.zeros(5, np.int64)

    def foot_xy(self):
        return np.stack([np.asarray(g.get_pos().detach().cpu().numpy() if hasattr(g.get_pos(), 'detach') else g.get_pos()).reshape(-1)[:2]
                         for g in self.feet])

    def update(self):
        on = on_patch(self.foot_xy(), self.patch_cells)
        # The floor's ratio is re-applied every step too: settling and other resets restore ratios to 1.
        self.solver.set_geoms_friction_ratio(np.asarray([self.mu if on[k] else 1. for k, leg in enumerate(self.legs) for _ in leg]
                                                        + [self.mu]*len(self.floor)), self.ids)
        self.updates += 1
        self.on_counts[int(on.sum())] += 1
        self.last_on = on  # the state applied to the coming policy step
        return on

    def receipt(self):
        floor_ratio = self.solver.get_geoms_friction_ratio(self.floor) if hasattr(self.solver, 'get_geoms_friction_ratio') else None
        return dict(foot_geoms=[int(g.idx) for g in self.feet], leg_geoms=self.legs, floor_geoms=self.floor, mu=self.mu,
                    floor_ratio=None if floor_ratio is None else np.asarray(floor_ratio.detach().cpu().numpy() if hasattr(floor_ratio, 'detach') else floor_ratio).reshape(-1).tolist(),
                    policy_step_updates=int(self.updates), ticks_by_feet_on_patch=self.on_counts.tolist())


def patch_session(make_session, placement, mu, marked):
    """Wrap the owner's make_session: markers (if marked) added to the appearance build, friction field on every policy step."""
    _install_marker_hook()

    def make(spec, directory, full_frames=False):
        _MARKERS[:] = [list(c) for c in placement['patch_cells']] if marked else []
        _MARKED_QUADS[:] = [0]
        try:
            session = make_session(spec, directory, full_frames=full_frames)
        finally:
            _MARKERS[:] = []
        field = FrictionField(session.ctx.build, placement['patch_cells'], mu)
        step = session.command_policy_step

        def command_policy_step(*args, **kwargs):
            field.update()
            return step(*args, **kwargs)
        session.command_policy_step = command_policy_step
        session.dynamics_patches = field
        record = dict(perturbation='low_friction_patches', marked=bool(marked), mu=float(mu), placement=placement,
                      marker_rgb=list(MARKER_RGB) if marked else None, marked_floor_quads=int(_MARKED_QUADS[0]),
                      field=field.receipt())
        Path(directory, 'dynamics_patches.json').write_text(json.dumps(record, indent=1)+'\n')
        if placement['patch_cells'] and marked != (record['marked_floor_quads'] > 0):
            raise ValueError('marked floor quads do not match the marked condition')
        return session
    make.dynamics = dict(perturbation='low_friction_patches', mu=float(mu), marked=bool(marked))
    return make
