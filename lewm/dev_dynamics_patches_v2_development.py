"""Dynamics stage 2, placement v2: low-friction strips on straight route segments (development; Andrew, 5 October 2026).

Same friction field and floor marker as lewm/dev_dynamics_patches_development.py (v1, unchanged), with a new placement
rule. The goal is to minimise in-place turns on patches.

Route: the shortest home-to-beacon cell path (v1's route_cells). Each route cell is classed as:
- a junction: degree 3 or more in the layout graph;
- a dead end: degree 1;
- an endpoint: the home or beacon cell;
- a corner: the route changes direction there;
- straight: everything else.
A straight segment is a maximal run of consecutive straight cells; its cells are collinear. A patch is an axis-aligned
strip inside one segment:
- across the corridor it spans the full cell width;
- along the route it stays at least 0.5 m from any neighbouring junction, dead-end or endpoint cell (CORNER_MARGIN_M
  from a neighbouring corner cell; the corner cell itself is never patched);
- it is at least 1.5 m long, the visibility requirement.
Strips are drawn with a seed until they cover 20-40% of the route length (route cells x 1.3 m); a strip that would pass
40% is shortened, never below 1.5 m. Episodes without a usable segment get no patch, and placement says so.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import dev_dynamics_patches_development as v1
from lewm.dev_dynamics_patches_development import MARKER_RGB, PITCH_M, route_cells

MARGIN_M = .5
CORNER_MARGIN_M = 0.
MIN_LENGTH_M = 1.5
COVERAGE = (.20, .40)
QUAD_M = .125
_MARKERS = []
_MARKED_QUADS = []


def placement_seed(set_name, maze, episode):
    return int.from_bytes(hashlib.sha256(f'patches-v2:{set_name}:{int(maze)}:{int(episode)}'.encode()).digest()[:8], 'big')


def classify(spec, route):
    layout = spec['evaluation_layout']
    degree = {tuple(c): 0 for c in layout['cells']}
    for a, b in layout['edges']:
        degree[tuple(a)] += 1
        degree[tuple(b)] += 1
    kinds = []
    for i, c in enumerate(route):
        if degree[c] >= 3:
            kinds.append('junction')
        elif degree[c] == 1:
            kinds.append('dead_end')
        elif i in (0, len(route)-1):
            kinds.append('endpoint')
        else:
            din = (c[0]-route[i-1][0], c[1]-route[i-1][1])
            dout = (route[i+1][0]-c[0], route[i+1][1]-c[1])
            kinds.append('straight' if din == dout else 'corner')
    return kinds


def segments(route, kinds):
    """Usable intervals: (axis, lateral centre, start, end) in metres along the segment's axis, after the margins."""
    runs, current = [], []
    for i, k in enumerate(kinds):
        if k == 'straight':
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    out = []
    for run in runs:
        first, last = route[run[0]], route[run[-1]]
        step = (route[run[0]+1][0]-first[0], route[run[0]+1][1]-first[1])  # straight cells are never the last route cell
        axis = 0 if step[0] else 1
        sign = step[axis]
        lo_cell, hi_cell = (first, last) if sign > 0 else (last, first)
        lo_neighbour = kinds[run[0]-1] if sign > 0 else kinds[run[-1]+1]
        hi_neighbour = kinds[run[-1]+1] if sign > 0 else kinds[run[0]-1]
        margin = lambda k: CORNER_MARGIN_M if k == 'corner' else MARGIN_M
        start = (lo_cell[axis]-.5)*PITCH_M+margin(lo_neighbour)
        end = (hi_cell[axis]+.5)*PITCH_M-margin(hi_neighbour)
        lateral = first[1-axis]*PITCH_M
        if end-start >= MIN_LENGTH_M-1e-9:
            out.append(dict(axis=axis, lateral_m=lateral, start_m=start, end_m=end,
                            cells=[list(route[i]) for i in run]))
    return out


def rectangle(segment, start, end):
    half = PITCH_M/2
    if segment['axis'] == 0:
        return [start, end, segment['lateral_m']-half, segment['lateral_m']+half]
    return [segment['lateral_m']-half, segment['lateral_m']+half, start, end]


def place_patches(spec, packet, seed):
    route = route_cells(spec, packet)
    kinds = classify(spec, route)
    usable = segments(route, kinds)
    route_m = len(route)*PITCH_M
    base = dict(route=[list(c) for c in route], cell_kinds=kinds, seed=int(seed), pitch_m=PITCH_M,
                rule=f'strips on straight route segments; >= {MARGIN_M} m from junction, dead-end and endpoint cells, '
                     f'{CORNER_MARGIN_M} m from corner cells; >= {MIN_LENGTH_M} m long; 20-40% of route length',
                usable_segments=usable)
    if not usable:
        return dict(base, patches=[], coverage=0., note='no straight segment long enough')
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(usable)))
    patches, covered = [], 0.
    for i in order:
        if covered >= COVERAGE[0]*route_m:
            break
        s = usable[int(i)]
        room = COVERAGE[1]*route_m-covered
        available = s['end_m']-s['start_m']
        length = min(available, max(MIN_LENGTH_M, room))
        if length < MIN_LENGTH_M-1e-9 or covered+length > max(COVERAGE[1]*route_m, MIN_LENGTH_M)+1e-9:
            continue
        length = np.floor(length/QUAD_M+1e-9)*QUAD_M
        slack = available-length
        offset = np.floor(rng.uniform(0, slack)/QUAD_M)*QUAD_M if slack > 0 else 0.
        start = s['start_m']+offset
        patches.append(dict(rect=rectangle(s, start, start+length), length_m=float(length), segment=s['cells']))
        covered += length
    return dict(base, patches=patches, coverage=covered/route_m)


def rects_of(placement):
    return [p['rect'] for p in placement['patches']]


def on_patch(xy, rects):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not rects:
        return np.zeros(len(xy), bool)
    r = np.asarray(rects, float)
    return np.any((xy[:, None, 0] >= r[None, :, 0]) & (xy[:, None, 0] <= r[None, :, 1])
                  & (xy[:, None, 1] >= r[None, :, 2]) & (xy[:, None, 1] <= r[None, :, 3]), axis=1)


def signed_edge_distance(xy, rects):
    """Signed distance from a point to the nearest patch boundary: negative inside a patch, positive outside."""
    p = np.asarray(xy, float)[:2]
    best = np.inf
    for x0, x1, y0, y1 in rects:
        dx, dy = max(x0-p[0], 0., p[0]-x1), max(y0-p[1], 0., p[1]-y1)
        if dx == 0 and dy == 0:
            d = -min(p[0]-x0, x1-p[0], p[1]-y0, y1-p[1])
        else:
            d = float(np.hypot(dx, dy))
        best = min(best, d)
    return float(best)


def mark_floor(mesh, rects):
    vertices = np.asarray(mesh.vertices)
    if len(vertices) % 4 or not np.allclose(vertices[:, 2], 0.):
        raise ValueError('flat quad-per-four-vertices floor mesh required')
    centres = vertices.reshape(-1, 4, 3).mean(axis=1)[:, :2]
    inside = on_patch(centres, rects)
    colours = np.asarray(mesh.visual.vertex_colors).copy()
    colours[np.repeat(inside, 4)] = np.asarray([round(255*c) for c in MARKER_RGB]+[255], np.uint8)
    marked = mesh.copy()
    marked.visual.vertex_colors = colours
    return marked, int(inside.sum())


def _install_marker_hook():
    from lewm_genesis import visible_robot_union_rgbd_scene_development as builder
    if getattr(builder.independently_seeded_union_surfaces, 'dev_patch_markers_v2', False):
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
    surfaces.dev_patch_markers_v2 = True
    builder.independently_seeded_union_surfaces = surfaces


class FrictionField(v1.FrictionField):
    """v1's per-leg field, with strip (rectangle) membership instead of whole cells."""

    def update(self):
        on = on_patch(self.foot_xy(), self.patch_cells)
        self.solver.set_geoms_friction_ratio(np.asarray([self.mu if on[k] else 1. for k, leg in enumerate(self.legs) for _ in leg]
                                                        + [self.mu]*len(self.floor)), self.ids)
        self.updates += 1
        self.on_counts[int(on.sum())] += 1
        self.last_on = on
        return on


def patch_session(make_session, placement, mu, marked):
    """As v1's patch_session, with strips: markers on the appearance build if marked, the field on every policy step."""
    _install_marker_hook()
    rects = rects_of(placement)

    def make(spec, directory, full_frames=False):
        _MARKERS[:] = [list(r) for r in rects] if marked else []
        _MARKED_QUADS[:] = [0]
        try:
            session = make_session(spec, directory, full_frames=full_frames)
        finally:
            _MARKERS[:] = []
        field = FrictionField(session.ctx.build, rects, mu)
        step = session.command_policy_step

        def command_policy_step(*args, **kwargs):
            field.update()
            return step(*args, **kwargs)
        session.command_policy_step = command_policy_step
        session.dynamics_patches = field
        record = dict(perturbation='low_friction_patches', placement_version='v2', marked=bool(marked), mu=float(mu),
                      placement=placement, marker_rgb=list(MARKER_RGB) if marked else None,
                      marked_floor_quads=int(_MARKED_QUADS[0]), field=field.receipt())
        Path(directory, 'dynamics_patches.json').write_text(json.dumps(record, indent=1)+'\n')
        if rects and marked != (record['marked_floor_quads'] > 0):
            raise ValueError('marked floor quads do not match the marked condition')
        return session
    make.dynamics = dict(perturbation='low_friction_patches', placement_version='v2', mu=float(mu), marked=bool(marked))
    return make
