"""Dynamics stage 2, patches v3: v2 strips with a textured marker and the 0.40-m/s session guard (development; 5 October 2026).

Two changes from lewm/dev_dynamics_patches_v2_development.py (v2, unchanged), both from the stage-2 smoke test
(`s2smoke_mup030`, C1, marked strips at mu_p = 0.3, prelim mazes 33, 42 and 47).

1. Session guard raised to 0.40 m/s on patch missions, marked and unmarked, for every controller.
   - Andrew's rule (4 October): only if patch edges still trip the 0.3-m/s guard.
   - They did: mazes 42 and 47 stopped at 0.300-0.301 m/s with the body centre 0.13-0.19 m inside the strip, just
     after entry.
   - The check is otherwise identical: any non-foot ground contact and any exit from the floor domain still stop the
     mission. Peak speeds remain in the physics trace.

2. Textured (tinted) marker in place of v2's uniform colour.
   - In maze 33 the uniform 2.5 x 1.3 m marker starved the visual tracker. On the strip the weaker camera's selected
     features had a median of 66 with a 10th percentile of 0, against 119 off the strip; the same corridor at uniform
     mu = 0.3 without a marker never starved (median 120, 10th percentile 86).
   - C1 then scanned on the strip for about 250 s (499 left turns) until the visual pose was lost.
   - A uniform colour also confounds marked with unmarked, since unmarked strips keep their texture.
   - v3 tints each floor quad's own colour, (r, g, b) -> (0.45 r, 0.70 g, 1.00 b), with no clipping. Every quad keeps
     its contrast with its neighbours, and the strip reads clearly blue against the grey floor.

Placement, the friction field and everything else are v2's.
"""
import json
from pathlib import Path

import numpy as np

from lewm import dev_dynamics_patches_v2_development as v2
from lewm.dev_dynamics_patches_v2_development import on_patch, place_patches, placement_seed, rects_of  # noqa: F401

TINT = (.45, .70, 1.00)
PATCH_SPEED_LIMIT_M_S = .40
_MARKERS = []
_MARKED_QUADS = []


def mark_floor(mesh, rects):
    """Copy of the floor mesh with every quad whose centre lies in a strip tinted, keeping its own variation."""
    vertices = np.asarray(mesh.vertices)
    if len(vertices) % 4 or not np.allclose(vertices[:, 2], 0.):
        raise ValueError('flat quad-per-four-vertices floor mesh required')
    centres = vertices.reshape(-1, 4, 3).mean(axis=1)[:, :2]
    inside = on_patch(centres, rects)
    colours = np.asarray(mesh.visual.vertex_colors).copy()
    mask = np.repeat(inside, 4)
    colours[mask, :3] = np.rint(colours[mask, :3].astype(float)*np.asarray(TINT)).astype(np.uint8)
    marked = mesh.copy()
    marked.visual.vertex_colors = colours
    return marked, int(inside.sum())


def _install_marker_hook():
    from lewm_genesis import visible_robot_union_rgbd_scene_development as builder
    if getattr(builder.independently_seeded_union_surfaces, 'dev_patch_markers_v3', False):
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
    surfaces.dev_patch_markers_v3 = True
    builder.independently_seeded_union_surfaces = surfaces


def raise_speed_guard(session, limit=PATCH_SPEED_LIMIT_M_S):
    """Session guard with the speed limit raised; contact and domain checks unchanged (copied from NovelMazeBaseSession)."""
    from scripts.novel_maze_round_trip_physical_session_development import PhysicalStop, nonfoot_ground_contact_indices
    original = session._sample

    def _sample(requested, applied, timestamp_s):
        guard = session.guard
        session.guard = None  # the original check is applied below with the raised limit
        try:
            row = original(requested, applied, timestamp_s)
        finally:
            session.guard = guard
        if guard is not None:
            packet = {k: np.asarray(v)[0] for k, v in session.packets[-1].items()}
            indices = nonfoot_ground_contact_indices(packet, **guard)
            speed = float(np.linalg.norm(row['base_twist_world'][:3]))
            inside = bool((np.abs(row['base_pose_world'][:2]) < 8).all())
            session.guard_rows.append(dict(sample_index=len(session.samples)-1, nonfoot_ground_contact_indices=indices,
                                           base_speed_m_s=speed, in_domain=inside, evaluator_only=True,
                                           speed_limit_m_s=limit))
            if indices or speed > limit or not inside:
                raise PhysicalStop('CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP')
        return row
    session._sample = _sample
    return session


def patch_session(make_session, placement, mu, marked):
    """v2's patch session with the tinted marker and the raised guard."""
    _install_marker_hook()
    rects = rects_of(placement)

    def make(spec, directory, full_frames=False):
        _MARKERS[:] = [list(r) for r in rects] if marked else []
        _MARKED_QUADS[:] = [0]
        try:
            session = make_session(spec, directory, full_frames=full_frames)
        finally:
            _MARKERS[:] = []
        field = v2.FrictionField(session.ctx.build, rects, mu)
        step = session.command_policy_step

        def command_policy_step(*args, **kwargs):
            field.update()
            return step(*args, **kwargs)
        session.command_policy_step = command_policy_step
        session.dynamics_patches = field
        raise_speed_guard(session)
        record = dict(perturbation='low_friction_patches', placement_version='v2', patches_version='v3', marked=bool(marked),
                      mu=float(mu), placement=placement, marker='tinted floor quads' if marked else None,
                      marker_tint=list(TINT) if marked else None, marked_floor_quads=int(_MARKED_QUADS[0]),
                      session_speed_limit_m_s=PATCH_SPEED_LIMIT_M_S, field=field.receipt())
        Path(directory, 'dynamics_patches.json').write_text(json.dumps(record, indent=1)+'\n')
        if rects and marked != (record['marked_floor_quads'] > 0):
            raise ValueError('marked floor quads do not match the marked condition')
        return session
    make.dynamics = dict(perturbation='low_friction_patches', placement_version='v2', patches_version='v3', mu=float(mu),
                         marked=bool(marked), session_speed_limit_m_s=PATCH_SPEED_LIMIT_M_S)
    return make
