"""Demonstration missions in a textured world (development; Andrew, 5 October 2026). Presentation only, not an experiment.

Andrew asked for a hero demo in which the robot itself sees the nicer textures, not only the presentation camera.

**Appearance.** The capability scene draws exactly two static visual surfaces, `ground_visual` and `wall_union_visual`,
which are vertex-coloured meshes. Normally they are 12.5-cm random-grey quads. This module replaces them with:
- walls: running-bond brickwork (21.5 x 6.5 cm bricks, 1-cm mortar joints) on every vertical face of the wall union,
  each brick its own quad coloured from a brick sampled from the CC0 texture Bricks097, mortar from its joints; light
  stone caps on the tops;
- floor: wood-strip parquet (45 x 9 cm strips, 6-mm dark seams, each row at a random offset) over the maze bounding box
  plus 1 m, each strip coloured along its length from WoodFloor043; a coarse, never-visible surround keeps the floor's
  exact [-16, 16] extent (the scene's raster-order contract identifies the floor by it).

Every brick, strip, joint and seam has its own vertices, so the edges stay crisp. This matters: the visual pose tracker
(SIFT and optical flow on grayscale) needs corners spread over the image. A first version (smooth texture-sampled
vertex colours at 2.5 cm) gave a tenth of the original keypoints (mean 27 vs 257) and lost tracking within 5 s on mazes
44 and 30.

Geometry, collision, depth (the same surfaces cover the maze interior), the draw order and the identity witnesses (built
from these meshes) are unchanged in kind. Only colours and tessellation differ.

**Ego frames.** Each consumed primary RGB frame is also saved as ego_frames/NNNN.png (10 Hz) in the session directory,
so the demo video's robot-camera panel is the frames the controller actually received.

The JEPA models were never trained on these textures; demo videos must say so.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

REPO = Path(__file__).resolve().parents[1]
FLOOR_TEX = REPO/'assets/textures/floor/WoodFloor043.jpg'
WALL_TEX = REPO/'assets/textures/wall/Bricks097.jpg'
SEED = 20261005
BRICK = dict(length=.225, height=.075, joint=.010)
STRIP = dict(length=.45, width=.09, seam=.006, segments=3)


def _texture(path, pixels=256):
    return np.asarray(Image.open(path).convert('RGB').resize((pixels, pixels), Image.Resampling.LANCZOS), np.float32)


def palettes(rng):
    """Brick and mortar colours from Bricks097; wood colours along the grain from WoodFloor043."""
    bricks = _texture(WALL_TEX)
    n = bricks.shape[0]
    centres = rng.integers(3, n-3, size=(4000, 2))
    patches = np.array([bricks[i-2:i+3, j-2:j+3].reshape(-1, 3).mean(0) for i, j in centres])
    red = patches[(patches[:, 0]-patches[:, 2] > 30) & (patches.sum(1) < 450)]
    flat = bricks.reshape(-1, 3)
    light = flat[flat.sum(1) > np.quantile(flat.sum(1), .85)]
    wood = _texture(FLOOR_TEX)
    rows = rng.integers(0, wood.shape[0], size=512)
    cols = rng.integers(0, wood.shape[1]-40, size=512)
    grain = np.stack([wood[r, c:c+40:13][:STRIP['segments']+1] for r, c in zip(rows, cols)])
    return red[:512], light.mean(0), grain


class Quads:
    def __init__(self):
        self.points, self.colours = [], []

    def add(self, corners, colours):
        self.points.append(np.asarray(corners, float))
        self.colours.append(np.broadcast_to(np.asarray(colours, float), (4, 3)))

    def mesh(self):
        points = np.concatenate(self.points)
        n = len(self.points)
        base = 4*np.arange(n)[:, None]
        faces = np.concatenate((base+[0, 1, 2], base+[0, 2, 3]))
        rgba = np.concatenate((np.clip(np.concatenate(self.colours), 0, 255), np.full((4*n, 1), 255.)), 1).astype(np.uint8)
        return trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=rgba, process=False)


def _bond(s0, s1, z0, z1, *, length, height, joint, offsets):
    """Running bond (or strip rows) over [s0, s1] x [z0, z1]: yields (rect, kind, course, unit); rect = (sa, sb, za, zb)."""
    for course in range(int(np.ceil((z1-z0)/height-1e-9))):
        za, zb = z0+course*height, min(z0+(course+1)*height, z1)
        yield (s0, s1, za, min(za+joint, zb)), 'joint', course, 0
        if za+joint >= zb:
            continue
        offset = offsets(course)
        for unit in range(int(np.floor((s0-offset)/length)), int(np.ceil((s1-offset)/length))):
            a = offset+unit*length
            for (sa, sb), kind in (((a, a+joint), 'joint'), ((a+joint, a+length), 'unit')):
                sa, sb = max(sa, s0), min(sb, s1)
                if sb-sa > 1e-6:
                    yield (sa, sb, za+joint, zb), kind, course, unit


def textured_surfaces(boxes):
    from lewm_genesis.union_wall_surface_development import wall_union_boundary
    rng = np.random.default_rng(SEED)
    brick, mortar, grain = palettes(rng)
    shade = rng.uniform(.85, 1.12, size=4096)
    # Floor: parquet strips over the maze bounding box plus 1 m.
    xs = [b['centre_xyz'][0] for b in boxes]
    ys = [b['centre_xyz'][1] for b in boxes]
    x0, x1, y0, y1 = min(xs)-1., max(xs)+1., min(ys)-1., max(ys)+1.
    row_offsets = rng.uniform(0, STRIP['length'], size=4096)
    seam = grain.reshape(-1, 3).mean(0)*.35
    floor = Quads()
    k = STRIP['segments']
    for (sa, sb, za, zb), kind, row, unit in _bond(x0, x1, y0, y1, length=STRIP['length'], height=STRIP['width'],
                                                    joint=STRIP['seam'], offsets=lambda r: row_offsets[r % 4096]):
        if kind == 'joint':
            floor.add([[sa, za, 0], [sb, za, 0], [sb, zb, 0], [sa, zb, 0]], seam)
            continue
        h = (row*7919+unit*104729) % 4096
        colours = grain[h % len(grain)]*shade[h]
        cuts = np.linspace(sa, sb, k+1)
        for i in range(k):
            floor.add([[cuts[i], za, 0], [cuts[i+1], za, 0], [cuts[i+1], zb, 0], [cuts[i], zb, 0]],
                      [colours[i], colours[i+1], colours[i+1], colours[i]])
    tone = grain.reshape(-1, 3).mean(0)*.6
    for a0, a1, b0, b1 in ((-16, 16, -16, y0), (-16, 16, y1, 16), (-16, x0, y0, y1), (x1, 16, y0, y1)):
        floor.add([[a0, b0, 0], [a1, b0, 0], [a1, b1, 0], [a0, b1, 0]], tone)
    # Walls: brickwork on the vertical faces, stone caps on the horizontal ones (same winding as the face's u x v).
    walls = Quads()
    for face in wall_union_boundary(boxes)['faces']:
        axis, sign = face['axis'], face['normal_sign']
        j, k3 = (axis+1) % 3, (axis+2) % 3
        origin = np.zeros(3)
        origin[axis] = face['plane_um']/1e6
        origin[j] = face['u0_um']/1e6
        origin[k3] = face['v0_um' if sign == 1 else 'v1_um']/1e6
        u, v = np.eye(3)[j], sign*np.eye(3)[k3]
        uj = np.array([face['u0_um'], face['u1_um']])/1e6
        vk = np.array([face['v0_um'], face['v1_um']])/1e6

        def emit(cj, ck, colour):
            """A world rectangle cj x ck on this face, as a quad in the face's own (a, b) orientation."""
            a = sorted(c-origin[j] for c in cj)
            b = sorted((c-origin[k3])*sign for c in ck)
            corners = [origin+a[0]*u+b[0]*v, origin+a[1]*u+b[0]*v, origin+a[1]*u+b[1]*v, origin+a[0]*u+b[1]*v]
            walls.add(corners, colour)
        if axis == 2:
            emit(uj, vk, mortar*.92)
            continue
        # axis 0: (u, v) = (y, z); axis 1: (u, v) = (z, x). s is the horizontal coordinate along the face.
        s_range, z_range = (uj, vk) if axis == 0 else (vk, uj)
        for (sa, sb, za, zb), kind, course, unit in _bond(*s_range, *z_range, offsets=lambda c: (c % 2)*BRICK['length']/2,
                                                        **BRICK):
            if kind == 'joint':
                colour = mortar
            else:
                h = (course*7919+unit*104729+axis*15485863) % 4096
                colour = brick[h % len(brick)]*shade[h]
            emit((sa, sb), (za, zb), colour) if axis == 0 else emit((za, zb), (sa, sb), colour)
    return [('ground_visual', floor.mesh()), ('wall_union_visual', walls.mesh())]


def install():
    """Replace the capability scene's two visual surfaces with the textured ones (this process only)."""
    from lewm_genesis import visible_robot_union_rgbd_scene_development as builder
    if getattr(builder.independently_seeded_union_surfaces, 'dev_demo_textures', False):
        return
    original = builder.independently_seeded_union_surfaces

    def surfaces(boxes, arm, seed):
        original(boxes, arm, seed)  # keep its argument validation
        return textured_surfaces(boxes)
    surfaces.dev_demo_textures = True
    builder.independently_seeded_union_surfaces = surfaces


def ego_session(make_session):
    """Save each consumed primary RGB frame as ego_frames/NNNN.png in the session directory.

    The session's hash-retention mixin clears the captured images inside sensor_packets, right after record_consumed
    hashes them, so the frame is taken inside record_consumed (looked up from the module at call time)."""
    from lewm import navigation_capability_sensor_retention_development as retention
    latest = {}
    if not getattr(retention.record_consumed, 'dev_demo_ego', False):
        original_record = retention.record_consumed

        def record_consumed(row, packets):
            latest['image'] = np.asarray(row['images'][0][0]).astype(np.uint8).copy()
            return original_record(row, packets)
        record_consumed.dev_demo_ego = True
        record_consumed.latest = latest
        retention.record_consumed = record_consumed
    latest = retention.record_consumed.latest

    def make(spec, directory, full_frames=False):
        session = make_session(spec, directory, full_frames=full_frames)
        out = Path(directory)/'ego_frames'
        out.mkdir(exist_ok=True)
        original = session.sensor_packets
        count = [0]

        def sensor_packets(*args, **kwargs):
            latest.pop('image', None)
            packets = original(*args, **kwargs)
            Image.fromarray(latest.pop('image')).save(out/f'{count[0]:04d}.png')
            count[0] += 1
            return packets
        session.sensor_packets = sensor_packets
        return session
    make.dynamics = dict(getattr(make_session, 'dynamics', {}) or {}, demo_textures=True, ego_frames_saved=True)
    return make
