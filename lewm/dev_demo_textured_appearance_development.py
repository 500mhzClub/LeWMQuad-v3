"""Demonstration missions in a textured world (development; Andrew, 5 October 2026). Presentation only, not an experiment.

Andrew asked for a hero demo in which the robot itself sees the nicer textures, not only the presentation camera.

**Appearance.** The capability scene draws exactly two static visual surfaces, `ground_visual` and `wall_union_visual`,
which are vertex-coloured meshes. Normally they are 12.5-cm random-grey quads. This module replaces them with
vertex-coloured grids at 2.5-cm spacing, coloured by sampling CC0 textures from assets/textures:
- the floor: WoodFloor043, tiled every 1.5 m, over the maze bounding box plus 1 m;
- every face of the wall union: PaintedPlaster017, tiled every 1.5 m, with a 10-cm darker skirting band at the bottom.
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
WALL_TEX = REPO/'assets/textures/wall/PaintedPlaster017.jpg'
SPACING = .025
TILE_M = 1.5
SKIRT_M, SKIRT_SHADE = .10, .55


def _texture(path, pixels=512):
    return np.asarray(Image.open(path).convert('RGB').resize((pixels, pixels), Image.Resampling.LANCZOS), np.float32)


def _sample(texture, u, v):
    """Bilinear-free nearest sample of a tiled texture at metric coordinates (u, v)."""
    n = texture.shape[0]
    i = (np.floor((v/TILE_M % 1.)*n).astype(int)) % n
    j = (np.floor((u/TILE_M % 1.)*n).astype(int)) % n
    return texture[i, j]


def grid(origin, u, v, lengths, texture, skirt=False):
    """A shared-vertex grid over origin + a*u + b*v, coloured from the texture (vertex colours interpolate smoothly)."""
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    n = max(1, int(np.ceil(lengths[0]/SPACING)))
    m = max(1, int(np.ceil(lengths[1]/SPACING)))
    a, b = np.meshgrid(np.linspace(0, lengths[0], n+1), np.linspace(0, lengths[1], m+1))
    points = origin+a[..., None]*u+b[..., None]*v
    world_u = (points@u)
    world_v = points[..., 2] if abs(v[2]) > .5 else (points@v)
    colour = _sample(texture, world_u, world_v)
    if skirt:
        colour = np.where((points[..., 2] < SKIRT_M)[..., None], colour*SKIRT_SHADE, colour)
    index = np.arange((n+1)*(m+1)).reshape(m+1, n+1)
    q = np.stack((index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1).reshape(-1, 4)
    faces = np.concatenate((q[:, [0, 1, 2]], q[:, [0, 2, 3]]))
    rgba = np.concatenate((np.clip(colour, 0, 255).reshape(-1, 3), np.full(((n+1)*(m+1), 1), 255.)), axis=1).astype(np.uint8)
    return trimesh.Trimesh(vertices=points.reshape(-1, 3), faces=faces, vertex_colors=rgba, process=False)


def textured_surfaces(boxes):
    from lewm_genesis.union_wall_surface_development import wall_union_boundary
    floor_tex, wall_tex = _texture(FLOOR_TEX), _texture(WALL_TEX)
    xs = [b['centre_xyz'][0] for b in boxes]
    ys = [b['centre_xyz'][1] for b in boxes]
    x0, x1, y0, y1 = min(xs)-1., max(xs)+1., min(ys)-1., max(ys)+1.
    floor = grid([x0, y0, 0.], [1, 0, 0], [0, 1, 0], [x1-x0, y1-y0], floor_tex)
    walls = []
    for face in wall_union_boundary(boxes)['faces']:
        axis = face['axis']
        j, k = (axis+1) % 3, (axis+2) % 3
        sign = face['normal_sign']
        origin = np.zeros(3)
        origin[axis] = face['plane_um']/1e6
        origin[j] = face['u0_um']/1e6
        origin[k] = face['v0_um' if sign == 1 else 'v1_um']/1e6
        u, v = np.eye(3)[j], sign*np.eye(3)[k]
        lengths = np.array([face['u1_um']-face['u0_um'], face['v1_um']-face['v0_um']])/1e6
        walls.append(grid(origin, u, v, lengths, wall_tex, skirt=True))
    return [('ground_visual', floor), ('wall_union_visual', trimesh.util.concatenate(walls))]


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
    """Save each consumed primary RGB frame as ego_frames/NNNN.png in the session directory."""
    def make(spec, directory, full_frames=False):
        session = make_session(spec, directory, full_frames=full_frames)
        out = Path(directory)/'ego_frames'
        out.mkdir(exist_ok=True)
        original = session.sensor_packets
        count = [0]

        def sensor_packets(*args, **kwargs):
            packets = original(*args, **kwargs)
            image = session.captured_pairs[-1]['images'][0][0]
            Image.fromarray(np.asarray(image).astype(np.uint8)).save(out/f'{count[0]:04d}.png')
            count[0] += 1
            return packets
        session.sensor_packets = sensor_packets
        return session
    make.dynamics = dict(getattr(make_session, 'dynamics', {}) or {}, demo_textures=True, ego_frames_saved=True)
    return make
