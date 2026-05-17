"""Scene-family registry and deterministic per-family builders.

Each family produces a :class:`SceneManifest` from a single integer seed.
Mazes use a randomized spanning tree with optional extra cycles; open fields
scatter box obstacles; composite-motif scenes hand-author topology around a
single navigation skill; rough-dynamics scenes scatter slopes and steps;
visual-stress scenes inherit medium-maze topology with aggressive lighting
and texture randomization.

Geometry is sampled per scene to match the data-spec §9 corridor distribution
(`1.6 x W_body` to `3.0 x W_body`, with ≥25% in the difficult
`1.6 x W_body` to `2.0 x W_body` band). Visual, physics, and camera-extrinsic
randomization (§14) is drawn from :mod:`lewm_worlds.randomization`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphEdge,
    GraphNode,
    SceneManifest,
    SpawnSpec,
    stable_scene_id,
)
from lewm_worlds.randomization import (
    BASE_PALETTE,
    DEFAULT_PROFILE,
    LANDMARK_PALETTE,
    Profile,
    ROUGH_DYNAMICS_PROFILE,
    VISUAL_STRESS_PROFILE,
    draw_camera_extrinsic_jitter,
    draw_physics_randomization,
    draw_visual_randomization,
)


# ---------------------------------------------------------------------------
# Robot-normalized geometry constants (Go2)
# ---------------------------------------------------------------------------


# Body width at hip/shoulder envelope. Used to express corridor widths in
# robot-normalized units (data spec §9). Source: Unitree Go2 nominal
# dimensions.
W_BODY_M = 0.31

# Corridor width band: spec wants [1.6, 3.0] x W_body.
CORRIDOR_NARROW_BAND_M = (1.6 * W_BODY_M, 2.0 * W_BODY_M)  # 0.496..0.620
CORRIDOR_WIDE_BAND_M = (2.0 * W_BODY_M, 3.0 * W_BODY_M)  # 0.620..0.930
# At least 25% of scenes should sample from the narrow band. We use 30% so
# that small-scale corpora reliably contain a few narrow scenes.
CORRIDOR_NARROW_BAND_PROB = 0.30

# Floor wall thickness — the wall slab gets thicker when corridors are narrow
# so that cells always have enough room for the robot to occupy a junction
# (cell_size = corridor_width + wall_thickness). Cells are kept ≥0.80 m so
# the Go2 (L_body ≈ 0.65 m) fits comfortably in junctions.
WALL_THICKNESS_FLOOR_M = 0.12
CELL_SIZE_FLOOR_M = 0.80
WALL_HEIGHT_M = 0.80

# Camera-validity constraints — must satisfy the smallest wall thickness so
# synthetic-render checks do not fail at the narrowest cells.
_CAMERA_CONSTRAINTS = CameraValidityConstraints(
    min_wall_thickness_m=WALL_THICKNESS_FLOOR_M,
    near_m=0.05,
    far_m=200.0,
    min_camera_clearance_m=0.10,
)


# ---------------------------------------------------------------------------
# Family registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    """Static metadata describing a registered scene family."""

    name: str
    difficulty_tier: str
    description: str
    builder: Callable[["BuildContext"], SceneManifest]


@dataclass(frozen=True)
class BuildContext:
    """Inputs supplied to a per-family builder."""

    scene_seed: int
    family: str
    split: str | None
    difficulty_tier: str


def build_family_manifest(
    *,
    scene_seed: int,
    family: str,
    split: str | None,
    difficulty_tier: str | None,
) -> SceneManifest:
    spec = _registry().get(family)
    if spec is None:
        raise ValueError(
            f"unknown scene family '{family}'; "
            f"registered families: {sorted(_registry())}"
        )
    ctx = BuildContext(
        scene_seed=int(scene_seed),
        family=family,
        split=split,
        difficulty_tier=(difficulty_tier or spec.difficulty_tier),
    )
    return spec.builder(ctx)


def registered_families() -> tuple[str, ...]:
    return tuple(sorted(_registry()))


def family_spec(family: str) -> FamilySpec:
    spec = _registry().get(family)
    if spec is None:
        raise ValueError(f"unknown scene family '{family}'")
    return spec


# ---------------------------------------------------------------------------
# Corridor geometry (data spec §9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorridorGeometry:
    """Per-scene cell / corridor / wall geometry sampled from the spec band."""

    cell_size_m: float
    corridor_width_m: float
    wall_thickness_m: float
    band: str  # "narrow" or "wide"


def sample_corridor_geometry(rng: random.Random) -> CorridorGeometry:
    """Return per-scene geometry honoring the §9 corridor distribution.

    The wall thickness scales up when corridors are narrow so the cell size
    stays above ``CELL_SIZE_FLOOR_M`` — that keeps junctions large enough
    for the Go2 (L_body ≈ 0.65 m) to make pivot turns, while still presenting
    the narrow-corridor visual/spatial signal the spec wants.
    """

    if rng.random() < CORRIDOR_NARROW_BAND_PROB:
        corridor_width = rng.uniform(*CORRIDOR_NARROW_BAND_M)
        band = "narrow"
    else:
        corridor_width = rng.uniform(*CORRIDOR_WIDE_BAND_M)
        band = "wide"
    wall_thickness = max(WALL_THICKNESS_FLOOR_M, CELL_SIZE_FLOOR_M - corridor_width)
    cell_size = corridor_width + wall_thickness
    return CorridorGeometry(
        cell_size_m=round(cell_size, 4),
        corridor_width_m=round(corridor_width, 4),
        wall_thickness_m=round(wall_thickness, 4),
        band=band,
    )


# ---------------------------------------------------------------------------
# Shared assembly
# ---------------------------------------------------------------------------


def _assemble(
    *,
    ctx: BuildContext,
    world_half: float,
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
    walls: tuple[BoxObject, ...],
    obstacles: tuple[BoxObject, ...],
    landmarks: tuple[BoxObject, ...],
    profile: Profile = DEFAULT_PROFILE,
    extra_palette: tuple[str, ...] = (),
    spawn_xy: tuple[float, float] | None = None,
    distractor_bounds_half_m: float | None = None,
) -> SceneManifest:
    spawn_x, spawn_y = spawn_xy if spawn_xy is not None else (0.0, 0.0)
    spawn = SpawnSpec(
        xyz_m=(float(spawn_x), float(spawn_y), 0.375),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    visual_seed = int(ctx.scene_seed) + 100_000
    physics_seed = int(ctx.scene_seed) + 200_000
    bounds_half = distractor_bounds_half_m if distractor_bounds_half_m is not None else max(0.5, world_half - 0.5)
    visual_random = draw_visual_randomization(
        visual_seed=visual_seed,
        family=ctx.family,
        profile=profile,
        extra_palette=extra_palette,
        distractor_bounds_half_m=bounds_half,
    )
    physics_random = draw_physics_randomization(
        physics_seed=physics_seed,
        family=ctx.family,
        profile=profile,
    )
    camera_jitter = draw_camera_extrinsic_jitter(
        physics_seed=physics_seed,
        family=ctx.family,
        profile=profile,
    )
    return SceneManifest(
        scene_id=stable_scene_id(ctx.family, ctx.scene_seed),
        family=ctx.family,
        difficulty_tier=ctx.difficulty_tier,
        topology_seed=int(ctx.scene_seed),
        visual_seed=visual_seed,
        physics_seed=physics_seed,
        world_bounds_xy_m=(
            (-world_half, -world_half),
            (world_half, world_half),
        ),
        spawn=spawn,
        graph_nodes=nodes,
        graph_edges=edges,
        obstacles=obstacles,
        landmarks=landmarks,
        camera_constraints=_CAMERA_CONSTRAINTS,
        split=ctx.split,
        walls=walls,
        visual_randomization=visual_random,
        physics_randomization=physics_random,
        camera_extrinsic_jitter=camera_jitter,
    )


# ---------------------------------------------------------------------------
# Family: open obstacle field
# ---------------------------------------------------------------------------


def _build_open_obstacle_field(ctx: BuildContext) -> SceneManifest:
    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    world_half = 5.0
    grid_side = 3
    spacing = 2.4

    nodes = _grid_nodes(grid_side=grid_side, spacing=spacing, node_width=1.2)
    edges = _grid_edges(grid_side=grid_side, edge_width=1.0)

    obstacle_count = 5
    obstacles = _scatter_box_obstacles(
        rng=rng,
        count=obstacle_count,
        bounds_half=3.6,
        size_x_range=(0.35, 0.9),
        size_y_range=(0.35, 1.1),
        size_z_range=(0.25, 0.9),
        keepout_radius=0.8,
    )

    landmarks = _corner_landmarks(world_half=world_half)
    return _assemble(
        ctx=ctx,
        world_half=world_half,
        nodes=nodes,
        edges=edges,
        walls=(),
        obstacles=obstacles,
        landmarks=landmarks,
    )


# ---------------------------------------------------------------------------
# Family: enclosed mazes (small / medium / large)
# ---------------------------------------------------------------------------


def _enclosed_maze_builder(
    side_range: tuple[int, int],
    extra_loop_density: float,
    landmark_pairs: int = 1,
    profile: Profile = DEFAULT_PROFILE,
) -> Callable[[BuildContext], SceneManifest]:
    def builder(ctx: BuildContext) -> SceneManifest:
        rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
        side = rng.randint(*side_range)
        geometry = sample_corridor_geometry(rng)
        cell_size = geometry.cell_size_m
        corridor_width = geometry.corridor_width_m
        wall_thickness = geometry.wall_thickness_m

        world_half = (side * cell_size) / 2.0 + wall_thickness

        nodes, traversable, all_pairs = _maze_topology(
            rng=rng,
            side=side,
            cell_size=cell_size,
            corridor_width=corridor_width,
            extra_loop_density=extra_loop_density,
        )
        edges = _maze_edges(
            traversable=traversable,
            all_pairs=all_pairs,
            corridor_width=corridor_width,
        )
        walls = _maze_walls(
            side=side,
            cell_size=cell_size,
            wall_thickness=wall_thickness,
            wall_height=WALL_HEIGHT_M,
            traversable=traversable,
        )
        landmarks = _maze_landmarks(
            rng=rng,
            nodes=nodes,
            pairs=landmark_pairs,
            wall_height=WALL_HEIGHT_M,
        )
        return _assemble(
            ctx=ctx,
            world_half=world_half,
            nodes=nodes,
            edges=edges,
            walls=walls,
            obstacles=(),
            landmarks=landmarks,
            spawn_xy=nodes[0].center_xy_m,
            profile=profile,
        )

    return builder


# ---------------------------------------------------------------------------
# Family: loop / alias stress
# ---------------------------------------------------------------------------


def _build_loop_alias_stress(ctx: BuildContext) -> SceneManifest:
    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    side = rng.randint(6, 7)
    geometry = sample_corridor_geometry(rng)
    cell_size = geometry.cell_size_m
    corridor_width = geometry.corridor_width_m
    wall_thickness = geometry.wall_thickness_m

    world_half = (side * cell_size) / 2.0 + wall_thickness

    nodes, traversable, all_pairs = _maze_topology(
        rng=rng,
        side=side,
        cell_size=cell_size,
        corridor_width=corridor_width,
        extra_loop_density=0.25,
    )
    edges = _maze_edges(
        traversable=traversable,
        all_pairs=all_pairs,
        corridor_width=corridor_width,
    )
    walls = _maze_walls(
        side=side,
        cell_size=cell_size,
        wall_thickness=wall_thickness,
        wall_height=WALL_HEIGHT_M,
        traversable=traversable,
    )
    # Alias stress: two identical-colored landmark pairs at mirrored positions.
    landmarks = _aliased_landmarks(nodes=nodes, wall_height=WALL_HEIGHT_M)
    return _assemble(
        ctx=ctx,
        world_half=world_half,
        nodes=nodes,
        edges=edges,
        walls=walls,
        obstacles=(),
        landmarks=landmarks,
        spawn_xy=nodes[0].center_xy_m,
    )


# ---------------------------------------------------------------------------
# Family: local composite motifs
# ---------------------------------------------------------------------------


# Each motif is a small hand-authored topology that exercises one navigation
# skill named in data-spec §10 (T-junction, S-bend, doorway, slalom,
# short dead-end). One motif is picked per scene; the scene seed selects the
# motif type and its parameters.
_MOTIF_TYPES = ("t_junction", "s_bend", "doorway", "slalom", "short_dead_end")


def _build_local_composite_motifs(ctx: BuildContext) -> SceneManifest:
    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    geometry = sample_corridor_geometry(rng)
    motif = rng.choice(_MOTIF_TYPES)
    builder_fn = _MOTIF_BUILDERS[motif]
    nodes, traversable, all_pairs, walls, obstacles, landmarks, spawn_xy, world_half = builder_fn(
        rng=rng,
        geometry=geometry,
    )
    edges = _maze_edges(
        traversable=traversable,
        all_pairs=all_pairs,
        corridor_width=geometry.corridor_width_m,
    )
    return _assemble(
        ctx=ctx,
        world_half=world_half,
        nodes=nodes,
        edges=edges,
        walls=walls,
        obstacles=obstacles,
        landmarks=landmarks,
        spawn_xy=spawn_xy,
    )


def _motif_t_junction(*, rng: random.Random, geometry: CorridorGeometry):
    """Stem corridor opening into a T-shape with two arm branches.

    Topology (cells, 'S' = spawn, 'T' = top of T, 'L'/'R' = arms):

        L L T R R
              |
              s
              |
              s
              |
              S

    The robot walks up the stem and is forced to commit to one branch
    without seeing the dead-end behind the opposite branch.
    """

    cell = geometry.cell_size_m
    arm_len = rng.randint(2, 3)
    stem_len = rng.randint(2, 4)

    cells: dict[tuple[int, int], int] = {}
    nodes: list[GraphNode] = []
    width = geometry.corridor_width_m

    def add(row: int, col: int, tags: tuple[str, ...] = ()) -> int:
        if (row, col) in cells:
            return cells[(row, col)]
        node_id = len(nodes)
        cells[(row, col)] = node_id
        nodes.append(
            GraphNode(
                node_id=node_id,
                center_xy_m=(round(col * cell, 4), round(row * cell, 4)),
                width_m=width,
                tags=tags,
            )
        )
        return node_id

    # Stem cells from row=0 (spawn) up to row=stem_len-1
    for r in range(stem_len):
        tags = ("spawn",) if r == 0 else ()
        add(r, 0, tags=tags)
    # T-junction: at row=stem_len, columns -arm_len..+arm_len
    for c in range(-arm_len, arm_len + 1):
        add(stem_len, c)

    traversable: set[frozenset[int]] = set()
    all_pairs: list[frozenset[int]] = []

    def link(a: int, b: int, walkable: bool = True) -> None:
        pair = frozenset({a, b})
        all_pairs.append(pair)
        if walkable:
            traversable.add(pair)

    for r in range(stem_len - 1):
        link(cells[(r, 0)], cells[(r + 1, 0)])
    link(cells[(stem_len - 1, 0)], cells[(stem_len, 0)])
    for c in range(-arm_len, arm_len):
        link(cells[(stem_len, c)], cells[(stem_len, c + 1)])

    walls = _walls_for_motif(nodes=nodes, traversable=traversable, geometry=geometry)
    landmarks = _two_landmarks_at_nodes(
        nodes=nodes,
        first=cells[(stem_len, -arm_len)],
        second=cells[(stem_len, arm_len)],
    )
    spawn_xy = nodes[cells[(0, 0)]].center_xy_m
    world_half = max(arm_len, stem_len) * cell + geometry.wall_thickness_m + 1.0
    return tuple(nodes), traversable, all_pairs, walls, (), landmarks, spawn_xy, world_half


def _motif_s_bend(*, rng: random.Random, geometry: CorridorGeometry):
    """Two opposite turns in close succession (S-shape).

    Topology (cell grid):

        . . G G
        . . . T2
        S B T1 .
        S . . .
    """

    cell = geometry.cell_size_m
    width = geometry.corridor_width_m
    cells: dict[tuple[int, int], int] = {}
    nodes: list[GraphNode] = []

    def add(row: int, col: int, tags: tuple[str, ...] = ()) -> int:
        if (row, col) in cells:
            return cells[(row, col)]
        node_id = len(nodes)
        cells[(row, col)] = node_id
        nodes.append(
            GraphNode(
                node_id=node_id,
                center_xy_m=(round(col * cell, 4), round(row * cell, 4)),
                width_m=width,
                tags=tags,
            )
        )
        return node_id

    spawn_len = rng.randint(2, 3)
    seg1 = rng.randint(2, 3)
    seg2 = rng.randint(2, 3)
    goal_len = rng.randint(2, 3)

    # Bottom segment: vertical column then horizontal
    for r in range(spawn_len):
        tags = ("spawn",) if r == 0 else ()
        add(r, 0, tags=tags)
    add(spawn_len - 1, 0)  # corner B
    for c in range(1, seg1 + 1):
        add(spawn_len - 1, c)
    for r in range(spawn_len, spawn_len + seg2):
        add(r, seg1)
    for c in range(seg1 + 1, seg1 + goal_len + 1):
        add(spawn_len + seg2 - 1, c)

    traversable: set[frozenset[int]] = set()
    all_pairs: list[frozenset[int]] = []

    def link(a: int, b: int) -> None:
        pair = frozenset({a, b})
        all_pairs.append(pair)
        traversable.add(pair)

    # Walk the cells in declaration order, connecting consecutive ones.
    declared = list(nodes)
    for i in range(len(declared) - 1):
        link(declared[i].node_id, declared[i + 1].node_id)

    walls = _walls_for_motif(nodes=nodes, traversable=traversable, geometry=geometry)
    landmarks = _two_landmarks_at_nodes(
        nodes=nodes,
        first=declared[-1].node_id,
        second=declared[0].node_id,
    )
    spawn_xy = declared[0].center_xy_m
    world_half = max(seg1 + goal_len + 1, spawn_len + seg2) * cell + geometry.wall_thickness_m + 1.0
    return tuple(nodes), traversable, all_pairs, walls, (), landmarks, spawn_xy, world_half


def _motif_doorway(*, rng: random.Random, geometry: CorridorGeometry):
    """Two open rooms connected by a single narrow doorway.

    Doorway width = corridor_width (already narrow per sampling). Rooms are
    2x2 cell areas with a single landmark in each.
    """

    cell = geometry.cell_size_m
    width = geometry.corridor_width_m
    cells: dict[tuple[int, int], int] = {}
    nodes: list[GraphNode] = []
    room_side = rng.randint(2, 3)
    gap_offset = rng.randint(0, room_side - 1)

    def add(row: int, col: int, tags: tuple[str, ...] = ()) -> int:
        if (row, col) in cells:
            return cells[(row, col)]
        node_id = len(nodes)
        cells[(row, col)] = node_id
        nodes.append(
            GraphNode(
                node_id=node_id,
                center_xy_m=(round(col * cell, 4), round(row * cell, 4)),
                width_m=width,
                tags=tags,
            )
        )
        return node_id

    # Room A (rows 0..room_side-1) at columns 0..room_side-1, spawn at (0,0)
    for r in range(room_side):
        for c in range(room_side):
            tags = ("spawn",) if (r, c) == (0, 0) else ()
            add(r, c, tags=tags)
    # Doorway cell linking rooms
    door_row = gap_offset
    door_col = room_side
    add(door_row, door_col)
    # Room B at columns room_side+1..2*room_side, same row range
    for r in range(room_side):
        for c in range(room_side + 1, 2 * room_side + 1):
            add(r, c)

    traversable: set[frozenset[int]] = set()
    all_pairs: list[frozenset[int]] = []

    def link(a: int, b: int) -> None:
        pair = frozenset({a, b})
        all_pairs.append(pair)
        traversable.add(pair)

    # Connect all cells inside each room.
    for r in range(room_side):
        for c in range(room_side - 1):
            link(cells[(r, c)], cells[(r, c + 1)])
    for r in range(room_side - 1):
        for c in range(room_side):
            link(cells[(r, c)], cells[(r + 1, c)])
    # Same for room B
    for r in range(room_side):
        for c in range(room_side + 1, 2 * room_side):
            link(cells[(r, c)], cells[(r, c + 1)])
    for r in range(room_side - 1):
        for c in range(room_side + 1, 2 * room_side + 1):
            link(cells[(r, c)], cells[(r + 1, c)])
    # Doorway links
    link(cells[(door_row, room_side - 1)], cells[(door_row, door_col)])
    link(cells[(door_row, door_col)], cells[(door_row, door_col + 1)])

    walls = _walls_for_motif(nodes=nodes, traversable=traversable, geometry=geometry)
    landmarks = _two_landmarks_at_nodes(
        nodes=nodes,
        first=cells[(room_side - 1, 0)],
        second=cells[(room_side - 1, 2 * room_side)],
    )
    spawn_xy = nodes[cells[(0, 0)]].center_xy_m
    world_half = (2 * room_side + 1) * cell * 0.6 + geometry.wall_thickness_m + 1.0
    return tuple(nodes), traversable, all_pairs, walls, (), landmarks, spawn_xy, world_half


def _motif_slalom(*, rng: random.Random, geometry: CorridorGeometry):
    """Straight corridor with alternating offset obstacles forcing a slalom."""

    cell = geometry.cell_size_m
    width = geometry.corridor_width_m
    length = rng.randint(5, 7)
    cells: dict[tuple[int, int], int] = {}
    nodes: list[GraphNode] = []

    def add(row: int, col: int, tags: tuple[str, ...] = ()) -> int:
        node_id = len(nodes)
        cells[(row, col)] = node_id
        nodes.append(
            GraphNode(
                node_id=node_id,
                center_xy_m=(round(col * cell, 4), round(row * cell, 4)),
                width_m=width,
                tags=tags,
            )
        )
        return node_id

    for r in range(length):
        tags = ("spawn",) if r == 0 else ()
        add(r, 0, tags=tags)

    traversable: set[frozenset[int]] = set()
    all_pairs: list[frozenset[int]] = []
    for r in range(length - 1):
        pair = frozenset({cells[(r, 0)], cells[(r + 1, 0)]})
        all_pairs.append(pair)
        traversable.add(pair)

    walls = _walls_for_motif(nodes=nodes, traversable=traversable, geometry=geometry)

    obstacle_size = (round(width * 0.45, 3), round(cell * 0.30, 3), 0.60)
    obstacles: list[BoxObject] = []
    # Alternate y-offsets so the robot has to weave.
    for idx, r in enumerate(range(1, length - 1)):
        offset = (width * 0.30) * (1 if idx % 2 == 0 else -1)
        obstacles.append(
            BoxObject(
                object_id=f"slalom_pillar_{idx:02d}",
                kind="box_obstacle",
                center_xyz_m=(round(offset, 3), round(r * cell, 4), 0.30),
                size_xyz_m=obstacle_size,
                yaw_rad=0.0,
                material_id="mat_obstacle_0",
            )
        )

    landmarks = _two_landmarks_at_nodes(
        nodes=nodes,
        first=cells[(length - 1, 0)],
        second=cells[(0, 0)],
    )
    spawn_xy = nodes[cells[(0, 0)]].center_xy_m
    world_half = length * cell * 0.6 + geometry.wall_thickness_m + 1.0
    return tuple(nodes), traversable, all_pairs, walls, tuple(obstacles), landmarks, spawn_xy, world_half


def _motif_short_dead_end(*, rng: random.Random, geometry: CorridorGeometry):
    """Straight corridor with a single short dead-end branch."""

    cell = geometry.cell_size_m
    width = geometry.corridor_width_m
    main_len = rng.randint(4, 6)
    branch_len = rng.randint(1, 3)
    branch_at = rng.randint(1, main_len - 2)
    branch_dir = rng.choice((-1, 1))

    cells: dict[tuple[int, int], int] = {}
    nodes: list[GraphNode] = []

    def add(row: int, col: int, tags: tuple[str, ...] = ()) -> int:
        node_id = len(nodes)
        cells[(row, col)] = node_id
        nodes.append(
            GraphNode(
                node_id=node_id,
                center_xy_m=(round(col * cell, 4), round(row * cell, 4)),
                width_m=width,
                tags=tags,
            )
        )
        return node_id

    for r in range(main_len):
        tags = ("spawn",) if r == 0 else ()
        add(r, 0, tags=tags)
    for k in range(1, branch_len + 1):
        add(branch_at, branch_dir * k)

    traversable: set[frozenset[int]] = set()
    all_pairs: list[frozenset[int]] = []

    def link(a: int, b: int) -> None:
        pair = frozenset({a, b})
        all_pairs.append(pair)
        traversable.add(pair)

    for r in range(main_len - 1):
        link(cells[(r, 0)], cells[(r + 1, 0)])
    prev = cells[(branch_at, 0)]
    for k in range(1, branch_len + 1):
        link(prev, cells[(branch_at, branch_dir * k)])
        prev = cells[(branch_at, branch_dir * k)]

    walls = _walls_for_motif(nodes=nodes, traversable=traversable, geometry=geometry)
    landmarks = _two_landmarks_at_nodes(
        nodes=nodes,
        first=cells[(main_len - 1, 0)],
        second=cells[(branch_at, branch_dir * branch_len)],
    )
    spawn_xy = nodes[cells[(0, 0)]].center_xy_m
    world_half = max(main_len, branch_len + 2) * cell + geometry.wall_thickness_m + 1.0
    return tuple(nodes), traversable, all_pairs, walls, (), landmarks, spawn_xy, world_half


_MOTIF_BUILDERS: dict[str, Callable[..., tuple]] = {
    "t_junction": _motif_t_junction,
    "s_bend": _motif_s_bend,
    "doorway": _motif_doorway,
    "slalom": _motif_slalom,
    "short_dead_end": _motif_short_dead_end,
}


# ---------------------------------------------------------------------------
# Family: rough / local dynamics
# ---------------------------------------------------------------------------


def _build_rough_local_dynamics(ctx: BuildContext) -> SceneManifest:
    """Open arena with ramps, low steps, and friction variation.

    Topology is a 3x3 graph so privileged labels still work, but the scene is
    dominated by physics features (ramps, steps, friction-stress floor patches)
    rather than walls. Visual randomization is mild; physics randomization is
    the wide :data:`ROUGH_DYNAMICS_PROFILE`.
    """

    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    world_half = 5.0
    grid_side = 3
    spacing = 2.4

    nodes = _grid_nodes(grid_side=grid_side, spacing=spacing, node_width=1.2)
    edges = _grid_edges(grid_side=grid_side, edge_width=1.0)

    obstacles: list[BoxObject] = []
    # 1–3 ramps (tilted slabs).
    ramp_count = rng.randint(1, 3)
    for idx in range(ramp_count):
        ramp = _build_ramp(rng, world_half=world_half - 0.5, index=idx)
        if ramp is not None:
            obstacles.append(ramp)

    # 1–2 small steps (raised low platforms).
    step_count = rng.randint(1, 2)
    for idx in range(step_count):
        step = _build_step(rng, world_half=world_half - 0.5, index=idx)
        if step is not None:
            obstacles.append(step)

    # 0–2 slick patches (very flat box objects with high-restitution
    # visualization). Treated as obstacles with low height so the robot can
    # walk on them; the physics profile already moves friction globally.
    patch_count = rng.randint(0, 2)
    for idx in range(patch_count):
        patch = _build_slick_patch(rng, world_half=world_half - 1.0, index=idx)
        if patch is not None:
            obstacles.append(patch)

    landmarks = _corner_landmarks(world_half=world_half)
    extra_palette = ("ramp_concrete", "step_platform", "slick_patch")
    return _assemble(
        ctx=ctx,
        world_half=world_half,
        nodes=nodes,
        edges=edges,
        walls=(),
        obstacles=tuple(obstacles),
        landmarks=landmarks,
        profile=ROUGH_DYNAMICS_PROFILE,
        extra_palette=extra_palette,
    )


def _build_ramp(rng: random.Random, *, world_half: float, index: int) -> BoxObject | None:
    pitch_deg = rng.uniform(6.0, 14.0)
    pitch = math.radians(pitch_deg)
    length = round(rng.uniform(1.0, 1.8), 3)
    width = round(rng.uniform(0.8, 1.2), 3)
    thickness = 0.05
    cx = round(rng.uniform(-world_half + 1.0, world_half - 1.0), 3)
    cy = round(rng.uniform(-world_half + 1.0, world_half - 1.0), 3)
    # Lift center so the lower edge sits on the ground.
    cz = round((length * 0.5) * math.sin(pitch) + thickness * 0.5, 3)
    yaw = round(rng.uniform(-math.pi, math.pi), 4)
    return BoxObject(
        object_id=f"ramp_{index:02d}",
        kind="ramp",
        center_xyz_m=(cx, cy, cz),
        size_xyz_m=(length, width, thickness),
        yaw_rad=yaw,
        material_id="ramp_concrete",
        pitch_rad=round(pitch, 4),
    )


def _build_step(rng: random.Random, *, world_half: float, index: int) -> BoxObject | None:
    height = round(rng.uniform(0.04, 0.10), 3)
    size_x = round(rng.uniform(0.7, 1.4), 3)
    size_y = round(rng.uniform(0.7, 1.4), 3)
    cx = round(rng.uniform(-world_half + 0.8, world_half - 0.8), 3)
    cy = round(rng.uniform(-world_half + 0.8, world_half - 0.8), 3)
    return BoxObject(
        object_id=f"step_{index:02d}",
        kind="step",
        center_xyz_m=(cx, cy, round(height * 0.5, 3)),
        size_xyz_m=(size_x, size_y, height),
        yaw_rad=round(rng.uniform(-math.pi, math.pi), 4),
        material_id="step_platform",
    )


def _build_slick_patch(
    rng: random.Random, *, world_half: float, index: int
) -> BoxObject | None:
    size_x = round(rng.uniform(0.8, 1.6), 3)
    size_y = round(rng.uniform(0.8, 1.6), 3)
    thickness = 0.01
    cx = round(rng.uniform(-world_half, world_half), 3)
    cy = round(rng.uniform(-world_half, world_half), 3)
    return BoxObject(
        object_id=f"slick_patch_{index:02d}",
        kind="slick_patch",
        center_xyz_m=(cx, cy, round(thickness * 0.5, 3)),
        size_xyz_m=(size_x, size_y, thickness),
        yaw_rad=round(rng.uniform(-math.pi, math.pi), 4),
        material_id="slick_patch",
    )


# ---------------------------------------------------------------------------
# Family: visual / sensor stress
# ---------------------------------------------------------------------------


def _build_visual_sensor_stress(ctx: BuildContext) -> SceneManifest:
    """Medium-maze topology with aggressive visual randomization.

    Uses the same maze topology generator as ``medium_enclosed_maze`` so the
    underlying navigation problem stays comparable, but draws materials,
    lighting, distractors, and camera jitter from
    :data:`VISUAL_STRESS_PROFILE` for the rendered observations.
    """

    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    side = rng.randint(6, 8)
    geometry = sample_corridor_geometry(rng)
    cell_size = geometry.cell_size_m
    corridor_width = geometry.corridor_width_m
    wall_thickness = geometry.wall_thickness_m

    world_half = (side * cell_size) / 2.0 + wall_thickness

    nodes, traversable, all_pairs = _maze_topology(
        rng=rng,
        side=side,
        cell_size=cell_size,
        corridor_width=corridor_width,
        extra_loop_density=0.10,
    )
    edges = _maze_edges(
        traversable=traversable,
        all_pairs=all_pairs,
        corridor_width=corridor_width,
    )
    walls = _maze_walls(
        side=side,
        cell_size=cell_size,
        wall_thickness=wall_thickness,
        wall_height=WALL_HEIGHT_M,
        traversable=traversable,
    )
    landmarks = _maze_landmarks(
        rng=rng,
        nodes=nodes,
        pairs=2,
        wall_height=WALL_HEIGHT_M,
    )
    # Distractor bounds are the maze interior so distractors land inside the
    # walls (they get culled if they overlap, but the spawn step is best-effort).
    distractor_bounds = max(0.5, world_half - cell_size)
    return _assemble(
        ctx=ctx,
        world_half=world_half,
        nodes=nodes,
        edges=edges,
        walls=walls,
        obstacles=(),
        landmarks=landmarks,
        spawn_xy=nodes[0].center_xy_m,
        profile=VISUAL_STRESS_PROFILE,
        extra_palette=("distractor_pole",),
        distractor_bounds_half_m=distractor_bounds,
    )


# ---------------------------------------------------------------------------
# Shared geometry / topology helpers
# ---------------------------------------------------------------------------


def _grid_nodes(
    *, grid_side: int, spacing: float, node_width: float
) -> tuple[GraphNode, ...]:
    offset = (grid_side - 1) * spacing * 0.5
    nodes: list[GraphNode] = []
    node_id = 0
    for row in range(grid_side):
        for col in range(grid_side):
            x = round(col * spacing - offset, 3)
            y = round(row * spacing - offset, 3)
            tags = ("spawn",) if row == 0 and col == 0 else ()
            nodes.append(
                GraphNode(node_id=node_id, center_xy_m=(x, y), width_m=node_width, tags=tags)
            )
            node_id += 1
    return tuple(nodes)


def _grid_edges(*, grid_side: int, edge_width: float) -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []
    for row in range(grid_side):
        for col in range(grid_side):
            node = row * grid_side + col
            if col + 1 < grid_side:
                edges.append(GraphEdge(source=node, target=node + 1, width_m=edge_width))
            if row + 1 < grid_side:
                edges.append(GraphEdge(source=node, target=node + grid_side, width_m=edge_width))
    return tuple(edges)


def _scatter_box_obstacles(
    *,
    rng: random.Random,
    count: int,
    bounds_half: float,
    size_x_range: tuple[float, float],
    size_y_range: tuple[float, float],
    size_z_range: tuple[float, float],
    keepout_radius: float,
) -> tuple[BoxObject, ...]:
    obstacles: list[BoxObject] = []
    for index in range(count):
        size_x = round(rng.uniform(*size_x_range), 3)
        size_y = round(rng.uniform(*size_y_range), 3)
        size_z = round(rng.uniform(*size_z_range), 3)
        x = round(rng.uniform(-bounds_half, bounds_half), 3)
        y = round(rng.uniform(-bounds_half, bounds_half), 3)
        if abs(x) < keepout_radius and abs(y) < keepout_radius:
            x += keepout_radius + 0.4
        obstacles.append(
            BoxObject(
                object_id=f"obstacle_{index:03d}",
                kind="box_obstacle",
                center_xyz_m=(x, y, size_z * 0.5),
                size_xyz_m=(size_x, size_y, size_z),
                yaw_rad=round(rng.uniform(-math.pi, math.pi), 4),
                material_id=f"mat_obstacle_{index % 3}",
            )
        )
    return tuple(obstacles)


def _corner_landmarks(*, world_half: float) -> tuple[BoxObject, ...]:
    pos = world_half - 0.6
    return (
        BoxObject(
            object_id="landmark_red",
            kind="landmark",
            center_xyz_m=(-pos, pos, 0.75),
            size_xyz_m=(0.35, 0.35, 1.5),
            yaw_rad=0.0,
            material_id="landmark_red",
        ),
        BoxObject(
            object_id="landmark_blue",
            kind="landmark",
            center_xyz_m=(pos, -pos, 0.75),
            size_xyz_m=(0.35, 0.35, 1.5),
            yaw_rad=0.0,
            material_id="landmark_blue",
        ),
    )


# ---- Maze helpers --------------------------------------------------------


def _maze_topology(
    *,
    rng: random.Random,
    side: int,
    cell_size: float,
    corridor_width: float,
    extra_loop_density: float,
) -> tuple[tuple[GraphNode, ...], set[frozenset[int]], list[frozenset[int]]]:
    """Return graph nodes plus the set of traversable adjacency pairs."""

    offset = (side - 1) * cell_size * 0.5
    nodes: list[GraphNode] = []
    cell_to_id: dict[tuple[int, int], int] = {}
    node_id = 0
    for row in range(side):
        for col in range(side):
            x = round(col * cell_size - offset, 3)
            y = round(row * cell_size - offset, 3)
            tags = ("spawn",) if row == 0 and col == 0 else ()
            nodes.append(
                GraphNode(node_id=node_id, center_xy_m=(x, y), width_m=corridor_width, tags=tags)
            )
            cell_to_id[(row, col)] = node_id
            node_id += 1

    all_pairs: list[frozenset[int]] = []
    candidate_pairs: list[frozenset[int]] = []
    for row in range(side):
        for col in range(side):
            here = cell_to_id[(row, col)]
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < side and 0 <= nc < side:
                    pair = frozenset({here, cell_to_id[(nr, nc)]})
                    all_pairs.append(pair)
                    candidate_pairs.append(pair)

    # Randomized DFS spanning tree.
    visited: set[tuple[int, int]] = {(0, 0)}
    stack: list[tuple[int, int]] = [(0, 0)]
    traversable: set[frozenset[int]] = set()
    while stack:
        row, col = stack[-1]
        neighbors: list[tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in visited:
                neighbors.append((nr, nc))
        if neighbors:
            rng.shuffle(neighbors)
            nr, nc = neighbors[0]
            traversable.add(frozenset({cell_to_id[(row, col)], cell_to_id[(nr, nc)]}))
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # Optional extra cycles.
    remaining = [p for p in candidate_pairs if p not in traversable]
    rng.shuffle(remaining)
    extras = int(round(extra_loop_density * len(candidate_pairs)))
    for pair in remaining[: max(0, extras)]:
        traversable.add(pair)

    return tuple(nodes), traversable, all_pairs


def _maze_edges(
    *,
    traversable: set[frozenset[int]],
    all_pairs: list[frozenset[int]],
    corridor_width: float,
) -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []
    for pair in all_pairs:
        a, b = sorted(pair)
        edges.append(
            GraphEdge(
                source=a,
                target=b,
                width_m=corridor_width,
                traversable=pair in traversable,
            )
        )
    return tuple(edges)


def _maze_walls(
    *,
    side: int,
    cell_size: float,
    wall_thickness: float,
    wall_height: float,
    traversable: set[frozenset[int]],
) -> tuple[BoxObject, ...]:
    offset = (side - 1) * cell_size * 0.5
    half_extent = side * cell_size * 0.5
    walls: list[BoxObject] = []

    def cell_id(row: int, col: int) -> int:
        return row * side + col

    wall_index = 0

    # Interior walls between adjacent cells whose edge is not traversable.
    for row in range(side):
        for col in range(side):
            here = cell_id(row, col)
            for dr, dc, axis in ((0, 1, "x"), (1, 0, "y")):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < side and 0 <= nc < side):
                    continue
                pair = frozenset({here, cell_id(nr, nc)})
                if pair in traversable:
                    continue
                if axis == "x":
                    cx = round((col + 0.5) * cell_size - offset, 3)
                    cy = round(row * cell_size - offset, 3)
                    size = (cell_size, wall_thickness, wall_height)
                else:
                    cx = round(col * cell_size - offset, 3)
                    cy = round((row + 0.5) * cell_size - offset, 3)
                    size = (wall_thickness, cell_size, wall_height)
                walls.append(
                    BoxObject(
                        object_id=f"wall_int_{wall_index:04d}",
                        kind="wall",
                        center_xyz_m=(cx, cy, wall_height * 0.5),
                        size_xyz_m=size,
                        yaw_rad=0.0,
                        material_id="wall_interior",
                    )
                )
                wall_index += 1

    # Outer perimeter (four boxes).
    outer = half_extent + wall_thickness * 0.5
    span = side * cell_size + 2 * wall_thickness
    for index, (cx, cy, sx, sy) in enumerate(
        (
            (0.0, outer, span, wall_thickness),
            (0.0, -outer, span, wall_thickness),
            (outer, 0.0, wall_thickness, span),
            (-outer, 0.0, wall_thickness, span),
        )
    ):
        walls.append(
            BoxObject(
                object_id=f"wall_outer_{index}",
                kind="wall",
                center_xyz_m=(round(cx, 3), round(cy, 3), wall_height * 0.5),
                size_xyz_m=(sx, sy, wall_height),
                yaw_rad=0.0,
                material_id="wall_perimeter",
            )
        )

    return tuple(walls)


def _maze_landmarks(
    *,
    rng: random.Random,
    nodes: tuple[GraphNode, ...],
    pairs: int,
    wall_height: float,
) -> tuple[BoxObject, ...]:
    palette = ("landmark_red", "landmark_blue", "landmark_green", "landmark_yellow")
    height = max(0.6, wall_height * 1.1)
    sample = rng.sample(nodes[1:], k=min(len(nodes) - 1, pairs * 2))
    landmarks: list[BoxObject] = []
    for index, node in enumerate(sample):
        x, y = node.center_xy_m
        material = palette[index % len(palette)]
        landmarks.append(
            BoxObject(
                object_id=f"landmark_{index:02d}_{material}",
                kind="landmark",
                center_xyz_m=(round(x, 3), round(y, 3), round(height * 0.5, 3)),
                size_xyz_m=(0.3, 0.3, height),
                yaw_rad=0.0,
                material_id=material,
            )
        )
    return tuple(landmarks)


def _aliased_landmarks(
    *, nodes: tuple[GraphNode, ...], wall_height: float
) -> tuple[BoxObject, ...]:
    """Place two repeated red/blue pairs to create explicit visual aliases."""

    height = max(0.6, wall_height * 1.1)
    interior = [n for n in nodes if "spawn" not in n.tags]
    if len(interior) < 4:
        return ()
    by_xy = sorted(interior, key=lambda n: (n.center_xy_m[0], n.center_xy_m[1]))
    picks = (by_xy[0], by_xy[-1], by_xy[len(by_xy) // 4], by_xy[-len(by_xy) // 4 - 1])
    materials = ("landmark_red", "landmark_red", "landmark_blue", "landmark_blue")
    landmarks: list[BoxObject] = []
    for index, (node, material) in enumerate(zip(picks, materials)):
        x, y = node.center_xy_m
        landmarks.append(
            BoxObject(
                object_id=f"landmark_alias_{index:02d}_{material}",
                kind="landmark",
                center_xyz_m=(round(x, 3), round(y, 3), round(height * 0.5, 3)),
                size_xyz_m=(0.3, 0.3, height),
                yaw_rad=0.0,
                material_id=material,
            )
        )
    return tuple(landmarks)


def _two_landmarks_at_nodes(
    *, nodes: list[GraphNode], first: int, second: int
) -> tuple[BoxObject, ...]:
    height = max(0.6, WALL_HEIGHT_M * 1.1)
    output: list[BoxObject] = []
    for index, (node_id, material) in enumerate(((first, "landmark_red"), (second, "landmark_blue"))):
        node = nodes[node_id]
        x, y = node.center_xy_m
        output.append(
            BoxObject(
                object_id=f"landmark_{index:02d}_{material}",
                kind="landmark",
                center_xyz_m=(round(x, 3), round(y, 3), round(height * 0.5, 3)),
                size_xyz_m=(0.3, 0.3, height),
                yaw_rad=0.0,
                material_id=material,
            )
        )
    return tuple(output)


def _walls_for_motif(
    *,
    nodes: list[GraphNode],
    traversable: set[frozenset[int]],
    geometry: CorridorGeometry,
) -> tuple[BoxObject, ...]:
    """Build wall slabs around an irregular (non-grid) motif graph.

    For each node, check the four cardinal neighbours; if no node exists at
    that neighbour or the connection is non-traversable, emit a wall slab
    on that side of the node. Outer-perimeter walls fall out naturally.
    """

    cell = geometry.cell_size_m
    wall_thickness = geometry.wall_thickness_m
    wall_height = WALL_HEIGHT_M
    by_xy: dict[tuple[float, float], GraphNode] = {n.center_xy_m: n for n in nodes}
    walls: list[BoxObject] = []
    half = cell * 0.5
    eps = cell * 0.05

    def has_traversable(a: GraphNode, dx: float, dy: float) -> bool:
        neighbour_xy = (round(a.center_xy_m[0] + dx, 4), round(a.center_xy_m[1] + dy, 4))
        b = by_xy.get(neighbour_xy)
        if b is None:
            return False
        return frozenset({a.node_id, b.node_id}) in traversable

    wall_index = 0
    for node in nodes:
        x, y = node.center_xy_m
        # Right wall (+x)
        if not has_traversable(node, cell, 0.0):
            walls.append(
                BoxObject(
                    object_id=f"wall_motif_{wall_index:04d}",
                    kind="wall",
                    center_xyz_m=(round(x + half, 3), round(y, 3), wall_height * 0.5),
                    size_xyz_m=(wall_thickness, cell + eps, wall_height),
                    yaw_rad=0.0,
                    material_id="wall_perimeter",
                )
            )
            wall_index += 1
        # Top wall (+y)
        if not has_traversable(node, 0.0, cell):
            walls.append(
                BoxObject(
                    object_id=f"wall_motif_{wall_index:04d}",
                    kind="wall",
                    center_xyz_m=(round(x, 3), round(y + half, 3), wall_height * 0.5),
                    size_xyz_m=(cell + eps, wall_thickness, wall_height),
                    yaw_rad=0.0,
                    material_id="wall_perimeter",
                )
            )
            wall_index += 1
        # Left wall (-x); only emit when no neighbour exists at all so we don't
        # double-emit between two cells.
        neighbour_xy = (round(x - cell, 4), round(y, 4))
        if neighbour_xy not in by_xy:
            walls.append(
                BoxObject(
                    object_id=f"wall_motif_{wall_index:04d}",
                    kind="wall",
                    center_xyz_m=(round(x - half, 3), round(y, 3), wall_height * 0.5),
                    size_xyz_m=(wall_thickness, cell + eps, wall_height),
                    yaw_rad=0.0,
                    material_id="wall_perimeter",
                )
            )
            wall_index += 1
        # Bottom wall (-y); same edge-symmetry rule.
        neighbour_xy = (round(x, 4), round(y - cell, 4))
        if neighbour_xy not in by_xy:
            walls.append(
                BoxObject(
                    object_id=f"wall_motif_{wall_index:04d}",
                    kind="wall",
                    center_xyz_m=(round(x, 3), round(y - half, 3), wall_height * 0.5),
                    size_xyz_m=(cell + eps, wall_thickness, wall_height),
                    yaw_rad=0.0,
                    material_id="wall_perimeter",
                )
            )
            wall_index += 1
    return tuple(walls)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry() -> dict[str, FamilySpec]:
    return _REGISTRY


_REGISTRY: dict[str, FamilySpec] = {
    spec.name: spec
    for spec in (
        FamilySpec(
            name="open_obstacle_field",
            difficulty_tier="open",
            description="3x3 grid graph with scattered box obstacles and two corner landmarks.",
            builder=_build_open_obstacle_field,
        ),
        FamilySpec(
            name="local_composite_motifs",
            difficulty_tier="motif",
            description=(
                "Single-motif scenes (T-junction, S-bend, doorway, slalom, short dead-end) "
                "exercising one navigation skill per scene."
            ),
            builder=_build_local_composite_motifs,
        ),
        FamilySpec(
            name="small_enclosed_maze",
            difficulty_tier="small",
            description="4-5 cell side enclosed maze, short routes, one landmark pair.",
            builder=_enclosed_maze_builder(
                side_range=(4, 5), extra_loop_density=0.05, landmark_pairs=1
            ),
        ),
        FamilySpec(
            name="medium_enclosed_maze",
            difficulty_tier="medium",
            description="6-8 cell side enclosed maze, the main navigation distribution.",
            builder=_enclosed_maze_builder(
                side_range=(6, 8), extra_loop_density=0.10, landmark_pairs=2
            ),
        ),
        FamilySpec(
            name="large_enclosed_maze",
            difficulty_tier="large",
            description="9-12 cell side enclosed maze, long horizons for H-JEPA and routing.",
            builder=_enclosed_maze_builder(
                side_range=(9, 12), extra_loop_density=0.15, landmark_pairs=3
            ),
        ),
        FamilySpec(
            name="loop_alias_stress",
            difficulty_tier="alias",
            description="Medium maze with extra cycles and duplicated-color landmark pairs.",
            builder=_build_loop_alias_stress,
        ),
        FamilySpec(
            name="rough_local_dynamics",
            difficulty_tier="dynamics",
            description=(
                "Open arena with ramps, low steps, and slick patches under wide friction "
                "randomization."
            ),
            builder=_build_rough_local_dynamics,
        ),
        FamilySpec(
            name="visual_sensor_stress",
            difficulty_tier="visual",
            description=(
                "Medium-maze topology with aggressive lighting, texture, distractor, and "
                "camera-extrinsic randomization."
            ),
            builder=_build_visual_sensor_stress,
        ),
    )
}
