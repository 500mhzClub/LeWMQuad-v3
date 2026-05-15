"""Scene-family registry and deterministic per-family builders.

Each family produces a :class:`SceneManifest` from a single integer seed. The
geometry is intentionally compact but real: mazes use a randomized spanning
tree with optional extra cycles, walls are emitted along the non-traversable
edges, and open fields scatter box obstacles. This module is the source of
truth for what a "scene family" means in the corpus.
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


# Shared camera validity constraints — must satisfy the wall thickness used
# below so synthetic-render checks do not fail at the smallest cells.
_CAMERA_CONSTRAINTS = CameraValidityConstraints(
    min_wall_thickness_m=0.08,
    near_m=0.05,
    far_m=200.0,
    min_camera_clearance_m=0.10,
)


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
) -> Callable[[BuildContext], SceneManifest]:
    def builder(ctx: BuildContext) -> SceneManifest:
        rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
        side = rng.randint(*side_range)
        cell_size = 1.6
        corridor_width = 1.1
        wall_thickness = 0.12
        wall_height = 0.8

        world_half = (side * cell_size) / 2.0 + wall_thickness

        nodes, traversable, all_pairs = _maze_topology(
            rng=rng,
            side=side,
            cell_size=cell_size,
            corridor_width=corridor_width,
            extra_loop_density=extra_loop_density,
        )
        edges = _maze_edges(traversable=traversable, all_pairs=all_pairs, corridor_width=corridor_width)
        walls = _maze_walls(
            side=side,
            cell_size=cell_size,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            traversable=traversable,
        )
        landmarks = _maze_landmarks(
            rng=rng,
            nodes=nodes,
            pairs=landmark_pairs,
            wall_height=wall_height,
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
        )

    return builder


# ---------------------------------------------------------------------------
# Family: loop / alias stress
# ---------------------------------------------------------------------------


def _build_loop_alias_stress(ctx: BuildContext) -> SceneManifest:
    rng = random.Random(f"{ctx.family}:{ctx.scene_seed}")
    side = rng.randint(6, 7)
    cell_size = 1.6
    corridor_width = 1.1
    wall_thickness = 0.12
    wall_height = 0.8

    world_half = (side * cell_size) / 2.0 + wall_thickness

    nodes, traversable, all_pairs = _maze_topology(
        rng=rng,
        side=side,
        cell_size=cell_size,
        corridor_width=corridor_width,
        extra_loop_density=0.25,
    )
    edges = _maze_edges(traversable=traversable, all_pairs=all_pairs, corridor_width=corridor_width)
    walls = _maze_walls(
        side=side,
        cell_size=cell_size,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        traversable=traversable,
    )
    # Alias stress: two identical-colored landmark pairs at mirrored positions.
    landmarks = _aliased_landmarks(nodes=nodes, wall_height=wall_height)
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
# Shared helpers
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
    spawn_xy: tuple[float, float] | None = None,
) -> SceneManifest:
    spawn_x, spawn_y = spawn_xy if spawn_xy is not None else (0.0, 0.0)
    spawn = SpawnSpec(
        xyz_m=(float(spawn_x), float(spawn_y), 0.375),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    return SceneManifest(
        scene_id=stable_scene_id(ctx.family, ctx.scene_seed),
        family=ctx.family,
        difficulty_tier=ctx.difficulty_tier,
        topology_seed=int(ctx.scene_seed),
        visual_seed=int(ctx.scene_seed) + 100_000,
        physics_seed=int(ctx.scene_seed) + 200_000,
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
    )


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
    """Return graph nodes plus the set of traversable adjacency pairs.

    Edges are encoded as ``frozenset({node_a_id, node_b_id})``. ``all_pairs``
    enumerates every grid-adjacent pair, including non-traversable ones, so
    callers can emit the corresponding walls.
    """

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
    # Pick four well-spaced corner-ish nodes by extreme coordinates.
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
            name="small_enclosed_maze",
            difficulty_tier="small",
            description="4-5 cell side enclosed maze, short routes, one landmark pair.",
            builder=_enclosed_maze_builder(side_range=(4, 5), extra_loop_density=0.05, landmark_pairs=1),
        ),
        FamilySpec(
            name="medium_enclosed_maze",
            difficulty_tier="medium",
            description="6-8 cell side enclosed maze, the main navigation distribution.",
            builder=_enclosed_maze_builder(side_range=(6, 8), extra_loop_density=0.10, landmark_pairs=2),
        ),
        FamilySpec(
            name="large_enclosed_maze",
            difficulty_tier="large",
            description="9-12 cell side enclosed maze, long horizons for H-JEPA and routing.",
            builder=_enclosed_maze_builder(side_range=(9, 12), extra_loop_density=0.15, landmark_pairs=3),
        ),
        FamilySpec(
            name="loop_alias_stress",
            difficulty_tier="alias",
            description="Medium maze with extra cycles and duplicated-color landmark pairs.",
            builder=_build_loop_alias_stress,
        ),
    )
}
