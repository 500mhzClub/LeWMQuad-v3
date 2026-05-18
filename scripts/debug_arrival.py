
import math
import numpy as np
from pathlib import Path
from lewm_worlds.scene_graph import SceneGraph
from lewm_worlds.manifest import SceneManifest, GraphNode, GraphEdge, BoxObject, SpawnSpec, CameraValidityConstraints
from lewm_genesis.collectors.route_teacher import RouteTeacher
from lewm_genesis.collectors.base import EnvObservation

def _grid_corridor_manifest():
    cell = 1.0
    wall_t = 0.2
    wall_h = 0.8
    nodes = tuple(
        GraphNode(node_id=i, center_xy_m=(i * cell, 0.0), width_m=cell - wall_t, tags=("spawn",) if i == 0 else ())
        for i in range(4)
    )
    edges = tuple(GraphEdge(source=i, target=i + 1, width_m=cell - wall_t, traversable=True) for i in range(3))
    walls = ()
    landmarks = (
        BoxObject(object_id="goal_landmark", kind="landmark", center_xyz_m=(3.0, 0.0, 0.5), size_xyz_m=(0.3, 0.3, 1.0), yaw_rad=0.0, material_id="landmark_red"),
    )
    return SceneManifest(
        scene_id="grid_corridor", family="test", difficulty_tier="test", topology_seed=0, visual_seed=0, physics_seed=0,
        world_bounds_xy_m=((-1.0, -1.0), (4.0, 1.0)), spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.375), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=nodes, graph_edges=edges, obstacles=(), landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(min_wall_thickness_m=0.08, near_m=0.05, far_m=200.0, min_camera_clearance_m=0.10),
        walls=walls,
    )

def debug_arrival():
    manifest = _grid_corridor_manifest()
    graph = SceneGraph(manifest)
    
    from lewm_genesis.lewm_contract import PrimitiveRegistry
    registry = PrimitiveRegistry(block_size=5, command_dt_s=0.1, command_order=('vx_body_mps', 'vy_body_mps', 'yaw_rate_radps'), primitives={}, defaults={})
    
    collector = RouteTeacher(registry, n_envs=1, landmark_standoff_m=0.85)
    
    obs = EnvObservation(
        env_idx=0, base_xy_world=(2.2, 0.0), base_yaw_world=0.0, current_cell_id=2,
        nearest_cell_distance_m=0.0, clearance_to_walls_m=1.0, last_executed_cmd=(0,0,0),
        episode_id=1, episode_step=0, block_idx_in_episode=0
    )
    
    goal_cell = 3
    print(f"Goal: {goal_cell}, Robot XY: {obs.base_xy_world}, Cell: {obs.current_cell_id}")
    
    # Trace _is_arrived
    gx, gy = collector._arrival_anchor_xy(goal_cell, graph)
    dist_m = math.hypot(obs.base_xy_world[0] - gx, obs.base_xy_world[1] - gy)
    threshold_m = collector._arrival_distance_threshold(goal_cell, graph)
    print(f"Anchor: ({gx}, {gy}), Dist: {dist_m:.3f}, Threshold: {threshold_m:.3f}")
    
    visible = collector._landmark_visible(obs, goal_cell, graph)
    print(f"Visible: {visible}")
    
    is_landmark = getattr(graph, "landmark_xy_for_cell", lambda _: None)(goal_cell) is not None
    print(f"Is Landmark: {is_landmark}")
    
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    dist_hops = graph.bfs_distance(obs.current_cell_id, goal_cell, transit_blocked=blocked)
    print(f"BFS Hops: {dist_hops}")
    
    arrived = collector._is_arrived(obs, goal_cell, graph)
    print(f"Arrived: {arrived}")

if __name__ == "__main__":
    debug_arrival()
