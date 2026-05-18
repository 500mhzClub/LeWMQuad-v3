
import os
import json
import math
import numpy as np
from pathlib import Path
from lewm_worlds.scene_graph import SceneGraph
from lewm_worlds.manifest import SceneManifest

def analyze_scene(scene_dir):
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.exists():
        manifest_path = scene_dir / "genesis_scene.json"
    
    from lewm_worlds.manifest import SceneManifest, parse_scene_manifest_dict
    
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    
    manifest = parse_scene_manifest_dict(manifest_data)
    graph = SceneGraph(manifest)
    
    spawn_cell = graph.spawn_cells()[0]
    landmarks = graph.landmark_cells
    
    print(f"Scene: {manifest.scene_id}")
    print(f"Spawn cell: {spawn_cell} at {graph.cell_center(spawn_cell)}")
    
    # Check if the maze is disconnected for the robot
    reachable = graph.reachable_cells(spawn_cell, transit_blocked=graph.nav_blocked_cells)
    print(f"Total cells: {graph.n_nodes}, Reachable by robot: {len(reachable)}")
    
    for lm_id, lm_cell in landmarks:
        lm_xy = graph.landmark_xy_for_cell(lm_cell)
        print(f"Landmark {lm_id} at cell {lm_cell}, xy {lm_xy}")
        
        # Shortest path to this landmark
        path = graph.shortest_path(spawn_cell, lm_cell, transit_blocked=graph.nav_blocked_cells)
        if not path:
            print(f"  NO PATH to {lm_id} (Blocked by navigation graph)")
            # Find the path ignoring nav_blocked_cells to see what's being blocked
            unblocked_path = graph.shortest_path(spawn_cell, lm_cell)
            if unblocked_path:
                print(f"    Unblocked path exists ({len(unblocked_path)} hops). Checking bottlenecks:")
                for cell in unblocked_path:
                    if cell in graph.nav_blocked_cells:
                        xy = graph.cell_center(cell)
                        clearance = graph.clearance_to_walls(xy, include_landmarks=True)
                        reason = "Beacon cell" if cell in graph.beacon_cells_set else f"Low clearance ({clearance:.3f}m)"
                        print(f"      - Cell {cell} is BLOCKED: {reason}")
            continue
            
        print(f"  Path to {lm_id}: {len(path)} hops")
        
        # Check clearance along the path
        for i, cell in enumerate(path):
            xy = graph.cell_center(cell)
            clearance = graph.clearance_to_walls(xy, include_landmarks=True)
            print(f"    Step {i}: Cell {cell} at {xy}, clearance {clearance:.3f}m")
            
            # Check neighbors
            for neighbor in graph.neighbors(cell):
                n_xy = graph.cell_center(neighbor)
                n_clearance = graph.clearance_to_walls(n_xy, include_landmarks=True)
                n_blocked = neighbor in graph.nav_blocked_cells
                n_type = "BLOCKED" if n_blocked else "OPEN"
                print(f"      - Neighbor {neighbor} at {n_xy}: {n_type}, clearance {n_clearance:.3f}m")

            # Identify if this cell is "behind" a beacon
            for other_lm_id, other_lm_cell in landmarks:
                other_lm_xy = graph.landmark_xy_for_cell(other_lm_cell)
                dist_to_lm = math.hypot(xy[0] - other_lm_xy[0], xy[1] - other_lm_xy[1])
                if dist_to_lm < 1.0: # Close enough to be "around" it
                    print(f"      - Close to {other_lm_id} (dist {dist_to_lm:.3f}m)")

if __name__ == "__main__":
    analyze_scene(".generated/scene_corpus/acceptance/train/large_enclosed_maze/large_enclosed_maze_6edb8f490970")
