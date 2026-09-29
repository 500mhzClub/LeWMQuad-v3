"""Training-role rest-start / in-place-turn recording views on the registered C3-v2 recording mazes.

Six registered recording mazes (fit: layouts 10-13, held-out: 14-15) x four cardinal views,
chosen from maze geometry only, exactly as the maze-view training collection chose views.
"""
from dataclasses import replace
from functools import lru_cache
import json
import math
from pathlib import Path

from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind

SETS = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
            'go2_navigation_capability_v1_attempt_001/c3v2_sets_v1')
LAYOUTS = tuple(range(10, 16))
CASE_COUNT = 4*len(LAYOUTS)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@lru_cache(maxsize=1)
def specifications():
    registry = json.loads((SETS/'registry.json').read_text())
    entries = {e['maze_id']: e for e in registry['entries']}
    cases = []
    for layout in LAYOUTS:
        entry = entries[layout]
        assert entry['role'] == 'rest_turn_recording'
        base = json.loads(Path(entry['maze']['path']).read_text())
        adjacency = {tuple(c): [] for c in base['evaluation_layout']['cells']}
        for a, b in base['evaluation_layout']['edges']:
            adjacency[tuple(a)].append(tuple(b))
            adjacency[tuple(b)].append(tuple(a))
        used = set()
        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            preferred_degree = (1, 2, 3, 2)[direction_index]
            candidates = [c for c, neighbours in adjacency.items() if (c[0]+dx, c[1]+dy) in neighbours]
            cell = min(candidates, key=lambda c: (c in used, abs(len(adjacency[c])-preferred_degree), c))
            used.add(cell)
            yaw = math.atan2(dy, dx)
            case = len(cases)
            cases.append(base | dict(
                scene_id=f'c3v2-rest-turn-recording-case-{case:02d}', layout_index=case,
                recording_maze_id=layout, recording_split=entry['split'], data_role='train',
                procedural_seed=2026113300+case,
                training_context=dict(cell=list(cell), heading_rad=yaw, open_neighbour=[cell[0]+dx, cell[1]+dy],
                                      degree=len(adjacency[cell])),
                geometry=base['geometry'] | dict(spawn_se2_world=[cell[0]*generator.PITCH_M, cell[1]*generator.PITCH_M, yaw])))
    assert len(cases) == CASE_COUNT
    return tuple(cases)


def specification(index):
    if type(index) is not int or not 0 <= index < CASE_COUNT:
        raise ValueError('one of the fixed C3-v2 recording cases required')
    return specifications()[index]


def pack(spec):
    definition = bind(generator.pack, specification=specification)(spec)
    x, y, yaw = spec['geometry']['spawn_se2_world']
    return replace(definition, robot=replace(definition.robot, spawn_xyz_m=(x, y, .375),
        spawn_quat_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2))))
