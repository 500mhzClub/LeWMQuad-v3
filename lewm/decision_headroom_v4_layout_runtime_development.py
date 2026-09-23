"""Read the frozen V4 layout identities; never generate layouts at execution."""
import json
from functools import lru_cache
from pathlib import Path
from lewm.dense_world_model_maze_layouts_development import generator
from lewm.eligible_floor_registration_development import bind

PATH=Path('docs/go2_decision_headroom_v4_layouts_2026-09-23.json')
@lru_cache(maxsize=1)
def inventory():return json.loads(PATH.read_text())
def specification(index):return inventory()['layouts'][index]['specification']
pack=bind(generator.pack,specification=specification)
public_mission=bind(generator.public_mission,specification=specification)
