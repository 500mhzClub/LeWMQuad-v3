"""Correct inherited corridor metadata using the verified instantiated maze.

Preserve original launch and evaluation records. Arrival/forecast scoring is
unchanged; only the corridor graph was taken from an older launch inventory.
"""
import hashlib
import json
from types import SimpleNamespace

from scripts import run_go2_sparse_corner_completion_development as run
from scripts.read_go2_interrupted_view_replan_development import physical_return_edges


def main():
    root = run.BASE/run.ROOT
    reference = run.BASE/run.transfer.root_name(1)
    read = lambda name: json.loads((root/name).read_text())
    plan = json.loads(run.PLAN.read_text())
    inventory_path = run.transfer.INVENTORY
    inventory = json.loads(inventory_path.read_text())
    digest = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    assert digest == plan['inventory_sha256']
    # The actual native objects, including collision geometry, must match the
    # completed transfer mission rather than merely trusting a descriptive flag.
    objects = read('native/static_objects.json')
    assert objects == json.loads((reference/'native/static_objects.json').read_text())
    previous_launch = json.loads((reference/'launch.json').read_text())
    assert previous_launch['fresh_layout_inventory'] == inventory
    for path in ('scripts/run_go2_route_turn_memory_transfer_development.py',
                 'lewm/route_turn_memory_transfer_layouts_development.py'):
        from pathlib import Path
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == plan['source_sha256'][path]
    corrected_launch = read('launch.json') | dict(fresh_layout_inventory=inventory,
        frozen_layout_inventory_sha256=digest)

    class EvaluationRoot:
        def __truediv__(self, name):
            if name == 'launch.json':
                return SimpleNamespace(read_text=lambda: json.dumps(corrected_launch))
            if name == 'physical_return_corridor_readout_v1.json':
                return root/'physical_return_corridor_readout_v2.json'
            return root/name

    backtracking = physical_return_edges(EvaluationRoot())
    original = read('frozen_readout_navigation_readout_v1.json')
    result = original | dict(schema='sparse_corner_completion_navigation_readout.v2',
        physical_backtracking=backtracking,
        corridor_inventory_corrected=True,
        original_launch_and_evaluation_preserved=True,
        native_static_objects_exactly_match_transfer_reference=True,
        instantiated_static_objects=len(objects), actual_inventory_sha256=digest,
        correction_scope='corridor graph only; original arrival, contact and forecast evaluation unchanged',
        supersedes_backtracking_of='frozen_readout_navigation_readout_v1.json')
    with (root/'sparse_corner_completion_navigation_readout_v2.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
