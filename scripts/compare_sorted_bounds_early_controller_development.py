"""Isolate sorted voxel insertion in the combined controller's paired replay."""
from pathlib import Path
from types import SimpleNamespace
import json

from lewm.eligible_floor_registration_development import bind
from lewm.sorted_sample_bounds_development import SortedSampleBoundsIndex
from scripts import compare_cached_chain_early_controller_development as source

OUTPUT = Path('docs/go2_sorted_bounds_early_controller_2026-09-13.json')


class Baseline(source.Candidate):
    def _result(self, *args, **kwargs):
        value = super()._result(*args, **kwargs)
        value.pop('sampled_plane_candidates_enabled')
        return value


class Candidate(source.Candidate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        memory = self.memory
        assert not memory.route
        owners = [(memory, 'index'), (memory, 'auxiliary_index')]
        owners += [(partition, name) for partition in
            (memory.partition, memory.auxiliary_partition, memory.confirmed_auxiliary_partition)
            for name in ('floor', 'other')]
        for owner, name in owners:
            assert not getattr(owner, name).cells
            setattr(owner, name, SortedSampleBoundsIndex())


def dump(value, destination, **kwargs):
    if value.get('status') == 'CACHED_CHAIN_EARLY_CONTROLLER_COMPLETE':
        value = value | dict(status='SORTED_BOUNDS_EARLY_CONTROLLER_COMPLETE',
            baseline='combined_cached_chain_controller',
            candidate='same_controller_with_sorted_voxel_insertion',
            extra_sources={p:source.source.digest(Path(p)) for p in (
                'lewm/sorted_sample_bounds_development.py',
                'scripts/compare_sorted_bounds_early_controller_development.py')})
    return json.dump(value, destination, **kwargs)


original = SimpleNamespace(**vars(source.source))
original.StopConditionedSettlingController = Baseline
report_json = SimpleNamespace(loads=json.loads, dumps=json.dumps, dump=dump)
main = bind(source.main, OUTPUT=OUTPUT, FAILURE=OUTPUT.with_suffix('.failure.json'),
    source=original, Candidate=Candidate, json=report_json)


if __name__ == '__main__': main()
