"""Serial successor: deferred auxiliary descriptors on the full sensor history."""
import json
from pathlib import Path
import time

import psutil

from lewm.eligible_floor_registration_development import bind
from lewm.full_consensus_tracker_development import FullConsensusVisualMotion
from lewm.lazy_auxiliary_features_development import LazyAuxiliaryVisualMotion
from scripts import compare_full_consensus_recorded_tracker_development as source

OUTPUT = source.source.trial.run.BASE/'go2_lazy_auxiliary_recorded_tracker_v1_attempt_001'
_write = bind(source.write, OUTPUT=OUTPUT)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(candidate='lazy_auxiliary_descriptors_with_full_consensus',
            paired_baseline='full_consensus_with_eager_auxiliary_descriptors',
            predecessor=str(source.OUTPUT),
            extra_sources={p:source.source.digest(Path(p)) for p in (
                'lewm/lazy_auxiliary_features_development.py',
                'scripts/compare_lazy_auxiliary_recorded_tracker_development.py')})
    elif name == 'result.json':
        value = value | dict(status='LAZY_AUXILIARY_RECORDED_TRACKER_COMPLETE',
            auxiliary_features_deferred=True, descriptor_and_fit_rules_unchanged=True)
    _write(name, value)


main = bind(source.main, OUTPUT=OUTPUT, write=write,
    FullConsensusVisualMotion=LazyAuxiliaryVisualMotion,
    SampledPlaneChainedVisualMotion=FullConsensusVisualMotion)


if __name__ == '__main__':
    me = psutil.Process()
    print('LAZY_AUXILIARY_COMPARISON_WAITING', me.pid, me.create_time(), flush=True)
    while True:
        try:
            previous = psutil.Process(3199419)
            live = abs(previous.create_time()-1789287181.58)<.01 and previous.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess: live = False
        if not live: break
        time.sleep(5)
    assert not (source.OUTPUT/'failure.json').exists()
    assert json.loads((source.OUTPUT/'result.json').read_text())['status'] == 'FULL_CONSENSUS_RECORDED_TRACKER_COMPLETE'
    main()
