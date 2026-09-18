"""Same paced input/deadlines, moving only routing-map work to a process."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_mapped_runtime_development import (
    initialize_mapping,mapping_ready,ProcessMappedRuntime)
from scripts import run_paced_multirate_recorded_prefix_development as source

OUTPUT=source.source.BASE/'go2_process_mapped_recorded_prefix_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(mapping_worker_separate_process=True,
            mapping_process_ready_before_stream=True,
            extra_sources={p:source.source.source.digest(Path(p)) for p in (
                'lewm/process_mapped_runtime_development.py',
                'scripts/run_process_mapped_recorded_prefix_development.py')})
    if name=='result.json':
        value=value|dict(status='PROCESS_MAPPED_RECORDED_PREFIX_COMPLETE',
            mapping_worker_separate_process=True,packet_transfer_included_in_stage_timing=True)
    _write(name,value)


def main():
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=initialize_mapping) as executor:
        assert executor.submit(mapping_ready).result()
        def controller(*args,**kwargs):return ProcessMappedRuntime(*args,mapping_executor=executor,**kwargs)
        run=bind(source.main,OUTPUT=OUTPUT,write=write,PacedMultirateController=controller)
        run()


if __name__=='__main__':main()
