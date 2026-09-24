"""Paced replay with separate pose and map processes, including transfer cost."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_mapped_runtime_development import (
    initialize_mapping,mapping_ready,initialize_pose,pose_ready,ProcessSeparatedRuntime)
from scripts import run_paced_multirate_recorded_prefix_development as source

OUTPUT=source.source.BASE/'go2_process_separated_recorded_prefix_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(mapping_and_pose_separate_processes=True,
            processes_ready_before_stream=True,
            extra_sources={p:source.source.source.digest(Path(p)) for p in (
                'lewm/process_mapped_runtime_development.py',
                'scripts/run_process_separated_recorded_prefix_development.py')})
    if name=='result.json':
        value=value|dict(status='PROCESS_SEPARATED_RECORDED_PREFIX_COMPLETE',
            mapping_and_pose_separate_processes=True,packet_transfer_included_in_stage_timing=True)
    _write(name,value)


def main():
    with (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_mapping) as mapping,
            ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_pose) as pose):
        assert mapping.submit(mapping_ready).result() and pose.submit(pose_ready).result()
        def controller(*args,**kwargs):
            return ProcessSeparatedRuntime(*args,mapping_executor=mapping,pose_executor=pose,**kwargs)
        bind(source.main,OUTPUT=OUTPUT,write=write,PacedMultirateController=controller)()


if __name__=='__main__':main()
