"""Fresh executions of two storage-interrupted cases; retain completed cases."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_task_goal_pilot_development as original

OUTPUT = original.OUTPUT.parent/'go2_dense_task_goal_pilot_storage_replacement_v1_attempt_001'
PLAN = Path('docs/go2_dense_task_goal_pilot_storage_replacement_plan_2026-09-17.json')
SOURCES = original.SOURCES+(__file__,)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    failures = []
    for case in (2,3):
        directory = original.OUTPUT/f'case_{case:02d}'
        failure = json.loads((directory/'failure.json').read_text())
        assert failure['reason'] == "RuntimeError('output storage reserve reached')"
        assert not (directory/'result.json').exists()
        failures.append(dict(case=case,path=str(directory),reason=failure['reason'],
            failure_sha256=original.pilot.base.digest(directory/'failure.json')))
    observed = []
    for case in (0,1):
        directory = original.OUTPUT/f'case_{case:02d}'
        assert json.loads((directory/'result.json').read_text())['status'] == 'COMPLETE'
        observed.append(sum(p.stat().st_size for p in directory.iterdir() if p.is_file()
            and p.name != 'sealed_test.json' and not p.name.startswith('sealed_')))
    required = 512*1024**2 + int(2.5*max(observed))
    assert shutil.disk_usage(OUTPUT.parent).free > required
    plan = json.loads(original.PLAN.read_text())|dict(
        source_sha256={p:original.pilot.base.digest(p) for p in SOURCES},
        original_plan_sha256=original.pilot.base.digest(original.PLAN),
        interrupted_attempts=failures,reexecute_cases=[2,3],completed_cases_retained=[0,1],
        case_directories={str(c):str((original.OUTPUT if c<2 else OUTPUT)/f'case_{c:02d}') for c in range(4)},
        change='exclusive output location after retiring unused completed diagnostic depth; model, tasks, seeds, commands and policy unchanged',
        resources=dict(output_free_bytes=shutil.disk_usage(OUTPUT.parent).free,
            observed_complete_case_bytes=observed,required_bytes_with_reserve_and_margin=required),
        depth_retirement_receipt='.generated/depth_retirement_dense_goal_diagnostics_2026-09-17/result.json')
    OUTPUT.mkdir(); original.pilot.base.save(PLAN,plan); original.pilot.base.save(OUTPUT/'plan.json',plan)
    print('STORAGE_REPLACEMENT_PREPARED',str(OUTPUT),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',type=int,choices=(2,3));a=p.parse_args()
    if a.prepare and a.case is None: prepare()
    elif not a.prepare and a.case is not None:
        bind(original.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCES=SOURCES)(a.case)
    else:p.error('prepare or execute one interrupted case from the start')
