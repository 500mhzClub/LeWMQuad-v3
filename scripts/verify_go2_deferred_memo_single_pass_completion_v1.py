"""Verify the complete paired timing result after the exact replay owner ends."""
import argparse
from datetime import datetime,timezone
import json
import math
import re
from scripts import run_go2_deferred_memo_single_pass_late_history_v1 as job
from scripts.startup_source_inventory_development import discover_sources

SOURCE='scripts/verify_go2_deferred_memo_single_pass_completion_v1.py'
TEST='lewm/tests/test_deferred_memo_single_pass_completion_development.py'
OUTPUT=job.ROOT/'docs/go2_deferred_memo_single_pass_completion_verification_2026-09-11.json'
LAUNCH_SHA='f7b6d7a7e49373f236019731454622fb2f8a473b29c0aeb826e0e0bdcbcdd8fb'
ARTIFACTS={'launch.json','comparison.jsonl'}


def ended(launch):
    if launch['boot_id'] != job.previous.Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('same original replay boot required')
    if job.previous.owner_live(launch['owner']): raise ValueError('original paired replay owner must end first')


def check_payload(result,launch,prior,rows):
    if (set(result) != {'status','source_sha256','artifact_sha256','report','sensing_scope',
                'original_inputs_reauthenticated_before_and_after','wall_s','native_execution',
                'real_time_qualified','navigation_qualified','goal_achieved'}
            or result['status'] != 'DEFERRED_MEMO_SINGLE_PASS_LATE_HISTORY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or set(result['artifact_sha256']) != ARTIFACTS
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or result['original_inputs_reauthenticated_before_and_after'] is not True
            or any(result[k] is not False for k in ('native_execution','real_time_qualified','navigation_qualified','goal_achieved'))
            or type(result['wall_s']) not in (int,float) or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0
            or result['sensing_scope'] != prior[1]['sensing_scope']):
        raise ValueError('complete original result, source, finite timing and negative scope required')
    expected=job.harness.scope() | dict(reference_completion_sha256=job.COMPLETION_SHA,
        reference_result_sha256=job.RESULT_SHA,reference_artifact_sha256=prior[0]['artifact_sha256'],
        last_cpu_completion_sha256=job.LAST_CPU_SHA,component_benchmark_sha256=job.BENCHMARK_SHA,
        environment=job.previous.ENVIRONMENT,frames=1428,state_frames=list(job.harness.original.STATE_FRAMES),
        native_execution=False,model_training=False,automatic_retry=False)
    if job.fingerprint({k:launch[k] for k in expected}) != job.fingerprint(expected):
        raise ValueError('exact controller pair, state scope, predecessor and launch environment required')
    return job.validate_report(result['report'],rows,prior)


def verify_result(result_sha):
    if (not __debug__ or any(job.previous.os.environ.get(k) != v for k,v in job.previous.ENVIRONMENT.items())
            or job.harness.original.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU verification environment required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion receipt required')
    if not isinstance(result_sha,str) or re.fullmatch('[0-9a-f]{64}',result_sha) is None:
        raise ValueError('actual completed result SHA-256 required')
    root=job.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve original paired replay failure')
    job.verify_artifacts(root,{'launch.json':LAUNCH_SHA})
    launch=json.loads((root/'launch.json').read_text());ended(launch)
    job.verify_artifacts(root,{'result.json':result_sha})
    result=json.loads((root/'result.json').read_text())
    job.verify_artifacts(root,result['artifact_sha256'])
    if (root/'comparison.jsonl').stat().st_size > 16*1024**2: raise ValueError('bounded complete comparison metadata required')
    original_sources=job.sources()
    if launch['source_sha256'] != original_sources: raise ValueError('exact original source manifest required')
    sources=discover_sources((SOURCE,TEST),original_sources);job.verify(sources)
    prior=job.admit()
    rows=[json.loads(line) for line in (root/'comparison.jsonl').read_text().splitlines()]
    timing=check_payload(result,launch,prior,rows)
    job.verify(sources);job.verify_artifacts(root,result['artifact_sha256'] | {'result.json':result_sha});ended(launch)
    total=timing['all_navigation']
    job.write_json(OUTPUT,dict(status='DEFERRED_MEMO_SINGLE_PASS_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,original_source_count=len(original_sources),
        result_sha256=result_sha,original_launch_sha256=LAUNCH_SHA,artifact_sha256=result['artifact_sha256'],
        original_owner=launch['owner'],original_owner_ended=True,complete_rows=1428,raw_model_forecasts=1425,
        observed_state_checks=result['report']['observed_state_checks'],timing_windows=timing,
        total_navigation_time_reduction_percent=100*(1-total['candidate_total_s']/total['baseline_total_s']),
        complete_report_and_timing_population_reconstructed=True,
        original_raw_and_model_inputs_reauthenticated=True,actual_model_or_observer_replay_reexecuted=False,
        sensing_scope=result['sensing_scope'],native_execution=False,real_time_qualified=False,
        navigation_qualified=False,goal_achieved=False))
    print('DEFERRED_MEMO_SINGLE_PASS_COMPLETION_VERIFIED',job.digest(OUTPUT),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result-sha256',required=True)
    verify_result(parser.parse_args().result_sha256)
