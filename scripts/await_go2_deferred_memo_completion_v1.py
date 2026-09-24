"""Verify the actual completed paired replay after its original owner ends."""
import argparse
from datetime import datetime,timezone
import json
import time
import psutil
from scripts import verify_go2_deferred_memo_single_pass_completion_v1 as checker
from scripts.startup_source_inventory_development import discover_sources

job=checker.job
SOURCE='scripts/await_go2_deferred_memo_completion_v1.py'
PROTOCOL='docs/go2_deferred_memo_completion_wait_v1_2026-09-11.md'
OUTPUT=job.OUTPUT.parent/'go2_deferred_memo_completion_wait_v1_attempt_001'


def main(preflight=False):
    if (not __debug__ or any(job.previous.os.environ.get(k) != v for k,v in job.previous.ENVIRONMENT.items())
            or job.harness.original.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    job.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink() or checker.OUTPUT.exists() or checker.OUTPUT.is_symlink():
        raise ValueError('exclusive waiter and uncreated completion receipt required')
    job.verify_artifacts(job.OUTPUT,{'launch.json':checker.LAUNCH_SHA})
    launch=json.loads((job.OUTPUT/'launch.json').read_text())
    sources=discover_sources((SOURCE,PROTOCOL,checker.SOURCE,checker.TEST),launch['source_sha256'])
    job.verify(sources);live=job.previous.owner_live(launch['owner'])
    if preflight:
        print('DEFERRED_MEMO_COMPLETION_PREFLIGHT',len(sources),'original_live',live,flush=True);return
    job.create_output(OUTPUT);p=psutil.Process()
    job.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,original_launch_sha256=checker.LAUNCH_SHA,
        original_owner=launch['owner'],boot_id=launch['boot_id'],environment=job.previous.ENVIRONMENT,
        owner=dict(pid=p.pid,created=p.create_time(),command=p.cmdline()),poll_seconds=30,
        automatic_retry=False,original_job_restart_permitted=False,native_execution=False))
    print('DEFERRED_MEMO_COMPLETION_WAITER_LAUNCHED',job.digest(OUTPUT/'launch.json'),flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            while job.previous.owner_live(launch['owner']):
                job.verify_artifacts(job.OUTPUT,{'launch.json':checker.LAUNCH_SHA})
                events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status='EXACT_ORIGINAL_REPLAY_OWNER_LIVE'))+'\n')
                events.flush();time.sleep(30)
            job.verify(sources)
            if (job.OUTPUT/'failure.json').exists() or (job.OUTPUT/'failure.json').is_symlink():
                raise ValueError('preserve original replay failure; no retry')
            sha=job.digest(job.OUTPUT/'result.json')
            job.write_json(OUTPUT/'observed_result_identity.json',dict(result_sha256=sha,
                original_owner_ended=True,original_launch_sha256=checker.LAUNCH_SHA))
            print('DEFERRED_MEMO_COMPLETION_VERIFICATION_STARTED',sha,flush=True)
            checker.verify_result(sha)
        job.verify(sources)
        ids={n:job.digest(OUTPUT/n) for n in ('launch.json','events.jsonl','observed_result_identity.json')}
        job.verify_artifacts(OUTPUT,ids)
        job.write_json(OUTPUT/'result.json',dict(status='DEFERRED_MEMO_COMPLETION_WAIT_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,original_result_sha256=sha,
            completion_receipt=str(checker.OUTPUT.relative_to(job.ROOT)),completion_sha256=job.digest(checker.OUTPUT),
            original_owner_ended=True,verification_executed_once=True,automatic_retry=False,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('DEFERRED_MEMO_COMPLETION_WAITER_COMPLETE',job.digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        job.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DEFERRED_MEMO_COMPLETION_WAITER_FAILURE',
            reason=repr(error),automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args().preflight)
