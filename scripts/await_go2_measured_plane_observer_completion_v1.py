"""Wait for the exact original owner, then verify its actual result once."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

from scripts import verify_go2_measured_plane_observer_history_v1 as checker
from scripts.startup_source_inventory_development import discover_sources

run = checker.run
SOURCE = 'scripts/await_go2_measured_plane_observer_completion_v1.py'
PROTOCOL = 'docs/go2_measured_plane_completion_prepared_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_observer_completion_wait_v1_attempt_001'


def main(preflight=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion waiter required')
    if checker.OUTPUT.exists() or checker.OUTPUT.is_symlink(): raise ValueError('completion already exists')
    run.verify_artifacts(run.OUTPUT, {'launch.json':checker.LAUNCH_SHA})
    launch = run.read_json(run.OUTPUT, 'launch.json')
    sources = discover_sources((SOURCE, PROTOCOL, checker.SOURCE, checker.TEST), launch['source_sha256'])
    run.verify(sources)
    live = run.owner_live(launch['owner'])
    if preflight:
        print('MEASURED_PLANE_COMPLETION_WAITER_PREFLIGHT', len(sources), 'original_live', live, flush=True)
        return
    run.create_output(OUTPUT)
    process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, original_launch_sha256=checker.LAUNCH_SHA,
        original_owner=launch['owner'], boot_id=launch['boot_id'], environment=run.ENV,
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        poll_seconds=30, automatic_retry=False, original_job_restart_permitted=False,
        execution_failure_is_preserved=True, native_execution=False))
    print('MEASURED_PLANE_COMPLETION_WAITER_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            while run.owner_live(launch['owner']):
                run.verify_artifacts(run.OUTPUT, {'launch.json':checker.LAUNCH_SHA})
                event = dict(utc=datetime.now(timezone.utc).isoformat(), status='EXACT_ORIGINAL_OWNER_LIVE')
                events.write(json.dumps(event)+'\n'); events.flush()
                time.sleep(30)
            run.verify(sources)
            run.verify_artifacts(run.OUTPUT, {'launch.json':checker.LAUNCH_SHA})
            if (run.OUTPUT/'failure.json').exists() or (run.OUTPUT/'failure.json').is_symlink():
                raise ValueError('original replay execution failed; preserve evidence and do not retry')
            sha = run.digest(run.OUTPUT/'result.json')
            run.write_json(OUTPUT/'observed_result_identity.json', dict(result_sha256=sha,
                original_owner_ended=True, original_launch_sha256=checker.LAUNCH_SHA))
            print('MEASURED_PLANE_COMPLETION_VERIFICATION_STARTED', sha, flush=True)
            checker.verify_result(sha)
        run.verify(sources)
        completion_sha = run.digest(checker.OUTPUT)
        ids = {n:run.digest(OUTPUT/n) for n in ('launch.json','events.jsonl','observed_result_identity.json')}
        run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_OBSERVER_COMPLETION_WAIT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, original_result_sha256=sha,
            completion_receipt=str(checker.OUTPUT.relative_to(run.ROOT)), completion_sha256=completion_sha,
            original_owner_ended=True, verification_executed_once=True, automatic_retry=False,
            native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_COMPLETION_WAITER_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_COMPLETION_WAITER_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
