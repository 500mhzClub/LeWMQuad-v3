"""Invoke the frozen external-profile checker once, after its parent ends."""
import argparse
from datetime import datetime, timezone
import subprocess
import sys
import time

from scripts import run_go2_body_projection_external_profile_v1 as run
from scripts import verify_go2_body_projection_external_profile_completion_v1 as check
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE='scripts/await_go2_body_projection_external_profile_completion_v1.py'
TEST='lewm/tests/test_body_projection_external_profile_watch_development.py'
WATCH=ROOT/'docs/go2_body_projection_external_profile_completion_watch_execution_2026-09-11.json'
RESULT=ROOT/'docs/go2_body_projection_external_profile_completion_watch_result_2026-09-11.json'
FAILURE=ROOT/'docs/go2_body_projection_external_profile_completion_watch_failure_2026-09-11.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch-sha256',required=True)
    args=parser.parse_args()
    if any(path.exists() or path.is_symlink() for path in (WATCH,RESULT,FAILURE,check.OUTPUT)):
        raise ValueError('exclusive completion watch and unexecuted checker required')
    launch=run.authenticate_launch(args.launch_sha256)
    sources=discover_sources((SOURCE,TEST),launch['source_sha256'])
    verify(sources)
    write_json(WATCH,dict(status='BODY_PROJECTION_EXTERNAL_PROFILE_COMPLETION_WATCH_ACTIVE',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        owner=run.owner(), boot_id=run.BOOT, profile_owner=launch['owner'],
        launch_sha256=args.launch_sha256, maximum_checker_invocations=1,
        native_execution=False, goal_achieved=False))
    print('EXTERNAL_PROFILE_COMPLETION_WATCH_ACTIVE',digest(WATCH),flush=True)
    try:
        while run.owner_live(launch['owner']):time.sleep(15)
        for name in ('failure.json','child_failure.json'):
            if (run.OUTPUT/name).exists() or (run.OUTPUT/name).is_symlink():
                raise ValueError('original failed profile preserved; no verification or retry')
        run.authenticate_launch(args.launch_sha256);verify(sources)
        result_sha=digest(run.OUTPUT/'result.json')
        subprocess.run([sys.executable,'-B',run.CHECKER,'--result-sha256',result_sha,
            '--launch-sha256',args.launch_sha256],cwd=ROOT,check=True)
        verify(sources)
        write_json(RESULT,dict(status='BODY_PROJECTION_EXTERNAL_PROFILE_COMPLETION_WATCH_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
            launch_sha256=args.launch_sha256,result_sha256=result_sha,
            completion_sha256=digest(check.OUTPUT),checker_invocations=1,
            original_profile_parent_ended=True,native_execution=False,goal_achieved=False))
        print('EXTERNAL_PROFILE_COMPLETION_WATCH_COMPLETE',digest(RESULT),flush=True)
    except BaseException as error:
        write_json(FAILURE,dict(status='TERMINAL_EXTERNAL_PROFILE_COMPLETION_WATCH_FAILURE',
            reason=repr(error),source_sha256=sources,automatic_retry=False,evidence_preserved=True))
        raise


if __name__ == '__main__':main()
