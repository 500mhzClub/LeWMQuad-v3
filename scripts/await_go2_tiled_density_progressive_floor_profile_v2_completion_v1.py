"""Record the actual profile launch and verify its completion once its owner ends."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts import verify_go2_tiled_density_progressive_floor_profile_completion_v2 as check
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import verify_artifacts

SOURCE = 'scripts/await_go2_tiled_density_progressive_floor_profile_v2_completion_v1.py'
INVOCATION = 'docs/go2_tiled_density_progressive_floor_profile_v2_invocation_2026-09-11.json'
INVOCATION_SHA = '83b6bc951879b869fbaa00e3d27dbf931f94b0bbe7bbf4f696a4d4270f544ce7'
PREPARATION = 'docs/go2_tiled_density_progressive_floor_profile_v2_completion_preparation_2026-09-11.json'
PREPARATION_SHA = '9e11005a039dab0061a614edf9c0a72d7aa5a59c9f4f4e92f942aa860ad73ad3'
WATCH = ROOT/'docs/go2_tiled_density_progressive_floor_profile_v2_completion_watch_execution_2026-09-11.json'
RESULT = ROOT/'docs/go2_tiled_density_progressive_floor_profile_v2_completion_watch_result_2026-09-11.json'
FAILURE = ROOT/'docs/go2_tiled_density_progressive_floor_profile_v2_completion_watch_failure_2026-09-11.json'


def main():
    if any(p.exists() or p.is_symlink() for p in (WATCH, RESULT, FAILURE, ROOT/check.EXECUTION, check.OUTPUT)):
        raise ValueError('exclusive original profile completion watch required')
    verify({INVOCATION: INVOCATION_SHA, PREPARATION: PREPARATION_SHA})
    invocation = json.loads((ROOT/INVOCATION).read_text())
    preparation = json.loads((ROOT/PREPARATION).read_text())
    sources = discover_sources((SOURCE, INVOCATION, PREPARATION), preparation['source_sha256'])
    verify(sources)
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != invocation['boot_id']:
        raise ValueError('original profile invocation boot required')
    process = psutil.Process()
    write_json(WATCH, dict(status='TILED_PROFILE_V2_COMPLETION_WATCH_ACTIVE',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        boot_id=invocation['boot_id'], owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        profile_owner=invocation['owner'], invocation_sha256=INVOCATION_SHA, preparation_sha256=PREPARATION_SHA,
        polling_seconds=15, maximum_checker_invocations=1, native_execution=False, goal_achieved=False))
    print('TILED_PROFILE_V2_COMPLETION_WATCH_ACTIVE', digest(WATCH), flush=True)
    try:
        launch_path = check.run.OUTPUT/'launch.json'
        while not launch_path.exists():
            if not check.owner_live(invocation['owner']):
                raise ValueError('profile owner ended before actual launch; preserve admission failure')
            time.sleep(15)
        launch_sha = digest(launch_path)
        verify_artifacts(check.run.OUTPUT, {'launch.json': launch_sha})
        launch = json.loads(launch_path.read_text())
        if (launch['source_sha256'] != invocation['source_sha256']
                or launch['completion_sha256'] != invocation['completion_sha256']
                or launch['completed_result_sha256'] != invocation['completed_result_sha256']
                or launch['completed_launch_sha256'] != check.run.LAUNCH_SHA
                or launch['controller'] != 'TiledDensityProgressiveFloorController'
                or launch['frames'] != 1428 or launch['native_execution'] is not False
                or launch['model_training'] is not False):
            raise ValueError('exact admitted original V2 profile launch required')
        verify(launch['source_sha256']); verify_artifacts(check.run.OUTPUT, {'launch.json': launch_sha})
        execution = ROOT/check.EXECUTION
        write_json(execution, dict(status='TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_V2_EXECUTION_RECORDED',
            utc=datetime.now(timezone.utc).isoformat(), boot_id=invocation['boot_id'], owner=invocation['owner'],
            tool_session=invocation['tool_session'], source_sha256=launch['source_sha256'],
            source_count=len(launch['source_sha256']), launch_sha256=launch_sha,
            invocation_sha256=INVOCATION_SHA, completion_sha256=invocation['completion_sha256'],
            completed_result_sha256=invocation['completed_result_sha256'], native_execution=False,
            model_training=False, goal_achieved=False))
        execution_sha = digest(execution)
        print('TILED_PROFILE_V2_EXECUTION_RECORDED', execution_sha, launch_sha, flush=True)
        while check.owner_live(invocation['owner']):
            time.sleep(15)
        if (check.run.OUTPUT/'failure.json').exists() or (check.run.OUTPUT/'failure.json').is_symlink():
            raise ValueError('terminal profile failure preserved; no completion or replacement')
        verify(sources)
        result_sha = digest(check.run.OUTPUT/'result.json')
        print('TILED_PROFILE_V2_OWNER_ENDED_VERIFYING', result_sha, flush=True)
        subprocess.run([sys.executable, '-B', check.SOURCE, '--result-sha256', result_sha,
            '--execution-sha256', execution_sha], cwd=ROOT, check=True)
        verify(sources)
        write_json(RESULT, dict(status='TILED_PROFILE_V2_COMPLETION_WATCH_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
            result_sha256=result_sha, execution_sha256=execution_sha,
            completion_sha256=digest(check.OUTPUT), checker_invocations=1,
            profile_owner_ended=True, native_execution=False, goal_achieved=False))
        print('TILED_PROFILE_V2_COMPLETION_WATCH_COMPLETE', digest(RESULT), digest(check.OUTPUT), flush=True)
    except BaseException as error:
        write_json(FAILURE, dict(status='TERMINAL_TILED_PROFILE_V2_COMPLETION_WATCH_FAILURE',
            utc=datetime.now(timezone.utc).isoformat(), reason=repr(error), source_sha256=sources,
            automatic_retry=False, evidence_preserved=True, native_execution=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
