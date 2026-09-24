"""Own one automatic transition from the original benchmark to fixed full fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from scripts.all_phase_fit_execution_development import BENCH, OUTPUT as FITS, SOURCE as FIT_SOURCE, benchmark_decision
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE/'go2_all_phase_matched_fits_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_all_phase_matched_fits_v1.py'
TEST = 'lewm/tests/test_all_phase_matched_fits_wait_development.py'
BENCH_LAUNCH_SHA = '961f7fadf5f955f3fecc26c1a9494e71a31abe81575a1c330ebaf4762a53be8e'
OWNER_PID = 2633175
OWNER_CREATED = 1789016590.7
OWNER_COMMAND = ['.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', FIT_SOURCE, '--phase', 'benchmark']
BOOT = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'


def owner_live():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('benchmark owner boot identity changed')
    try:
        process = psutil.Process(OWNER_PID)
        if process.create_time()!=OWNER_CREATED or process.cmdline()!=OWNER_COMMAND:
            raise ValueError('original benchmark process identity changed; no replacement permitted')
        return process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def admit_benchmark(sources):
    verify_artifacts(BENCH, {'launch.json':BENCH_LAUNCH_SHA})
    if (BENCH/'failure.json').exists(): raise ValueError('original benchmark failed; full fits remain unlaunched')
    if not (BENCH/'result.json').is_file(): raise ValueError('original owner ended without a complete benchmark result')
    sha = digest(BENCH/'result.json'); result = read_json(BENCH,'result.json')
    verify_artifacts(BENCH, result['artifact_sha256'] | {'result.json':sha}); verify(sources)
    if (result['status']!='ALL_PHASE_FIT_BENCHMARK_COMPLETE'
            or result['source_sha256']!=sources or result['artifact_sha256']['launch.json']!=BENCH_LAUNCH_SHA
            or result['decision']!=benchmark_decision(result['serial'],result['parallel'])):
        raise ValueError('exact complete original benchmark and reproduced worker decision required')
    return sha


def main():
    validate_root(OUTPUT,must_exist=False); validate_root(FITS,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink() or FITS.exists() or FITS.is_symlink():
        raise ValueError('exclusive single waiter and previously unlaunched full-fit root required')
    verify_artifacts(BENCH, {'launch.json':BENCH_LAUNCH_SHA}); bench_launch = read_json(BENCH,'launch.json')
    if not owner_live(): raise ValueError('original benchmark owner must be live when registering this waiter')
    inherited = bench_launch['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE,TEST), inherited)
    verify(sources); create_output(OUTPUT)
    launch = dict(source_sha256=sources,benchmark_source_sha256=inherited,
        benchmark_launch_sha256=BENCH_LAUNCH_SHA,owner_pid=OWNER_PID,owner_created=OWNER_CREATED,
        owner_command=OWNER_COMMAND,boot_id=BOOT,fit_root=str(FITS),waiter_pid=os.getpid(),
        maximum_wait_s=48*3600,automatic_retry=False,native_execution=False,
        complete_benchmark_required=True,source_changes_permitted=False)
    write_json(OUTPUT/'launch.json',launch);started=time.monotonic()
    print('ALL_PHASE_FIT_WAITER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status,**details):
                row=dict(status=status,elapsed_s=time.monotonic()-started,**details)
                events.write(json.dumps(row)+'\n');events.flush();print(status,details,flush=True)
            while owner_live():
                if time.monotonic()-started>launch['maximum_wait_s']:
                    raise ValueError('bounded wait expired; original benchmark is retained and fits remain unlaunched')
                event('ORIGINAL_BENCHMARK_OWNER_LIVE',pid=OWNER_PID);time.sleep(30)
            verify(sources); bench_sha=admit_benchmark(inherited)
            if FITS.exists() or FITS.is_symlink(): raise ValueError('full-fit root appeared outside original waiter ownership')
            command=[sys.executable,FIT_SOURCE,'--phase','fits','--benchmark-result-sha256',bench_sha]
            with (OUTPUT/'fits_stdout.log').open('xb') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                event('FULL_FITS_LAUNCHED',pid=process.pid,benchmark_result_sha256=bench_sha,command=command)
                while process.poll() is None:
                    event('FULL_FITS_OWNER_LIVE',pid=process.pid);time.sleep(30)
            event('FULL_FITS_OWNER_EXITED',pid=process.pid,returncode=process.returncode)
            if process.returncode!=0 or (FITS/'failure.json').exists():
                raise ValueError('original full-fit run failed; preserve outputs and do not retry')
            fit_sha=digest(FITS/'result.json');fitted=read_json(FITS,'result.json')
            verify_artifacts(FITS,fitted['artifact_sha256']|{'result.json':fit_sha})
            if (fitted['status']!='ALL_PHASE_EIGHTEEN_FITS_COMPLETE' or fitted['optimizer_updates']!=21600
                    or fitted['source_sha256']!=inherited):
                raise ValueError('complete unchanged eighteen-fit result required')
            if admit_benchmark(inherited)!=bench_sha:raise ValueError('original benchmark result changed')
            verify(sources)
        files=('launch.json','events.jsonl','fits_stdout.log')
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_MATCHED_FITS_WAIT_COMPLETE',
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in files},
            benchmark_result_sha256=bench_sha,fit_result_sha256=fit_sha,fit_pid=process.pid,
            full_model_admission_still_required=True,automatic_retry=False,native_execution=False,navigation_qualified=False))
        print('ALL_PHASE_FIT_WAITER_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_MATCHED_FITS_WAIT_FAILURE',reason=repr(error),automatic_retry=False))
        raise


if __name__=='__main__':main()
