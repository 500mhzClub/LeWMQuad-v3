"""Own the single transition from original full fits to admitted correction."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from scripts import await_go2_all_phase_matched_fits_v1 as previous
from scripts.all_phase_translation_bias_model_admission_development import admit
from scripts.fit_go2_all_phase_training_translation_bias_v1 import OUTPUT as CORRECTION, SOURCE as CORRECTION_SOURCE
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE/'go2_all_phase_translation_bias_wait_v1_attempt_001'
SOURCE = 'scripts/await_go2_all_phase_translation_bias_v1.py'
TEST = 'lewm/tests/test_all_phase_translation_bias_wait_development.py'
PROTOCOL = 'docs/go2_all_phase_translation_bias_wait_v1_2026-09-10.md'
WAIT_LAUNCH_SHA = '4494e8370968c2643eec51f417d05be658bda747286f3a8b072bd67d9e6e3acd'
FIT_LAUNCH_SHA = 'de9fb37f6cd51c609c21fb6ba03b2470167dedad2ecb9166907874bd351d1542'
BENCH_SHA = '2e74b02b76038f92a8b74256c083cf8ecfbe28d5a5c1bc07c5cbafda214b054f'
BOOT = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
WAIT_OWNER = dict(pid=2635725, created=1789017704.31, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', previous.SOURCE])
FIT_OWNER = dict(pid=2636352, created=1789018036.61, command=[
    '/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',
    previous.FIT_SOURCE, '--phase', 'fits', '--benchmark-result-sha256', BENCH_SHA])


def owner_live(owner):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original owner boot identity changed')
    try:
        process = psutil.Process(owner['pid'])
        if process.create_time()!=owner['created'] or process.cmdline()!=owner['command']:
            raise ValueError('original owner process identity changed; no replacement permitted')
        return process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def admit_completed_fits(fit_sources, waiter_sources):
    for root, expected in ((previous.OUTPUT, WAIT_LAUNCH_SHA), (previous.FITS, FIT_LAUNCH_SHA)):
        verify_artifacts(root, {'launch.json':expected})
        if (root/'failure.json').exists(): raise ValueError('original full fits or waiter failed; correction unlaunched')
        if not (root/'result.json').is_file(): raise ValueError('original owner ended without complete result')
    wait_sha = digest(previous.OUTPUT/'result.json'); fit_sha = digest(previous.FITS/'result.json')
    waited = read_json(previous.OUTPUT, 'result.json'); fitted = read_json(previous.FITS, 'result.json')
    verify_artifacts(previous.OUTPUT, waited['artifact_sha256']|{'result.json':wait_sha})
    verify_artifacts(previous.FITS, fitted['artifact_sha256']|{'result.json':fit_sha})
    verify(fit_sources); verify(waiter_sources)
    if (waited['status']!='ALL_PHASE_MATCHED_FITS_WAIT_COMPLETE'
            or waited['source_sha256']!=waiter_sources
            or waited['artifact_sha256']['launch.json']!=WAIT_LAUNCH_SHA
            or waited['fit_pid']!=FIT_OWNER['pid'] or waited['fit_result_sha256']!=fit_sha
            or waited['benchmark_result_sha256']!=BENCH_SHA
            or waited['automatic_retry'] is not False or waited['native_execution'] is not False
            or fitted['status']!='ALL_PHASE_EIGHTEEN_FITS_COMPLETE'
            or fitted['optimizer_updates']!=21600 or fitted['source_sha256']!=fit_sources
            or fitted['artifact_sha256']['launch.json']!=FIT_LAUNCH_SHA):
        raise ValueError('exact original completed waiter and eighteen-fit result required')
    if previous.admit_benchmark(fit_sources)!=BENCH_SHA:
        raise ValueError('original complete benchmark identity changed')
    return dict(wait_result_sha256=wait_sha, fit_result_sha256=fit_sha, benchmark_result_sha256=BENCH_SHA)


def main():
    for root in (OUTPUT, CORRECTION):
        validate_root(root, must_exist=False)
        if root.exists() or root.is_symlink(): raise ValueError('exclusive waiter and unlaunched correction required')
    verify_artifacts(previous.OUTPUT, {'launch.json':WAIT_LAUNCH_SHA})
    verify_artifacts(previous.FITS, {'launch.json':FIT_LAUNCH_SHA})
    if not owner_live(WAIT_OWNER) or not owner_live(FIT_OWNER):
        raise ValueError('original waiter and full-fit parent must be live at registration')
    wait_sources = read_json(previous.OUTPUT, 'launch.json')['source_sha256']
    fit_sources = read_json(previous.FITS, 'launch.json')['source_sha256']
    inherited = dict(fit_sources)
    for name, sha in wait_sources.items():
        if name in inherited and inherited[name]!=sha: raise ValueError('incompatible original source: '+name)
        inherited[name] = sha
    sources = discover_sources((SOURCE, TEST, PROTOCOL,
        'docs/go2_all_phase_training_translation_bias_v1_2026-09-10.md',
        'lewm/tests/test_all_phase_translation_bias_development.py',
        'lewm/tests/test_all_phase_model_admission_development.py'), inherited)
    verify(sources); create_output(OUTPUT)
    launch = dict(source_sha256=sources, fit_source_sha256=fit_sources, waiter_source_sha256=wait_sources,
        original_wait_launch_sha256=WAIT_LAUNCH_SHA, original_fit_launch_sha256=FIT_LAUNCH_SHA,
        original_wait_owner=WAIT_OWNER, original_fit_owner=FIT_OWNER, boot_id=BOOT,
        correction_root=str(CORRECTION), waiter_pid=os.getpid(), maximum_wait_s=48*3600,
        automatic_retry=False, native_execution=False, source_changes_permitted=False,
        complete_model_admission_before_correction=True, complete_correction_admission_after=True)
    write_json(OUTPUT/'launch.json', launch); started = time.monotonic()
    print('ALL_PHASE_CORRECTION_WAITER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-started, **details))+'\n')
                events.flush(); print(status, details, flush=True)
            while owner_live(WAIT_OWNER):
                if time.monotonic()-started>launch['maximum_wait_s']:
                    raise ValueError('bounded wait expired; original work retained and correction unlaunched')
                event('ORIGINAL_FIT_WAITER_LIVE', pid=WAIT_OWNER['pid'], fit_parent_live=owner_live(FIT_OWNER))
                time.sleep(30)
            if owner_live(FIT_OWNER): raise ValueError('original waiter ended with fit parent still live')
            verify(sources); completed = admit_completed_fits(fit_sources, wait_sources)
            if CORRECTION.exists() or CORRECTION.is_symlink():
                raise ValueError('correction root appeared outside original waiter ownership')
            command = [sys.executable, CORRECTION_SOURCE, '--fit-result-sha256', completed['fit_result_sha256']]
            with (OUTPUT/'correction_stdout.log').open('xb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                event('TRAINING_ONLY_CORRECTION_LAUNCHED', pid=process.pid, command=command)
                while process.poll() is None:
                    event('TRAINING_ONLY_CORRECTION_OWNER_LIVE', pid=process.pid); time.sleep(30)
            event('TRAINING_ONLY_CORRECTION_OWNER_EXITED', pid=process.pid, returncode=process.returncode)
            if process.returncode!=0 or (CORRECTION/'failure.json').exists():
                raise ValueError('original correction failed; preserve evidence and do not retry')
            correction_sha = digest(CORRECTION/'result.json')
            corrected = read_json(CORRECTION, 'result.json')
            if (corrected['fit_result_sha256']!=completed['fit_result_sha256']
                    or any(sources.get(n)!=sha for n,sha in corrected['source_sha256'].items())):
                raise ValueError('correction must bind the original fits and frozen waiter sources')
            admission = admit(correction_sha)
            write_json(OUTPUT/'correction_admission.json', admission)
            if admit_completed_fits(fit_sources, wait_sources)!=completed:
                raise ValueError('original completed fit identities changed')
            verify(sources)
        files = ('launch.json', 'events.jsonl', 'correction_stdout.log', 'correction_admission.json')
        write_json(OUTPUT/'result.json', dict(status='ALL_PHASE_TRANSLATION_BIAS_WAIT_COMPLETE',
            source_sha256=sources, artifact_sha256={n:digest(OUTPUT/n) for n in files}, **completed,
            correction_result_sha256=correction_sha, correction_pid=process.pid,
            all_models=18, trained_heads=30, fitted_scalars=480, all_coefficients_reconstructed=True,
            automatic_retry=False, native_execution=False, navigation_qualified=False))
        print('ALL_PHASE_CORRECTION_WAITER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_ALL_PHASE_TRANSLATION_BIAS_WAIT_FAILURE',
            reason=repr(error), automatic_retry=False))
        raise


if __name__=='__main__': main()
