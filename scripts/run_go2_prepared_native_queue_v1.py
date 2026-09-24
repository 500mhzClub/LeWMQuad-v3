"""One fixed sequential queue around unchanged, independently auditing runners."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.navigation_artifact_root_development import (
    BASE, artifact_path, create_output, validate_root, verify_artifacts)
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit, merge_sources

SOURCE = 'scripts/run_go2_prepared_native_queue_v1.py'
TEST = 'lewm/tests/test_prepared_native_queue_development.py'
PROTOCOL = 'docs/go2_prepared_native_queue_v1_2026-09-09.md'
OUTPUT = BASE/'go2_prepared_native_queue_v1_attempt_001'
PYTHON = ROOT/'.generated/venvs/genesis_rocm_0_4_6_v1/bin/python'
OWNER_PID = 2534319
OWNER_START_TICKS = 129848858
BOOT_ID = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
SUPERVISED_LAUNCH_SHA = '49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3'
LEARNED_SHA = 'a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720'
GIB = 1024**3
ENVIRONMENT = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
    PYTHONPATH='.:lewm_genesis:lewm_worlds', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def job(stem, result_status, worker_status, cases, option, value, runner_sha):
    return dict(stem=stem, runner='scripts/run_'+stem+'.py',
        output=stem+'_attempt_001', result_status=result_status,
        worker_status=worker_status, cases=cases, arguments=[option, value], runner_sha256=runner_sha)


SUPERVISED = job('go2_supervised_rollout_mazes_v1', 'SUPERVISED_ROLLOUT_MAZES_V1_COMPLETE',
    'SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED',
    [[f'full_supervised_rollout_novel_maze_{i:02d}', i] for i in (1, 2, 3)],
    '--learned-cohort-result-sha256', LEARNED_SHA,
    'e5baa6d61c3ebf9684b5dc549a1b7379019e8a8c5ab1572b1e8840d2be238699')
JOBS = (
    job('go2_direct_flow_maze03_pilot_v1', 'DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE',
        'DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED', [['full_jepa_direct_flow_maze_03', 3]],
        '--learned-cohort-result-sha256', LEARNED_SHA,
        '92d40f78f6d3d5ee1cba9b76ed4e90d8cea1554451fc1754baf10c98ce4cbb30'),
    job('go2_residual_anchored_continuation_maze_pilot_v1', 'RESIDUAL_ANCHORED_CONTINUATION_PILOT_V1_COMPLETE',
        'RESIDUAL_ANCHORED_CONTINUATION_COLLECTED_AND_RAW_AUDITED',
        [['full_jepa_residual_anchored_continuation_maze_02', 2]], '--prefix-result-sha256',
        '3cbd24abad8a6c70565648977ce8482df37b4c90b8a2e28c728799910cf402b5',
        'c9db3b3c1e5fc36afdff7dd1ed82e83756cbad35c450aca347942e9be48dad4a'),
    job('go2_recent_qualified_anchor_maze01_pilot_v1', 'RECENT_QUALIFIED_ANCHOR_MAZE01_PILOT_V1_COMPLETE',
        'RECENT_QUALIFIED_ANCHOR_MAZE01_COLLECTED_AND_RAW_AUDITED',
        [['full_jepa_recent_qualified_anchor_maze_01', 1]], '--prefix-result-sha256',
        '6e25c6c561473b60966a1a47388a01b48ab3e547da0c8ab13aa1e267b9fad302',
        '148b5877917943481d82980262122c6270ea132503691b52d189cbf12fb59cdc'),
)
PREFIXES = (
    ('go2_direct_flow_maze03_prefix_v1_attempt_001',
     '104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e'),
    ('go2_residual_anchored_continuation_prefix_v1_attempt_001', JOBS[1]['arguments'][1]),
    ('go2_recent_qualified_anchor_prefix_v1_attempt_001', JOBS[2]['arguments'][1]),
)

# The already reviewed native additions; inherited sources come from the exact
# completed prefix results and existing supervised launch, never a tree scan.
NATIVE_BINDINGS = {
    'scripts/direct_flow_maze03_episode_development.py': 'fa92dd286bab520e217d0e68ab4a3c613ff03d64f8a69672033d9b0a3ef8c9c0',
    'scripts/direct_flow_maze03_native_prefix_development.py': '44d0eebf519b42e341d86963b1953ff4b48a0e2351bb058f2e9623ed1f601354',
    'lewm/tests/test_direct_flow_maze03_native_development.py': 'e244d14301861656a70dcbb4a7f26d952f855cc2ed32a364392268126084c038',
    'lewm/tests/test_direct_flow_maze03_native_prefix_development.py': '6962f17d6e99c3a857169974b4cd737004f84ea58b236be5395eda3f2002ed79',
    'docs/go2_direct_flow_maze03_pilot_v1_2026-09-09.md': '36520dae7f6f1e64250571c3274f805043b87a22dea16cc658b20b04dfbf5874',
    'scripts/residual_anchored_continuation_maze_episode_development.py': '4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949',
    'scripts/residual_anchored_continuation_maze_audit_development.py': '6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3',
    'scripts/residual_anchored_continuation_native_prefix_development.py': '673b4f62e601c2270b17dcac5ce07888e75bbeb51f81531fe74d38c69bfcf30c',
    'lewm/tests/test_residual_anchored_continuation_native_development.py': '83ee1f42fefb8a396c1610efab3aff6b15b15f3ad6e731af576ae057b3647f88',
    'lewm/tests/test_residual_anchored_continuation_native_launcher_development.py': '9e189206e21bc82ffa4dc1275b67d37fd7b82752bd85b8546e95eba020a32391',
    'docs/go2_residual_anchored_continuation_maze_pilot_v1_2026-09-09.md': '3a9c9c52bb1c2c7cf58698e16598b9a77008ea4c25b70d2feaa7e38016bd989c',
    'scripts/recent_qualified_anchor_maze01_episode_development.py': 'd0e45f58a6586f4642c9e4308dc6ce2190e91d9c705ddbdc1b0a762b8dafae1d',
    'scripts/recent_qualified_anchor_maze01_audit_development.py': 'a67095a130b5095dff13262d11bfe565c4eee697edc16657dbe7c77e10d57bf7',
    'scripts/recent_qualified_anchor_native_prefix_development.py': '7ab78abc3fea55dd1f5c63192c60941f099f138a7f47f20686bc2e1d1f8aa0b0',
    'lewm/tests/test_recent_qualified_anchor_native_calculations_development.py': 'b2b30c9baef44d2e787e94cb7158006133a86bbda1f04322e2edb793b7465709',
    'lewm/tests/test_recent_qualified_anchor_native_prefix_development.py': 'd4c52845714078008ecbe760bc7ed92f84a3dcd93a640d08fb1d30a8b58a6c47',
    'lewm/tests/test_recent_qualified_anchor_native_launcher_development.py': '787de60742f01d3ec32a52ade95a68cae73c73555850705aff27ad5fb5607676',
    'docs/go2_recent_qualified_anchor_maze01_pilot_v1_2026-09-09.md': '74b1c7bed01f32965070a2c39b99d52d8590b623d46e01ab93dbb6702432ba0a',
    **{j['runner']: j['runner_sha256'] for j in (SUPERVISED, *JOBS)},
}
REPORTS = (
    'docs/go2_direct_flow_maze03_prefix_result_2026-09-09.md',
    'docs/go2_residual_anchored_continuation_prefix_result_2026-09-09.md',
    'docs/go2_recent_qualified_anchor_prefix_result_2026-09-09.md',
)


def read(output, name):
    return json.loads(artifact_path(output, name).read_text())


def owner_state(stat_text, boot_id):
    if boot_id != BOOT_ID:
        raise ValueError('original machine boot required; no queue resume')
    if stat_text is None:
        return False
    fields = stat_text.rsplit(')', 1)[1].split()
    if int(fields[19]) != OWNER_START_TICKS:
        raise ValueError('supervisor PID reused; original process identity required')
    return fields[0] not in ('Z', 'X')


def owner_live():
    try:
        stat = Path(f'/proc/{OWNER_PID}/stat').read_text()
    except FileNotFoundError:
        stat = None
    return owner_state(stat, Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def competing_command(command):
    # Conservatively serialize all same-user run_go2 runners and spawn workers.
    # This is a process observation, not a universal interprocess scene lock.
    if '--preflight-only' in command:
        return False
    return any((Path(arg).name.startswith('run_go2_') and arg.endswith('.py'))
               or 'multiprocessing.spawn' in arg for arg in command)


def competitors():
    found = []
    for process in psutil.process_iter():
        try:
            if (process.pid == os.getpid() or process.uids().real != os.getuid()
                    or 'python' not in process.name().lower()):
                continue
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            command = process.cmdline()
            if competing_command(command):
                found.append(dict(pid=process.pid, started=process.create_time(), command=command))
        except psutil.NoSuchProcess:
            continue
    return found


def wait_for_idle(event, *, original=False):
    deadline = time.monotonic()+48*3600
    while True:
        live = owner_live() if original else False
        other = competitors()
        if not live and not other:
            return
        if time.monotonic() >= deadline:
            raise ValueError('48-hour owner/competitor wait exhausted; no process interrupted')
        event('WAITING', original_supervisor_live=live, competing_processes=other)
        time.sleep(30)


def prepared_sources():
    prior = BASE/SUPERVISED['output']
    verify_artifacts(prior, {'launch.json': SUPERVISED_LAUNCH_SHA})
    sources = read(prior, 'launch.json')['source_sha256']
    for name, sha in PREFIXES:
        root = BASE/name
        verify_artifacts(root, {'result.json': sha})
        sources = merge_sources(sources, read(root, 'result.json')['source_sha256'])
    sources = merge_sources(sources, NATIVE_BINDINGS)
    verify(sources)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, *REPORTS), sources)
    verify(sources)
    return sources


def admit_outcomes(spec, result, launch, audits, prefixes, terminals):
    if (result['status'] != spec['result_status']
            or [[r['case'], r['layout_index']] for r in result['conditions']] != spec['cases']
            or launch['native_scene_workers'] != 1
            or launch['output_root'] != str(BASE/spec['output'])):
        raise ValueError('complete fixed ordered experiment required')
    planned = launch.get('planned_cases', [launch.get('planned_case')])
    if [c[:2] for c in planned] != spec['cases']:
        raise ValueError('fixed launch cases required')
    key = spec['arguments'][0][2:].replace('-', '_')
    if key == 'prefix_result_sha256':
        key = 'prospective_prefix_result_sha256'
    if launch[key] != spec['arguments'][1]:
        raise ValueError('original fixed CLI evidence required')
    for record, audit, prefix, terminal in zip(result['conditions'], audits, prefixes, terminals, strict=True):
        if (record['status'] != spec['worker_status'] or 'failure' in record
                or record != terminal or record['model_state_unchanged'] is not True
                or record['prefix_comparison'] != prefix
                or audit['layout_index'] != record['layout_index']):
            raise ValueError('complete identical original worker/audit/prefix evidence required')
        require_raw_audit(record, audit, learned=True)
        for flag in ('physical_and_public_prefix_exact', 'all_preintervention_requested_commands_exact',
                     'complete_candidate_decisions_match_prospective_prefix'):
            if prefix[flag] is not True:
                raise ValueError('physical causal prefix failed: '+flag)
    if result['measured_round_trip_successes'] != sum(int(r['verified_round_trip']) for r in result['conditions']):
        raise ValueError('all scientific outcomes retained in count')
    return dict(cases=spec['cases'], measured_round_trip_successes=result['measured_round_trip_successes'],
        scientific_success_required=False, all_raw_audits_pass=True, all_physical_prefixes_pass=True)


def authenticate_completed(spec, sources, expected_launch_sha=None):
    root = validate_root(BASE/spec['output'])
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original runner failed; retain evidence and stop queue')
    result_sha = digest(artifact_path(root, 'result.json'))
    result = read(root, 'result.json')
    bindings = result['artifact_sha256']
    if expected_launch_sha is not None and bindings['launch.json'] != expected_launch_sha:
        raise ValueError('original observed supervisor launch required')
    verify_artifacts(root, bindings)
    launch = read(root, 'launch.json')
    if result['source_sha256'] != launch['source_sha256']:
        raise ValueError('same original launch/result sources required')
    if any(sources.get(name) != sha for name, sha in result['source_sha256'].items()):
        raise ValueError('source outside frozen queue preparation')
    verify(sources)
    module = importlib.import_module(spec['runner'][:-3].replace('/', '.'))
    module.verify_inputs(launch)  # Original verifier, including its original scope if present.
    audits, prefixes, terminals = [], [], []
    for name, _ in spec['cases']:
        names = [name+suffix for suffix in ('_audit.json', '_prefix_comparison.json', '_worker_terminal.json')]
        if any(n not in bindings for n in names):
            raise ValueError('all audit/prefix/terminal files bound by original result')
        audit, prefix, terminal = [read(root, n) for n in names]
        if any(bindings.get(n) != sha for n, sha in terminal['artifact_sha256'].items()):
            raise ValueError('all original worker artifacts retained in final result')
        audits.append(audit); prefixes.append(prefix); terminals.append(terminal)
    report = admit_outcomes(spec, result, launch, audits, prefixes, terminals)
    if spec == SUPERVISED:
        learned = read(module.LEARNED, 'result.json')
        if (result['all_fixed_cases_executed'] is not True or result['original_case_order'] != [1, 2, 3]
                or result['paired_native_outcomes'] != module.paired_outcomes(learned['conditions'], result['conditions'])):
            raise ValueError('complete original paired cohort required')
    verify(sources)
    verify_artifacts(root, {**bindings, 'result.json': result_sha})
    return dict(**report, output_root=str(root), result_sha256=result_sha, launch_sha256=bindings['launch.json'],
        source_count=len(result['source_sha256']), output_count=len(bindings), original_verifier_reexecuted=True)


def execute_one(spec, sources, event, *, popen=subprocess.Popen):
    root = validate_root(BASE/spec['output'], must_exist=False)
    if root.exists() or root.is_symlink():
        raise ValueError('fresh original attempt required; no skip, retry or resume')
    wait_for_idle(event)
    verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 32*GIB or resources['artifact_free_bytes'] < 51*GIB:
        raise ValueError('native 32GiB RAM / 51GiB free-storage admission failed')
    if competitors():
        raise ValueError('competing owner appeared before native launch')
    environment = dict(os.environ, **ENVIRONMENT)
    # Remove optimization inherited through the environment; original assertions
    # and deterministic flags must remain effective in the fresh interpreter.
    environment.pop('PYTHONOPTIMIZE', None)
    command = [str(PYTHON), spec['runner'], *spec['arguments']]
    log_name = spec['stem']+'_stdout.log'
    with (OUTPUT/log_name).open('x') as log:
        child = popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        event('NATIVE_CHILD_STARTED', job=spec['stem'], pid=child.pid, command=command, hardware=resources)
        # Observation timeouts never stop/restart a native process.
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                event('NATIVE_CHILD_LIVE', job=spec['stem'], pid=child.pid)
    event('NATIVE_CHILD_EXITED', job=spec['stem'], pid=child.pid, returncode=code,
          stdout_sha256=digest(OUTPUT/log_name))
    if code != 0:
        raise ValueError('native runner exited unsuccessfully; no automatic retry')
    wait_for_idle(event)
    return authenticate_completed(spec, sources)


def run_sequence(sources, event, *, authenticate=authenticate_completed, execute=execute_one):
    wait_for_idle(event, original=True)
    completed = [authenticate(SUPERVISED, sources, SUPERVISED_LAUNCH_SHA)]
    write_json(OUTPUT/'supervised_completion.json', completed[0])
    event('SUPERVISED_AUTHENTICATED', **completed[0])
    for index, spec in enumerate(JOBS, 1):
        report = execute(spec, sources, event)
        completed.append(report)
        write_json(OUTPUT/f'completion_{index:02d}.json', report)
        event('NATIVE_CHILD_AUTHENTICATED', job=spec['stem'], **report)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__:
        raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive new scheduler; no resume')
    sources = prepared_sources()
    live = owner_live()
    resources = hardware()
    # A waiting scheduler/imported verifier allowance, alongside the live native
    # owner's unchanged 32GiB allowance. This is admission, not an OS limit.
    if resources['memory_available_bytes'] < 8*GIB or resources['artifact_free_bytes'] < 40*GIB+16*1024**2:
        raise ValueError('scheduler resource admission failed')
    for spec in JOBS:
        root = validate_root(BASE/spec['output'], must_exist=False)
        if root.exists() or root.is_symlink():
            raise ValueError('all three original queued outputs must be fresh')
    launch = dict(source_sha256=sources, existing_supervised_launch_sha256=SUPERVISED_LAUNCH_SHA,
        original_supervisor_pid=OWNER_PID, original_supervisor_start_ticks=OWNER_START_TICKS,
        boot_id=BOOT_ID, original_supervisor_live=live, jobs=JOBS, hardware=resources,
        subprocess_environment=ENVIRONMENT, python=str(PYTHON), native_scene_workers=1,
        maximum_scheduler_memory_allowance_bytes=8*GIB, planned_scheduler_metadata_allowance_bytes=16*1024**2,
        os_resource_limits_enforced=False, automatic_retry=False, frozen_runners_modified=False,
        universal_scene_lock_enforced=False, navigation_qualified=False, goal_achieved=False)
    if args.preflight_only:
        print('PREPARED_NATIVE_QUEUE_PREFLIGHT', json.dumps(dict(source_count=len(sources),
            original_supervisor_live=live, hardware=resources, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', launch)
    print('PREPARED_NATIVE_QUEUE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.monotonic()
    try:
        with (OUTPUT/'events.jsonl').open('x') as stream:
            def event(status, **details):
                stream.write(json.dumps(dict(status=status, elapsed_s=time.monotonic()-started, **details))+'\n')
                stream.flush()
                if status not in ('WAITING', 'NATIVE_CHILD_LIVE'):
                    print(status, json.dumps(details), flush=True)
            completed = run_sequence(sources, event)
        names = ['launch.json', 'events.jsonl', 'supervised_completion.json']
        names += [f'completion_{i:02d}.json' for i in (1, 2, 3)]
        names += [spec['stem']+'_stdout.log' for spec in JOBS]
        verify(sources)
        write_json(OUTPUT/'result.json', dict(status='PREPARED_NATIVE_QUEUE_V1_COMPLETE',
            completed=completed, source_sha256=sources,
            artifact_sha256={name:digest(OUTPUT/name) for name in names},
            automatic_retry=False, navigation_qualified=False, goal_achieved=False))
        print('PREPARED_NATIVE_QUEUE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='PREPARED_NATIVE_QUEUE_TERMINAL_FAILURE',
            reason=repr(error), automatic_retry=False, original_artifacts_retained=True))
        raise


if __name__ == '__main__':
    main()
