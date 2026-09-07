"""Two fixed tiny fit/group-OOM probes of the new durable outside relay.

Never consumes the real challenge unit/root, runs native code, changes a live
collector, restarts a used probe or grants native workload-fit authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from scripts import tracking_kernel_scope_development as kernel
from scripts import independent_tracking_memory_supervision_development as memory
from scripts import probe_go2_tracking_keeper_memory_development as child
from scripts import run_go2_independent_pulse_matched_study_v1 as learning
from scripts.startup_source_inventory_development import discover_sources, allowed_relative
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts

SOURCE = 'scripts/run_go2_tracking_keeper_memory_probe_v1.py'
TEST = 'lewm/tests/test_tracking_keeper_memory_probe_development.py'
PROTOCOL = 'docs/go2_tracking_keeper_memory_probe_v1_2026-09-07.md'
ENVIRONMENT = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='.:lewm_genesis:lewm_worlds',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LC_ALL='C', SYSTEMD_COLORS='0')


def identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def definition(mode):
    memory.require(mode in kernel.PROBE_UNITS, 'fixed tiny probe mode')
    old = learning.definition()
    memory.require(learning.identity(old) == '3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b',
        'original learning definition unchanged')
    sources = discover_sources((SOURCE, TEST, PROTOCOL), old['source_sha256'])
    return dict(schema='tracking_keeper_memory_probe_definition.v1', mode=mode,
        source_sha256=sources, original_learning_definition_sha256=learning.identity(old),
        output_root=str(memory.PROBE_OUTPUTS[mode]), unit=kernel.PROBE_UNITS[mode],
        profile=kernel.profile(kernel.PROBE_UNITS[mode]), environment=ENVIRONMENT,
        allocation_bytes=8*1024**2 if mode=='fit' else 128*1024**2,
        maximum_outside_bytes=memory.TOTAL_BYTES, minimum_free_bytes=memory.RESERVE_BYTES,
        minimum_available_memory_bytes=2*1024**3,
        native_execution=False, native_workload_fit_proved=False, retry_performed=False)


def verify_sources(d):
    for name, expected in d['source_sha256'].items():
        path = kernel.ROOT / allowed_relative(name)
        memory.require(path.resolve() == path and path.is_file() and memory.sha256(path) == expected,
            'fixed probe source identity changed: ' + name)


def command(d):
    mode = d['mode']
    memory.require(d['unit'] == kernel.PROBE_UNITS[mode]
        and d['profile'] == kernel.profile(d['unit']), 'fixed tiny unit/profile required')
    return kernel.service_prefix(d['unit']) + [
        f'--setenv={k}={v}' for k, v in sorted(ENVIRONMENT.items())] + [
        str(child.PYTHON), str(kernel.ROOT / child.SOURCE), '--mode', mode, '--role', 'parent',
        '--source-sha256', d['source_sha256'][child.SOURCE],
        '--kernel-sha256', d['source_sha256'][child.KERNEL_SOURCE]]


def fit_receipt(d):
    output = validate_root(memory.PROBE_OUTPUTS['fit'])
    p = artifact_path(output, 'terminal.json')
    memory.require(p.stat().st_size <= memory.FILE_BYTES, 'bounded prior tiny fit terminal')
    result = json.loads(p.read_text())
    memory.require(result['status'] == 'TINY_KEEPER_FIT_VERIFIED', 'completed same-source fit before overflow')
    verify_artifacts(output, result['output_sha256'])
    req = json.loads(artifact_path(output, 'request.json').read_text())
    memory.require(req['definition']['source_sha256'] == d['source_sha256'], 'same-source fit before overflow')
    return memory.sha256(p)


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('actual available memory required')


def preflight(mode, definition_sha256):
    memory.require(Path.cwd() == kernel.ROOT and str(sys.executable) == str(child.PYTHON)
        and all(os.environ.get(k) == v for k,v in ENVIRONMENT.items()), 'exact tiny probe runtime')
    outside = memory.outside_scope()
    memory.require(not any(unit in Path(outside['cgroup']).parts for unit in kernel.PROBE_UNITS.values()),
        'probe keeper must also be outside both tiny workload groups')
    d = definition(mode)
    memory.require(identity(d) == memory.hash_value(definition_sha256), 'exact preregistered tiny definition')
    verify_sources(d)
    root = validate_root(memory.PROBE_OUTPUTS[mode], must_exist=False)
    memory.require(not root.exists() and not root.is_symlink(), 'fresh tiny attempt; never retry')
    memory.require(available_memory() >= d['minimum_available_memory_bytes'], 'tiny workload memory reserve')
    memory.require(shutil.disk_usage(memory.BASE).free >= memory.RESERVE_BYTES + memory.TOTAL_BYTES,
        'tiny outside evidence storage reserve')
    kernel.require_fresh_unit(d['unit'])
    fit = fit_receipt(d) if mode == 'overflow' else None
    return d, outside, fit


def inspect_log(d, evidence, raw):
    """Validate exact probe events and manager classification, never exit-code OOM inference."""
    memory.require(evidence['child_handle_terminal'] and evidence['log_complete']
        and len(raw) == evidence['log_retained_bytes'] == evidence['log_total_bytes']
        and evidence['log_omitted_bytes'] == 0, 'complete bounded terminal diagnostic evidence')
    text = raw.decode('utf-8', errors='strict')
    events = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    memory.require(all(e['schema']=='tracking_keeper_memory_probe_event.v1' and e['mode']==d['mode']
        for e in events), 'exact tiny probe event schema and mode')
    admitted = [e for e in events if e['stage']=='SCOPE_ADMITTED']
    starts = [e for e in events if e['stage']=='LEAF_STARTED']
    requests = [e for e in events if e['stage']=='ALLOCATION_REQUEST']
    memory.require(len(admitted)==2 and {e['role'] for e in admitted}=={'parent','leaf'}
        and len(starts)==len(requests)==1, 'parent AND leaf admitted before intended allocation')
    roles = {e['role']:e['scope'] for e in admitted}
    expected = {'memory.max':str(d['profile']['memory_bytes']), 'memory.swap.max':'0',
        'memory.oom.group':'1', 'pids.max':str(d['profile']['tasks'])}
    for e in admitted:
        s=e['scope'];group=kernel.unified_group('0::'+s['cgroup'])
        memory.require(s['unit']==d['unit'] and group.name==d['unit'] and s['controls']==expected
            and group.parts[:4]==('/', 'user.slice', f'user-{os.getuid()}.slice', f'user@{os.getuid()}.service')
            and type(s['pid']) is int and s['pid']>0 and e['native_execution'] is False,
            'actual admitted fixed tiny kernel profile')
    memory.require(roles['parent']['pid'] != roles['leaf']['pid']
        and roles['parent']['cgroup'] == roles['leaf']['cgroup']
        and starts[0]['parent_pid']==roles['parent']['pid'] and starts[0]['leaf_pid']==roles['leaf']['pid']
        and requests[0]['pid']==roles['leaf']['pid'] and requests[0]['bytes']==d['allocation_bytes'],
        'same-group parent/leaf identity and exact intended allocation')
    returns = [e for e in events if e['stage']=='ALLOCATION_RETURNED']
    parents = [e for e in events if e['stage']=='PARENT_RETURNED']
    parent_admission=next(e for e in admitted if e['role']=='parent')
    leaf_admission=next(e for e in admitted if e['role']=='leaf')
    memory.require(events.index(parent_admission)<events.index(leaf_admission)<events.index(requests[0])
        and events.index(parent_admission)<events.index(starts[0]), 'actual causal probe event order')
    if d['mode']=='fit':
        memory.require(evidence['systemd_run_returncode']==0 and len(returns)==len(parents)==1
            and returns[0]['bytes']==d['allocation_bytes'] and returns[0]['pid']==roles['leaf']['pid']
            and parents[0]['returncode']==0 and len(events)==6
            and events.index(requests[0])<events.index(returns[0])<events.index(parents[0])
            and events.index(starts[0])<events.index(parents[0]), 'complete tiny parent/leaf normal return')
        status='TINY_KEEPER_FIT_VERIFIED'
    else:
        memory.require(evidence['systemd_run_returncode']!=0 and not returns and not parents
            and len(events)==4 and 'Finished with result: oom-kill' in text,
            'explicit manager OOM classification after admitted allocation; exit failure alone insufficient')
        status='TINY_KEEPER_GROUP_OOM_EVIDENCE_VERIFIED'
    return dict(status=status, events=events, parent_pid=roles['parent']['pid'], leaf_pid=roles['leaf']['pid'],
        manager_oom_classification_observed=d['mode']=='overflow', native_workload_fit_proved=False)


def execute(d, outside, fit):
    store = memory.EvidenceStore(probe_mode=d['mode'])
    request_sha = store.save('request.json', dict(definition=d, definition_sha256=identity(d),
        command=command(d), outside_supervisor=outside, prior_fit_terminal_sha256=fit))
    result = dict(status='TINY_KEEPER_INTERRUPTED_CHILD_STATE_UNVERIFIED',
        definition_sha256=identity(d), mode=d['mode'], output_sha256={'request.json':request_sha},
        child_handle_terminal=False, native_execution=False, navigation_qualified=False,
        goal_achieved=False, retry_performed=False)
    try:
        evidence = memory.relay_process(command(d), store)
        result.update(evidence)
        result['status']='TINY_KEEPER_PROBE_FAILED'
        result['output_sha256']['unit.log']=evidence['log_sha256']
        verify_artifacts(store.output, result['output_sha256']); verify_sources(d)
        log = artifact_path(store.output,'unit.log')
        memory.require(log.stat().st_size<=memory.FILE_BYTES,'bounded probe log')
        result.update(inspect_log(d,evidence,log.read_bytes()))
    except BaseException as error:
        result['error']=f'{type(error).__name__}: {str(error)[:4096]}'
        store.save('terminal.json',result)
        raise
    store.save('terminal.json',result)
    print(json.dumps(dict(status=result['status'],output=str(store.output),
        terminal_sha256=memory.sha256(artifact_path(store.output,'terminal.json')),
        parent_pid=result['parent_pid'],leaf_pid=result['leaf_pid'])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=kernel.PROBE_UNITS,required=True)
    p.add_argument('--definition-sha256',required=True)
    a=p.parse_args();d,outside,fit=preflight(a.mode,a.definition_sha256)
    execute(d,outside,fit)


if __name__=='__main__':
    main()
