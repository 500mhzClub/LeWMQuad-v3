"""Authenticate the ended extended-budget worker, not its still-running parent.

Retains the complete negative navigation result and every worker artifact. This
does not rerun raw/model inference, training ancestry, or physical comparison.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from scripts import run_go2_no_rgb_direct_extended_budget_maze02_pilot_v1 as native
from scripts import diagnose_go2_extended_budget_floor_boundary_v1 as diagnosis
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live

SOURCE = 'scripts/admit_go2_completed_extended_budget_worker_v1.py'
OWNER_RECORD = 'docs/go2_extended_budget_native_worker_launch_observation_2026-09-11.json'
OWNER_RECORD_SHA = 'ed8d0c8a6a98f8de2c9335e0847c8adbf9642adbd216d18a95b3954cd80326ed'
WORKER_SHA = '74520a1e3fd9a92af486454a7d788a90659138a8b60fc93212debe8d109c75f2'
OUTPUT = diagnosis.ROOT/'docs/go2_extended_budget_completed_worker_admission_2026-09-11.json'


def admit():
    diagnosis.verify({OWNER_RECORD:OWNER_RECORD_SHA})
    observation = json.loads((diagnosis.ROOT/OWNER_RECORD).read_text())
    owner = {k:observation['processes']['native_worker'][k] for k in ('pid','created','command')}
    if (observation['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or owner_live(owner)):
        raise ValueError('exact original native worker must be ended on its recorded boot')
    root, case = native.OUTPUT, native.CASE[0]
    terminal = case+'_worker_terminal.json'
    diagnosis.verify_artifacts(root, diagnosis.FIXED | {terminal:WORKER_SHA})
    launch = native.read_json(root, 'launch.json')
    if launch['source_sha256'] != observation['source_sha256'] or observation['launch_sha256'] != diagnosis.FIXED['launch.json']:
        raise ValueError('same original worker launch and complete source bindings required')
    sources = discover_sources((SOURCE,OWNER_RECORD), launch['source_sha256'])
    diagnosis.verify(sources)
    record = native.read_json(root, terminal)
    ids = record['artifact_sha256']
    expected = {case+'/'+n for n in native.pipeline.artifacts(2,record['collection'])}
    expected |= {case+s for s in ('_audit.json','_readout.json','_prefix_comparison.json')}
    if set(ids) != expected:
        raise ValueError('complete exact persisted worker artifact roster required')
    bindings = ids | {'launch.json':diagnosis.FIXED['launch.json'], terminal:WORKER_SHA,
        case+'_worker.log':record['worker_log_sha256']}
    diagnosis.verify_artifacts(root, bindings)
    audit = native.read_json(root, case+'_audit.json')
    native.require_worker(record, audit)
    collection = native.read_json(root/case, 'result.json')
    if (collection != record['collection'] or record['readout'] != native.read_json(root,case+'_readout.json')
            or record['prefix_comparison'] != native.read_json(root,case+'_prefix_comparison.json')):
        raise ValueError('same complete collection, readout and physical-prefix receipts required')
    with np.load(root/case/'physics_trace.npz',allow_pickle=False) as physics:
        contact = physics['physics_contact']
        if len(contact) != collection['physics_samples']:
            raise ValueError('complete actual physical contact population required')
        readout = native.case_readout(audit,collection,contact)
    if readout != record['readout']:
        raise ValueError('complete actual physical/contact/timing readout must reconstruct')
    diagnosis.verify(sources); diagnosis.verify_artifacts(root,bindings)
    if owner_live(owner): raise ValueError('original worker unexpectedly live')
    return dict(status='EXTENDED_BUDGET_COMPLETED_WORKER_ADMITTED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        worker_terminal_sha256=WORKER_SHA, native_launch_sha256=diagnosis.FIXED['launch.json'],
        owner_record_sha256=OWNER_RECORD_SHA, original_worker=owner, original_worker_ended=True,
        artifact_sha256=bindings, complete_worker_artifact_count=len(ids),
        original_collection=collection, readout=readout, prefix_comparison=record['prefix_comparison'],
        model_state_sha256=record['model_state_sha256'], model_state_unchanged=record['model_state_unchanged'],
        native_parent_or_waiter_completion_admitted=False, native_queue_advanced=False,
        raw_sensor_model_audit_reexecuted=False, physical_prefix_comparison_reexecuted=False,
        full_training_ancestry_reexecuted=False, original_physical_readout_reconstructed=True,
        new_native_execution=False, navigation_qualified=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False)


if __name__ == '__main__':
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completed-worker receipt required')
    report = admit()
    diagnosis.write_json(OUTPUT,report)
    print('EXTENDED_BUDGET_WORKER_ADMITTED',diagnosis.digest(OUTPUT),len(report['source_sha256']),
        report['complete_worker_artifact_count'],flush=True)
