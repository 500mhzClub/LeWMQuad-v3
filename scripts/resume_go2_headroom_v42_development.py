"""Authorized monitoring-only handover of the same V4.2 scientific attempt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import psutil

from lewm import decision_headroom_json_v42_development as output_json
from lewm.decision_headroom_resume_monitor_development import ResumeBudget
from scripts.handover_go2_headroom_v42_development import digest, reconstruct
from scripts.run_go2_decision_headroom_pilot_development import save

CONFIG = Path('docs/go2_decision_headroom_protocol_v42_2026-09-23.json')
AMENDMENT = Path('docs/go2_decision_headroom_v42_monitor_handover_2026-09-24.json')


def verify(amendment_sha256):
    if digest(AMENDMENT) != amendment_sha256:
        raise ValueError('monitoring amendment identity differs')
    amendment = json.loads(AMENDMENT.read_text())
    config = json.loads(CONFIG.read_text())
    if digest(CONFIG) != amendment['protocol_sha256']:
        raise ValueError('scientific protocol changed')
    approval = Path(amendment['original_approval'])
    if digest(approval) != amendment['original_approval_sha256']:
        raise ValueError('original approval changed')
    original = json.loads(approval.read_text())
    if original['protocol_sha256'] != digest(CONFIG) or not original['phase2_explicitly_approved']:
        raise ValueError('original V4.2 not approved')
    for path, binding in {**config['frozen_bindings'], **amendment['implementation_bindings']}.items():
        if digest(path) != binding['sha256']:
            raise ValueError('bound source changed: ' + path)
    for binding in config['execution_input_bindings'].values():
        path = Path(binding['path'])
        if path.stat().st_size != binding['bytes'] or digest(path) != binding['sha256']:
            raise ValueError('model/input changed: ' + str(path))
    return config, amendment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amendment-sha256', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--wait-for-boundary', action='store_true')
    args = parser.parse_args()
    if args.wait_for_boundary:
        # No simulation/model initialization while the original owner is active.
        root = Path(json.loads(CONFIG.read_text())['execution_caps']['output_root'])
        initial_owner = json.loads((root / 'pilot_execution_admission.json').read_text())
        print('WAITING_FOR_VERIFIED_HANDOVER_BOUNDARY', flush=True)
        while not (root / 'handover_boundary.json').exists():
            predecessor = psutil.Process(initial_owner['owner_pid'])
            if abs(predecessor.create_time()-initial_owner['owner_created']) > .01 or predecessor.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError('predecessor exited before boundary; no automatic recovery')
            if (root / 'failure.json').exists() or (root / 'collection_result.json').exists():
                raise RuntimeError('predecessor finished/stopped; no continuation admitted')
            time.sleep(5)
    config, amendment = verify(args.amendment_sha256)
    if args.verify_only:
        print('UNCHANGED_V42_SCIENCE_AND_MONITOR_AMENDMENT_VERIFIED', flush=True)
        return
    root = Path(config['execution_caps']['output_root'])
    boundary = json.loads((root / 'handover_boundary.json').read_text())
    order = [case for layout in config['execution_order'] for case in range(3*layout, 3*layout+3)]
    state = reconstruct(root, order)
    if state is not None:
        state = json.loads(json.dumps(state))
    if state is None or any(state[k] != boundary[k] for k in (
        'closed_cases', 'remaining_cases', 'source_ns', 'branches', 'snapshots',
        'journal_sha256', 'journal_bytes', 'closeout_sha256',
    )):
        raise ValueError('boundary evidence changed or active source exists')
    if (root / 'resume_admission.json').exists():
        raise ValueError('continuation already admitted; no duplicate owner')
    owner = psutil.Process(boundary['old_pid'])
    if abs(owner.create_time()-boundary['old_created']) > .01 or owner.status() != psutil.STATUS_STOPPED:
        raise ValueError('predecessor is not the recorded paused owner')
    if digest(root / 'pilot_execution_admission.json') != boundary['original_admission_sha256']:
        raise ValueError('original execution admission changed')
    if time.time()-boundary['original_wall_origin_epoch'] >= config['execution_caps']['compute_caps']['execution_wall_seconds']:
        raise RuntimeError('original wall cap exhausted; no continuation')
    children = owner.children(recursive=True)
    if any('multiprocessing.resource_tracker' not in ' '.join(p.cmdline()) for p in children):
        raise RuntimeError('predecessor still owns a worker')
    output_json.install(root)
    # Record the intent before retiring the paused, fully closed predecessor.
    save(root / 'handover_termination_intent.json', dict(
        old_pid=owner.pid, old_created=owner.create_time(), boundary_sha256=digest(root / 'handover_boundary.json'),
        closed_cases=boundary['closed_cases'], remaining_cases=boundary['remaining_cases'],
        reason='User-authorized monitoring-only handover after full assignment closeout',
        no_active_trial=True, no_retry=True, amendment_sha256=args.amendment_sha256))
    owner.kill()
    psutil.wait_procs([owner], timeout=10)
    if owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE:
        raise RuntimeError('predecessor did not exit; do not start another GPU owner')
    for process in children:
        try:
            if process.is_running():
                process.terminate()
        except psutil.NoSuchProcess:
            pass
    # Existing scientific source adapters read this original admission path.
    # Preserve its original bytes before the sole authorized owner-field update.
    old = json.loads((root / 'pilot_execution_admission.json').read_text())
    save(root / 'pilot_execution_admission_pre_handover.json', old)
    if digest(root / 'pilot_execution_admission_pre_handover.json') != boundary['original_admission_sha256']:
        raise ValueError('preserved admission differs')
    admission = dict(old, owner_pid=os.getpid(), owner_created=psutil.Process().create_time(),
                     monitoring_amendment_sha256=args.amendment_sha256)
    temp = root / 'pilot_execution_admission_successor.json'
    save(temp, admission)
    os.replace(temp, root / 'pilot_execution_admission.json')
    save(root / 'resume_admission.json', dict(
        **admission, boundary_sha256=digest(root / 'handover_boundary.json'),
        original_wall_origin_epoch=boundary['original_wall_origin_epoch'],
        cumulative_cpu_carry_s=boundary['prior_cpu_s'], remaining_cases=boundary['remaining_cases'],
        scientific_protocol_unchanged=True, original_caps_unchanged=True,
        original_journal_prefix_bytes=boundary['journal_bytes'],
        original_journal_prefix_sha256=boundary['journal_sha256'],
        gpu_owner_time_charged_conservatively_as_entire_audit_wall=True))
    os.sched_setaffinity(0, config['execution_caps']['compute_caps']['cpu_affinity'])
    budget = ResumeBudget(root, config['execution_caps'], admission, boundary)
    print('HANDOVER_COMPLETE_RESUMING_UNTOUCHED_ASSIGNMENTS', boundary['remaining_cases'], flush=True)
    error = None
    outcomes = list(boundary['outcomes'])
    try:
        budget.event('monitoring_handover', predecessor=boundary['old_pid'], successor=os.getpid(),
                     amendment_sha256=args.amendment_sha256, completed_cases=boundary['closed_cases'])
        budget.check('continuation_admission', force=True)
        from scripts.run_go2_headroom_v42_source_development import run, require_stage_a_closed
        require_stage_a_closed(root)
        for case in boundary['remaining_cases']:
            try:
                run(case)
                outcomes.append(dict(case=case, status='complete'))
            except Exception as exc:
                outcomes.append(dict(case=case, status='unresolved', reason=repr(exc), traceback=traceback.format_exc()))
                if budget.stopped:
                    raise
                budget.active_case = None
            save(root / f'cell_{case:02d}_closeout.json', outcomes[-1])
            budget.close_assignment(case)
        save(root / 'collection_result.json', dict(status='FIXED_ASSIGNMENTS_CLOSED', cells=outcomes,
             retries=False, source_navigation_outcomes_descriptive_only=True))
        from scripts.read_go2_headroom_v42_development import report
        report(root, budget=budget)
        from scripts.check_go2_headroom_v42_owner_development import assemble_outputs
        assemble_outputs(root, config, implementation_only=False)
        save(root / 'json_output_receipt.json', output_json.receipt())
        budget.check('analysis_complete', force=True)
    except BaseException as exc:
        error = exc
        save(root / 'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(),
                                        cells=outcomes, no_extension=True))
    finally:
        budget.finish(error)
        path = root / 'resource_result.json'
        result = json.loads(path.read_text())
        result.update(phase2_authorized=True, implementation_only=False,
                      gpu_owner_wall_s=time.monotonic()-budget.started,
                      approval_sha256=amendment['original_approval_sha256'],
                      monitoring_amendment_sha256=args.amendment_sha256,
                      cumulative_predecessor_resources_included=True,
                      predecessor_writer_checks_unchanged=True,
                      json_receipt_scope='Successor process. Predecessor used unchanged immediate readback checks; its in-memory receipt is not transferred.')
        path.write_text(json.dumps(result, indent=2)+'\n')
    if error:
        raise error


if __name__ == '__main__':
    main()
