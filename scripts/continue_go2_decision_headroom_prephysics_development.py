"""One prephysics continuation after the recorded two-label fixture erratum.

Not a general retry/resume mechanism. Requires zero source/branch attempts;
preserves the failed closeout and charges it to the original cumulative caps.
No analytical case is executed again, and no controller or model changes.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import psutil

from scripts.run_go2_decision_headroom_pilot_development import (
    CAPS, GIB, REPO, SOURCES, PilotBudget, save, vram,
)
from scripts.run_go2_decision_headroom_source_development import (
    ASSIGNMENTS, require_stage_a_closed, run,
)


CORRECTED_SHA = '6fc574d27e549e743ec7833ccaef77bb706e67cae2030a2f579579815ccefda5'


def main():
    entered = time.monotonic()
    caps = json.loads(CAPS.read_text())
    root = Path(caps['output_root'])
    require_stage_a_closed(root)
    old = json.loads((root/'resource_result.json').read_text())
    admission = json.loads((root/'pilot_execution_admission.json').read_text())
    if (old['source_attempts'] or old['snapshots'] or old['branch_attempts']
            or old['component_timing_attempts'] or old['resource_stop_latched']
            or old['error'] != "RuntimeError('reference sanity panel did not qualify; preserve result and stop')"):
        raise RuntimeError('only the recorded prephysics annotation failure permits this continuation')
    if any((root/f'source_{i:02d}').exists() for i in range(6)):
        raise RuntimeError('source evidence exists; no source restart permitted')
    archive = root/'prephysics_closeout_v1'
    if archive.exists():
        raise RuntimeError('one prephysics continuation only; preserve existing attempt')
    try:
        p = psutil.Process(admission['owner_pid'])
        if abs(p.create_time()-admission['owner_created']) < .01 and p.status()!=psutil.STATUS_ZOMBIE:
            raise RuntimeError('original pilot owner remains live')
    except psutil.NoSuchProcess:
        pass
    if admission['caps_sha256'] != hashlib.sha256(CAPS.read_bytes()).hexdigest():
        raise RuntimeError('original caps must remain unchanged')
    reference_path = root/'reference_sanity_v1/corrected_qualification.json'
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != CORRECTED_SHA:
        raise RuntimeError('exact documented annotation correction required')
    corrected = json.loads(reference_path.read_text())
    if (corrected['additional_case_evaluations'] != 0 or corrected['unique_analytical_cases'] != 24
            or corrected['cost_formula_changed'] or corrected['case_geometry_changed']):
        raise RuntimeError('no new cases or reference tuning permitted by this continuation')
    original = root/'reference_sanity_v1/result.json'
    if hashlib.sha256(original.read_bytes()).hexdigest() != corrected['original_result_sha256']:
        raise RuntimeError('original failed qualification changed')
    for path, digest in corrected['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('reference evaluator changed after qualification')
    for binding in admission['input_bindings'].values():
        with Path(binding['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != binding['sha256']:
                raise RuntimeError('frozen input changed')
    for path, digest in admission['source_hashes'].items():
        if path == 'scripts/read_go2_decision_headroom_pilot_development.py':
            continue  # Only the closeout reader now discloses the erratum.
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('source changed outside the documented audit closeout correction')
    for directory, reserve, peak in ((root, 12*GIB, 12*GIB), (REPO, 4*GIB, GIB)):
        if shutil.disk_usage(directory).free < reserve+peak:
            raise RuntimeError('original reserve plus peak-write admission unavailable')
    gpu = vram()
    if psutil.virtual_memory().available < 32*GIB or gpu['total']-gpu['used'] < 4*GIB:
        raise RuntimeError('original memory admission unavailable')
    os.sched_setaffinity(0, set(caps['compute_caps']['cpu_affinity']))
    os.environ['TMPDIR'] = str(root/'scratch/temp')
    tempfile.tempdir = str(root/'scratch/temp')
    archive.mkdir()
    for name in ('pilot_execution_admission.json', 'resource_result.json', 'failure.json',
                 'pilot_validity_report.json', 'budget_events.jsonl'):
        (root/name).rename(archive/name)
    admission.update(owner_pid=os.getpid(), owner_created=psutil.Process().create_time(),
        continuation_recorded_at=datetime.datetime.now().astimezone().isoformat(),
        prior_closeout='prephysics_closeout_v1',
        reference_sanity_result='reference_sanity_v1/corrected_qualification.json',
        reference_sanity_sha256=CORRECTED_SHA,
        reference_qualification_contains_post_execution_fixture_label_errata=True,
        source_hashes={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()
                       for p in (*SOURCES, str(Path(__file__).resolve().relative_to(REPO)))})
    save(root/'pilot_execution_admission.json', admission)
    budget = PilotBudget(root, caps, admission)
    # The previous interval is charged, not reset. Retained bytes include the
    # archived closeout and original failed panel, on the same filesystem.
    budget.started = entered-old['wall_s']
    own_cpu = psutil.Process().cpu_times()
    budget.cpu_start = -(old['cpu_s'])
    budget.cache_baseline = old['cache_baseline_bytes']
    budget.peak_rss = old['peak_sampled_aggregate_rss_bytes']
    budget.peak_vram = old['peak_sampled_total_gpu_used_bytes']
    budget.peak_retained = old['retained_bytes']
    budget.measurements = old['measurements']
    budget.event('prephysics_fixture_erratum_continuation', previous_wall_s=old['wall_s'],
        previous_cpu_s=old['cpu_s'], current_process_cpu_at_admission_s=own_cpu.user+own_cpu.system,
        analytical_cases_executed_again=0, original_caps_unchanged=True)
    error = None
    try:
        budget.check('continuation_admission', force=True)
        print('BLINDED_PILOT_SOURCE_COLLECTION_START; original cumulative caps apply', flush=True)
        for case in range(len(ASSIGNMENTS)):
            run(case)
            valid = sum(sum(r['restoration_passed'] for r in json.loads(
                (root/f'source_{i:02d}/branch_pilot_result.json').read_text())['states'])
                for i in range(case+1))
            if (case+1)*4-valid > caps['collection_caps']['sampled_states_total']*(1-admission['minimum_restored_state_fraction']):
                raise RuntimeError('fixed restoration-validity requirement cannot be reached within remaining assignments')
        save(root/'pilot_execution_complete.json', dict(status='EXECUTION_COMPLETE_REQUIRES_VALIDITY_CLOSEOUT',
            source_attempts=6, comparative_audit_scoring=False, phase2_authorized=False,
            reference_fixture_errata_disclosed=True, next='Checkpoint (a): frozen protocol and measured evidence; explicit approval required.'))
    except BaseException as exc:
        error = exc
        save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc()))
    finally:
        budget.finish(error)
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
