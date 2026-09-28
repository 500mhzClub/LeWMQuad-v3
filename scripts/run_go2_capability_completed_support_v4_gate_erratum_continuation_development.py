"""Resume the frozen V4 C0 gate under the 28 September executed-prefix erratum.

Same frozen harness, run owner, readers and fixed assignments. Completed gate
episodes are not rerun. The fifth attempt (02/0) reached a terminal mission
outcome, so it is re-evaluated from its preserved records. The frozen checker's
closeout error is requalified by the evaluator-side erratum; every other failure
follows the original cohort owner's rules.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from scripts import evaluate_go2_capability_oracle_prefix_erratum_development as erratum
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

FIDELITY = "RuntimeError('oracle executed-prefix fidelity check failed')"
FIFTH = 'v4_completed_support_gate_C0_dev02_ep0_attempt001'
ENVIRONMENT = dict(PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID='0', PYTHONPATH='.:lewm_genesis:lewm_worlds')


class GateStop(RuntimeError):
    pass


def closeout(base, root, assignment, result_code):
    """Classify one preserved attempt; returns its gate row or raises GateStop."""
    destination = base/'runs'/assignment
    if not (destination/'result.json').exists():
        raise GateStop('Owner failed before preserved closeout: '+assignment)
    result = json.loads((destination/'result.json').read_text())
    failure = result['error'] or ''
    controller_failure = False
    requalified = False
    if result_code:
        faults = destination/'pipeline_faults.json'
        closeout_defect = (destination/'closeout_failure.json').exists()
        controller_failure = faults.exists() and bool(json.loads(faults.read_text())) and not closeout_defect
        constructor_rejected = failure == "ValueError('finite initial-frame mission point within map bounds required')"
        if failure == FIDELITY and not closeout_defect and not (destination/'failure.json').exists():
            requalified = True
        elif (controller_failure and failure.startswith('RuntimeError(')) or constructor_rejected:
            controller_failure = True
        else:
            raise GateStop('Technical/resource/fidelity stop: '+assignment+' '+failure)
    prefix = None
    if (destination/'oracle_prefix_check.json').exists():
        path = destination/erratum.OUTPUT
        prefix = json.loads(path.read_text()) if path.exists() else erratum.write(destination)
        if not prefix['passed_under_erratum']:
            raise GateStop('C0 validity stop under erratum: '+assignment+' '+json.dumps(prefix['stops'][:5]))
    elif requalified:
        raise GateStop('Fidelity error without preserved checker rows: '+assignment)
    if not (destination/'episode_evaluation.json').exists():
        with (root/f'{assignment}_reader_continuation.log').open('x') as log:
            subprocess.run([sys.executable, ('scripts/read_go2_capability_completed_support_v4_failure_development.py' if controller_failure
                else 'scripts/read_go2_navigation_capability_completed_support_v4_development.py'),
                '--root', str(destination)], stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ | ENVIRONMENT)
    evaluation = json.loads((destination/'episode_evaluation.json').read_text())
    if evaluation.get('status') == 'STARTUP_FAILURE':
        raise GateStop('No scientific episode: '+assignment)
    row = dict(assignment=assignment, episode_id=evaluation['episode_id'],
        round_trip_success=evaluation['round_trip_success'], disallowed_contacts=evaluation['disallowed_contact_samples'],
        hard_violations=evaluation['safety']['hard']['confirmed_violation_samples'],
        hard_unresolved=evaluation['safety']['hard']['unresolved_sampled_samples'],
        failure_and_stall_taxonomy=evaluation['failure_and_stall_taxonomy'], wall_s=evaluation['wall_s'])
    if not (root/f'{assignment}_result.json').exists():
        owner.save(root/f'{assignment}_result.json', row)
    return row | dict(frozen_checker_error_requalified=requalified, prefix_erratum=None if prefix is None else {
        k: prefix[k] for k in ('passed_under_erratum', 'frozen_checker_passed', 'decisions', 'comparable_rows',
            'comparable_maximum_position_error_m', 'comparable_maximum_yaw_error_deg', 'no_matching_branch_decisions',
            'no_matching_branch_by_cause', 'dispatch_substitution_ticks', 'vetoed_selections', 'vetoed_movement_selections')})


def run(workers):
    assert Path.cwd().resolve() == owner.REPO and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    root = base/'cohorts/v4_completed_support_C0_gate'
    original = json.loads((root/'config.json').read_text())
    assert original['harness_sha256'] == owner.sha(owner.FREEZE) and not (root/'result.json').exists()
    assignments = [tuple(a) for a in original['assignments']]
    done = [a for a in assignments if (base/'runs'/a[2]).exists()]
    assert [a[2] for a in done] == [a[2] for a in assignments[:5]] and done[-1][2] == FIFTH
    remaining = assignments[5:]
    owner.save(root/'erratum_continuation_config.json', dict(
        completed_not_rerun=[a[2] for a in done[:4]], fifth_attempt_reevaluated_not_rerun=FIFTH,
        remaining=[a[2] for a in remaining], harness_sha256=owner.sha(owner.FREEZE),
        continuation_owner_sha256=owner.sha(__file__), run_owner_sha256=owner.sha(RUN_OWNER),
        erratum_sha256=owner.sha(erratum.ERRATUM), evaluator_sha256=owner.sha(erratum.__file__),
        concurrent_owners=workers, concurrency_basis='Handoff 5.7: output-preserving concurrency qualified by pilot 2/4-owner exact replays; each C0 episode independently prefix-qualified',
        automatic_retry=False))
    rows, stops, lock = {}, [], threading.Lock()
    with wall.job(base, 'C0 gate erratum continuation'):
        started = time.monotonic()
        for maze, episode, assignment in done:
            rows[assignment] = closeout(base, root, assignment, 1 if assignment == FIFTH else 0)
            print(json.dumps(dict(closeout=assignment, round_trip=rows[assignment]['round_trip_success'],
                prefix=rows[assignment]['prefix_erratum'] and rows[assignment]['prefix_erratum']['passed_under_erratum'])), flush=True)

        def attempt(item):
            maze, episode, assignment = item
            with lock:
                if stops:
                    return
                owner.Budget(base, protocol).check(force=True)
            try:
                if (base/'runs'/assignment).exists():
                    raise GateStop('Fresh assignment required: '+assignment)
                with (root/f'{assignment}.log').open('x') as log:
                    code = subprocess.run([sys.executable, RUN_OWNER, '--controller', 'C0', '--maze', str(maze),
                        '--episode', str(episode), '--assignment', assignment], stdout=log, stderr=subprocess.STDOUT,
                        env=os.environ | ENVIRONMENT).returncode
                row = closeout(base, root, assignment, code)
            except GateStop as exc:
                with lock:
                    stops.append(str(exc))
                return
            with lock:
                rows[assignment] = row
                if row['disallowed_contacts'] or row['hard_violations']:
                    stops.append('Safety disqualification: '+assignment)
            print(json.dumps(dict(assignment=assignment, round_trip=row['round_trip_success'],
                contacts=row['disallowed_contacts'], hard=row['hard_violations'],
                no_match=row['prefix_erratum'] and row['prefix_erratum']['no_matching_branch_decisions'],
                vetoed=row['prefix_erratum'] and row['prefix_erratum']['vetoed_selections'])), flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(attempt, remaining))
        ordered = [rows[a[2]] for a in assignments if a[2] in rows]
        successes = sum(r['round_trip_success'] for r in ordered)
        complete = len(ordered) == len(assignments)
        safe = all(r['disallowed_contacts'] == r['hard_violations'] == r['hard_unresolved'] == 0 for r in ordered)
        prefix_ok = all(r['prefix_erratum'] and r['prefix_erratum']['passed_under_erratum'] for r in ordered)
        passed = complete and not stops and successes >= 19 and safe and prefix_ok
        owner.save(root/'result.json', dict(controller='C0', harness='v4_completed_support', harness_sha256=owner.sha(owner.FREEZE),
            complete=complete, episodes=len(ordered), successes=successes, passed=passed, rows=ordered, stops=stops,
            prefix_erratum=dict(path='docs/go2_navigation_capability_oracle_prefix_erratum_2026-09-28.json', sha256=owner.sha(erratum.ERRATUM),
                no_matching_branch_decisions=sum(r['prefix_erratum']['no_matching_branch_decisions'] for r in ordered if r['prefix_erratum']),
                vetoed_selections=sum(r['prefix_erratum']['vetoed_selections'] for r in ordered if r['prefix_erratum'])),
            version_accounting=dict(cap_versions_including_v0=6, harness_versions_used_including_v0=6, charged_outcome_changes=5,
                cap_reached=True, ruling='Pre-registration governs (Andrew Knowles, 28 September 2026): V4 is the last version; a failed gate stops for his decision.'),
            continuation_wall_s=time.monotonic()-started, stop_on_safety_disqualification=any(s.startswith('Safety') for s in stops)))
        print(json.dumps(dict(gate_passed=passed, successes=successes, episodes=len(ordered), stops=stops,
            hours_used=round(wall.hours_used(base), 2))), flush=True)
    if stops:
        raise GateStop('; '.join(stops))


RUN_OWNER = 'scripts/run_go2_navigation_capability_completed_support_v4_development.py'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    run(p.parse_args().workers)
