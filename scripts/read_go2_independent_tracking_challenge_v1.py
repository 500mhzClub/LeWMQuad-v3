"""Read-only complete tracking accounting; never launch, retry or qualify a robot."""
import argparse
import json
import os
from pathlib import Path

from scripts import run_go2_independent_tracking_challenge_v1 as challenge
from scripts import independent_tracking_memory_supervision_development as memory
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_predecessor_comparison_development as predecessor
from scripts import independent_tracking_scored_population_verification_development as scores
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts


def _unqualified(row, keys):
    for key in keys:
        base.require(row[key] is False, 'unsupported qualification: ' + key)


def _no_failure(output):
    path = output / 'failure.json'
    base.require(not path.exists() and not path.is_symlink(), 'retained failure blocks complete-result admission')


def _supervision(terminal_sha256, challenge_sha256, definition_sha256, launch):
    output = validate_root(memory.OUTPUT)
    verify_artifacts(output, {'terminal.json': terminal_sha256})
    terminal = base.read(output, 'terminal.json')
    base.require(terminal['status'] == 'SCOPED_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE'
        and terminal['definition_sha256'] == definition_sha256
        and terminal['challenge_result_sha256'] == challenge_sha256
        and terminal['unit'] == memory.UNIT and terminal['child_handle_terminal'] is True
        and terminal['workload_completion_verified'] is True
        and type(terminal['systemd_run_returncode']) is int and terminal['systemd_run_returncode'] == 0
        and terminal['log_complete'] is True and 'error' not in terminal,
        'exact completed outside supervisor required; exit zero alone is insufficient')
    _unqualified(terminal, ('retry_performed', 'navigation_qualified', 'goal_achieved',
        'workload_termination_inferred_from_returncode', 'failure_cause_inferred_from_exit_code'))
    bindings = {'terminal.json': terminal_sha256, 'request.json': terminal['request_sha256'],
        'unit.log': terminal['log_sha256']}
    verify_artifacts(output, bindings)
    sizes = {n: artifact_path(output, n).stat().st_size for n in memory.FILES}
    base.require(all(n <= memory.FILE_BYTES for n in sizes.values())
        and sum(sizes.values()) <= memory.TOTAL_BYTES, 'bounded outside evidence required')
    base.require(all(type(terminal[k]) is int for k in ('log_total_bytes', 'log_retained_bytes', 'log_omitted_bytes'))
        and terminal['log_total_bytes'] == terminal['log_retained_bytes'] == sizes['unit.log']
        and terminal['log_omitted_bytes'] == 0, 'complete exact diagnostic byte accounting required')
    request = base.read(output, 'request.json')
    supervisor = request['supervisor']
    base.require(type(supervisor) is dict and set(supervisor) == {'pid', 'cgroup'}
        and type(supervisor['pid']) is int and supervisor['pid'] > 0, 'recorded outside identity required')
    outside = memory.unified_group('0::' + supervisor['cgroup'])
    base.require(memory.UNIT not in outside.parts, 'recorded keeper must be outside workload group')
    expected = dict(schema='independent_tracking_outside_supervision_request.v1',
        definition_sha256=definition_sha256,
        study_result_sha256=launch['completed_learning']['study_result_sha256'],
        scope_contract=memory.contract(), output_root=str(challenge.OUTPUT), supervisor=supervisor)
    base.require(request == expected, 'same bound outside request and learning prerequisite required')
    supervision = launch['memory_supervision']
    base.require(set(supervision) == {'request_sha256', 'scope', 'supervisor'}
        and supervision['request_sha256'] == terminal['request_sha256']
        and supervision['supervisor'] == supervisor, 'launch and keeper authority differ')
    scope = supervision['scope']; group = memory.unified_group('0::' + scope['cgroup'])
    base.require(set(scope) == {'unit', 'pid', 'cgroup', 'controls'} and scope['unit'] == memory.UNIT
        and type(scope['pid']) is int and scope['pid'] > 0 and scope['pid'] != supervisor['pid']
        and group.name == memory.UNIT and group.parts[:4] == (
            '/', 'user.slice', f'user-{os.getuid()}.slice', f'user@{os.getuid()}.service')
        and scope['controls'] == {'memory.max': str(memory.MEMORY_BYTES), 'memory.swap.max': '0',
            'memory.oom.group': '1', 'pids.max': str(memory.TASKS)}, 'exact saved workload-scope record required')
    # Do not query a historical PID or recreate a collected systemd unit. These
    # are authenticated records of source-side admission, not new kernel proof.
    return bindings, sizes


def authenticate(challenge_sha256, supervisor_sha256, definition_sha256):
    for value in (challenge_sha256, supervisor_sha256, definition_sha256): memory.hash_value(value)
    output = validate_root(challenge.OUTPUT); _no_failure(output)
    verify_artifacts(output, {'challenge_result.json': challenge_sha256})
    terminal = base.read(output, 'challenge_result.json')
    base.require(terminal['status'] == 'NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE'
        and terminal['definition_sha256'] == definition_sha256
        and terminal['resource_contract'] == challenge.resource_contract(), 'exact challenge terminal required')
    _unqualified(terminal, ('independent_result_verification_complete', 'full_challenge_pass',
        'navigation_qualified', 'real_time_qualified', 'goal_achieved'))
    bindings = terminal['output_sha256']
    base.require(type(bindings) is dict and set(bindings) == scores.expected_names() | {
        'result.json', 'predecessor_comparison.json'}, 'exact complete challenge artifact roster required')
    base.require(bindings['result.json'] == terminal['result_sha256']
        and bindings['predecessor_comparison.json'] == terminal['comparison_sha256'], 'terminal result bindings differ')
    verify_artifacts(output, bindings)
    launch = base.read(output, 'launch.json'); definition = challenge.definition()
    base.require(challenge.learning.identity(definition) == definition_sha256
        and launch['definition_sha256'] == definition_sha256 and launch['definition'] == definition,
        'current reviewed source/config must equal the frozen launched definition')
    completed = challenge.completed_learning(launch['completed_learning']['study_result_sha256'])
    base.require(launch['completed_learning'] == completed, 'complete authenticated parallel-learning prerequisite required')
    outside_bindings, outside_sizes = _supervision(supervisor_sha256, challenge_sha256, definition_sha256, launch)
    result = base.read(output, 'result.json')
    base.require(result['output_sha256'] == {n: bindings[n] for n in scores.expected_names()},
        'scoring and final terminal must bind the same complete artifacts')
    for index, trial in enumerate(challenge.TRIALS):
        request_name = trial + '_worker_request.json'; request_sha = bindings[request_name]
        expected = dict(trial=trial, launch_sha256=bindings['launch.json'],
            specification=challenge.specification(trial), protocol_sha256=definition['source_sha256'][challenge.PROTOCOL],
            previous_receipt_sha256=None if index == 0 else bindings[challenge.TRIALS[index - 1] + '_receipt.json'])
        base.require(base.read(output, request_name) == expected, 'ordered exact worker request required')
        exit_row = base.read(output, trial + '_worker_exit.json')
        base.require(exit_row == dict(trial=trial, returncode=0) and type(exit_row['returncode']) is int,
            'terminal successful worker handle required')
        episode = base.read(output, trial + '_receipt.json')
        expected_receipt = dict(trial=trial, launch_sha256=bindings['launch.json'], request_sha256=request_sha,
            result=episode['result'], receipt=episode['receipt'])
        base.require(base.read(output, trial + '_worker_receipt.json') == expected_receipt,
            'worker and admitted episode receipt differ')
    return terminal, launch, outside_bindings, outside_sizes


def _compare_predecessors(output, result_sha256, episodes):
    """Called only after full scored-population sensor admission and verification."""
    old = predecessor.load_predecessors(); current = {}; comparisons = {}
    for trial in challenge.TRIALS:
        entry = episodes[trial]['result']
        raw = predecessor._read_pose(output, trial + '/physics_trace.npz')
        reader = predecessor.IntentReturnRGBDReplay(output / trial) if entry['rgbd_frames'] else None
        static = base.read(output, trial + '/static_objects.json') if entry['setup_checked'] else []
        witness = predecessor.extract_witness(raw, static, reader, base.read(output, trial + '/camera_audit.json'))
        current[trial] = witness
        comparisons[trial] = {key + '/' + name: predecessor.compare_witnesses(previous, witness)
            for key, cohort in old['cohorts'].items() for name, previous in cohort['witnesses'].items()}
    return dict(status='ACTUAL_PREDECESSOR_NONIDENTITY_COMPARISON_COMPLETE', result_sha256=result_sha256,
        predecessors=old, current_witnesses=current, comparisons=comparisons,
        all_six_predecessor_nonidentity_checks_pass=all(
            pair['nonidentity_checks_pass'] for row in comparisons.values() for pair in row.values()),
        independent_observations_verified=False, full_challenge_pass=False, navigation_qualified=False, goal_achieved=False)


def read_result(challenge_sha256, supervisor_sha256, definition_sha256):
    terminal, launch, outside_bindings, outside_sizes = authenticate(
        challenge_sha256, supervisor_sha256, definition_sha256)
    output = challenge.OUTPUT; result_sha = terminal['result_sha256']
    scored = scores.verify_scored_population(output, result_sha, launch['definition']['source_sha256'][challenge.PROTOCOL])
    result = base.read(output, 'result.json')
    # Scored-population verification just reauthenticated all episode bytes.
    # Read those exact bound receipts, rather than hashing the full raw cohort
    # again immediately. A final reauthentication follows the comparison below.
    episodes = {t: base.read(output, t + '_receipt.json') for t in challenge.TRIALS}
    comparison = _compare_predecessors(output, result_sha, episodes)
    base.require(base.encode(comparison) == base.encode(base.read(output, 'predecessor_comparison.json')),
        'saved predecessor comparisons differ from reexecuted witnesses')
    # Reauthenticate predecessors after comparing; no second native decode.
    for cohort in comparison['predecessors']['cohorts'].values():
        verify_artifacts(Path(cohort['root']), cohort['receipt_sha256'] | cohort['selected_artifact_sha256'])
    bindings = terminal['output_sha256'] | {'challenge_result.json': challenge_sha256}
    sizes = {n: artifact_path(output, n).stat().st_size for n in bindings}
    for name, size in sizes.items():
        limit = base.MAX_ROW * challenge.MAX_FRAMES if name.endswith('.jsonl') else base.MAX_METADATA
        base.require(size <= limit, 'complete artifact exceeds its declared bound: ' + name)
    total = sum(sizes.values()) + sum(outside_sizes.values()) + sum(
        entry['receipt']['artifact_bytes'] for entry in episodes.values())
    base.require(total <= challenge.resource_contract()['combined_challenge_and_supervision_bound_bytes'],
        'complete recorded bytes exceed reviewed combined bound')
    _no_failure(output); challenge.verify_ordered_launch(launch['definition'])
    base.verify_collection(output, result['collection_sha256'])
    verify_artifacts(output, bindings); verify_artifacts(memory.OUTPUT, outside_bindings)
    return dict(status='COMPLETE_TRACKING_RESULT_ACCOUNTING_VERIFIED',
        challenge_result_sha256=challenge_sha256, supervisor_terminal_sha256=supervisor_sha256,
        definition_sha256=definition_sha256, recorded_artifact_bytes=total,
        source_launch_worker_and_supervisor_records_verified=True,
        historical_kernel_controls_independently_observed=False,
        scored_population=scored, predecessor_comparison=comparison,
        predecessor_algorithm_independent=False, full_challenge_pass=False,
        navigation_qualified=False, real_time_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--challenge-result-sha256', required=True)
    parser.add_argument('--supervisor-terminal-sha256', required=True)
    parser.add_argument('--definition-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(read_result(args.challenge_result_sha256, args.supervisor_terminal_sha256,
        args.definition_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
