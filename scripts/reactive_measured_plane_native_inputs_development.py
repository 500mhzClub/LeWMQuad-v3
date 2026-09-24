"""Admit complete learned and nominal episodes before the reactive comparison."""
from scripts import nominal_measured_plane_native_inputs_development as learned_inputs
from scripts import await_go2_nominal_measured_plane_native_v1 as nominal_wait
from scripts import reactive_measured_plane_native_prefix_development as prefix
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

run = prefix.run
NOMINAL_WAIT_LAUNCH_SHA = 'a31dc3ecde7c7e570d6b499ee783381e55eb1d87a97f702bcc65794458292994'
NOMINAL_WAIT_OWNER = dict(pid=2919853, created=1789164727.75, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/await_go2_nominal_measured_plane_native_v1.py'])


def waiter_launch():
    root = nominal_wait.OUTPUT
    run.verify_artifacts(root, {'launch.json': NOMINAL_WAIT_LAUNCH_SHA})
    launch = run.read_json(root, 'launch.json')
    if (launch['owner'] != NOMINAL_WAIT_OWNER
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or launch['original_launch_sha256'] != learned_inputs.LEARNED_LAUNCH_SHA
            or launch['original_owner'] != learned_inputs.LEARNED_OWNER):
        raise ValueError('exact current nominal waiter and learned predecessor required')
    run.verify(launch['source_sha256'])
    return launch


def prepared_sources(seeds):
    proof = prefix.completed_prefix(); nominal = waiter_launch()
    sources = discover_sources(seeds, merge_sources(proof['source_sha256'], nominal['source_sha256']))
    run.verify(sources)
    return sources


def require_waiter_result(result, launch):
    if (result['status'] != 'NOMINAL_MEASURED_PLANE_NATIVE_WAIT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != NOMINAL_WAIT_LAUNCH_SHA
            or result['automatic_retry'] is not False
            or any(result[k] is not False for k in
                ('navigation_qualified', 'real_time_qualified', 'hardware_qualified', 'goal_achieved'))):
        raise ValueError('complete exact nominal waiter without a success selection required')
    report = result['report']
    if (report['complete_native_worker_and_artifact_roster_verified'] is not True
            or report['actual_physical_prefix_reconstructed'] is not True
            or report['scientific_success_required'] is not False
            or report['raw_controller_audit_reexecuted'] is not False
            or type(report['nominal_measured_round_trip_successes']) is not int
            or report['nominal_measured_round_trip_successes'] not in (0, 1)):
        raise ValueError('completed original nominal verification with either scientific outcome required')
    return report


def admit(wait_sha, sources):
    launch = waiter_launch(); root = nominal_wait.OUTPUT
    if run.owner_live(NOMINAL_WAIT_OWNER): raise ValueError('nominal waiter must end before reactive native execution')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed nominal execution; no automatic reactive bypass')
    run.verify_artifacts(root, {'result.json': wait_sha})
    result = run.read_json(root, 'result.json'); report = require_waiter_result(result, launch)
    run.verify_artifacts(root, result['artifact_sha256'])
    if report != run.read_json(root, 'native_completion.json'):
        raise ValueError('same complete nominal verification receipt required')
    # The original waiter has already reconstructed the physical prefix and
    # verified the full raw worker. Reauthenticate the bound closed artifacts
    # here without repeating its controller replay or predecessor recursion.
    native = nominal_wait.native; native_root = native.OUTPUT
    run.verify_artifacts(native_root, {'result.json': report['nominal_result_sha256']})
    child = run.read_json(native_root, 'result.json'); child_launch = run.read_json(native_root, 'launch.json')
    if (run.owner_live(child_launch['owner']) or (native_root/'failure.json').exists()
            or (native_root/'failure.json').is_symlink()
            or child_launch['boot_id'] != launch['boot_id']
            or child['status'] != 'NOMINAL_MEASURED_PLANE_MAZE02_V1_COMPLETE'
            or child['source_sha256'] != child_launch['source_sha256']
            or child['learned_result_sha256'] != report['learned_result_sha256']
            or child['measured_round_trip_successes'] != report['nominal_measured_round_trip_successes']
            or len(child['conditions']) != 1):
        raise ValueError('same ended nominal child and reported outcome required')
    run.verify_artifacts(native_root, child['artifact_sha256'])
    record = child['conditions'][0]; name = native.CASE[0]
    if record != run.read_json(native_root, name+'_worker_terminal.json'):
        raise ValueError('same complete nominal worker required')
    native.require_worker(record, run.read_json(native_root, name+'_audit.json'))
    learned = learned_inputs.admit(report['learned_result_sha256'], sources)
    reactive = prefix.completed_prefix()
    for prior in (launch['source_sha256'], child['source_sha256'], reactive['source_sha256']):
        if any(sources.get(k) != v for k, v in prior.items()):
            raise ValueError('all original frozen source bindings required')
    run.verify(sources)
    return dict(nominal_wait_result_sha256=wait_sha, nominal_result_sha256=report['nominal_result_sha256'],
        learned_result_sha256=report['learned_result_sha256'], learned_admission=learned,
        reactive_prefix_result_sha256=prefix.RESULT_SHA, prefix_report=reactive['report'],
        complete_predecessor_artifact_rosters_reauthenticated=True,
        original_verification_receipt_retained=True, predecessor_controller_audits_reexecuted=False,
        predecessor_scientific_success_required=False, original_owners_ended=True)
