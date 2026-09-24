"""Authenticate a completed tracking replay before a future native experiment.

This helper creates no output, model, replay, process, or simulator scene.
The future native launcher must separately freeze its protocol/source closure,
admit hardware resources, serialize native execution, and audit its full run.
"""
from scripts import await_go2_measured_plane_chained_controller_prefix_v1 as chained_wait
from scripts import await_go2_reactive_measured_plane_native_v1 as reactive_wait
from scripts import measured_plane_chained_native_prefix_development as prefix
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

run = chained_wait.run
SOURCE = 'scripts/measured_plane_chained_native_inputs_development.py'
TEST = 'lewm/tests/test_measured_plane_chained_native_inputs_development.py'
CHAINED_LAUNCH_SHA = '4e0ed3e7a14c61b8a31ded1c83f34e15643e40bcd7668840bf163bc485311416'
CHAINED_OWNER = dict(pid=2930187, created=1789170044.92, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/await_go2_measured_plane_chained_controller_prefix_v1.py'])
REACTIVE_LAUNCH_SHA = 'b896918786c965a1b6e7d70aa796ca26f0983917cc40f26623327d2402b0150e'
REACTIVE_OWNER = dict(pid=2922235, created=1789165929.81, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/await_go2_reactive_measured_plane_native_v1.py'])
REACTIVE_WAIT_RESULT_SHA = 'b11b395e19eecaf1a7b8d7052b7762eea31db9e9a4266ae5640d935a02133c75'
REACTIVE_NATIVE_RESULT_SHA = 'd20747931ff557206698ad7f5ae8d46145d2af5317b12cc7ed62a772039172f6'
NOMINAL_WAIT_RESULT_SHA = '287e83fc4b3591a24db68afc5bb97bb10c13495e99f6eb112310b12495993e28'
LEARNED_RESULT_SHA = '4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18'


def _launch(module, launch_sha, owner):
    run.verify_artifacts(module.OUTPUT, {'launch.json': launch_sha})
    launch = run.read_json(module.OUTPUT, 'launch.json')
    if (launch['owner'] != owner
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or launch['automatic_retry'] is not False):
        raise ValueError('exact original waiter identity, boot and no-retry launch required')
    run.verify(launch['source_sha256'])
    return launch


def launches():
    chained = _launch(chained_wait, CHAINED_LAUNCH_SHA, CHAINED_OWNER)
    reactive = _launch(reactive_wait, REACTIVE_LAUNCH_SHA, REACTIVE_OWNER)
    if (chained['predecessor_owner'] != chained_wait.job.CPU_OWNER
            or chained['predecessor_launch_sha256'] != chained_wait.job.CPU_LAUNCH_SHA
            or reactive['original_owner'] != reactive_wait.native.inputs.NOMINAL_WAIT_OWNER
            or reactive['original_launch_sha256'] != reactive_wait.native.inputs.NOMINAL_WAIT_LAUNCH_SHA):
        raise ValueError('original timing and nominal queue predecessors required')
    return chained, reactive


def prepared_sources(seeds=()):
    chained, reactive = launches()
    sources = discover_sources((SOURCE, TEST, *seeds), merge_sources(
        chained['source_sha256'], reactive['source_sha256']))
    run.verify(sources)
    return sources


def _completed_waiter(module, launch, launch_sha, result_sha, status, completion, sources, flags):
    root = module.OUTPUT
    if run.owner_live(launch['owner']):
        raise ValueError('original waiter must finish and end before native admission')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed predecessor; no automatic native bypass')
    run.verify_artifacts(root, {'result.json': result_sha})
    result = run.read_json(root, 'result.json')
    required = {'launch.json', 'events.jsonl', completion,
        'replay_stdout.log' if module is chained_wait else 'native_stdout.log'}
    if (result['status'] != status or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256'].get('launch.json') != launch_sha
            or set(result['artifact_sha256']) != required
            or any(sources.get(k) != v for k, v in result['source_sha256'].items())
            or any(result.get(k) is not False for k in flags)):
        raise ValueError('complete exact waiter result, source closure and artifact roster required')
    ids = result['artifact_sha256'] | {'result.json': result_sha}
    run.verify_artifacts(root, ids)
    if result['report'] != run.read_json(root, completion):
        raise ValueError('same saved complete waiter verification receipt required')
    return result, ids


def admit(chained_wait_result_sha, sources):
    chained_launch, reactive_launch = launches()
    # Reject the live reserved replay before doing any expensive reauthentication.
    for launch in (chained_launch, reactive_launch):
        if run.owner_live(launch['owner']):
            raise ValueError('both original queue owners must finish and end first')
    chained, chained_ids = _completed_waiter(chained_wait, chained_launch,
        CHAINED_LAUNCH_SHA, chained_wait_result_sha,
        'MEASURED_PLANE_CHAINED_CONTROLLER_WAIT_V1_COMPLETE', 'completion.json', sources,
        ('automatic_retry', 'native_execution', 'navigation_qualified', 'goal_achieved'))
    reactive, reactive_ids = _completed_waiter(reactive_wait, reactive_launch,
        REACTIVE_LAUNCH_SHA, REACTIVE_WAIT_RESULT_SHA,
        'REACTIVE_MEASURED_PLANE_NATIVE_WAIT_V1_COMPLETE', 'native_completion.json', sources,
        ('automatic_retry', 'navigation_qualified', 'real_time_qualified', 'hardware_qualified', 'goal_achieved'))
    completed = chained['report']
    if (completed['native_result_sha256'] != LEARNED_RESULT_SHA
            or reactive['report']['learned_result_sha256'] != LEARNED_RESULT_SHA
            or reactive['report']['reactive_result_sha256'] != REACTIVE_NATIVE_RESULT_SHA
            or reactive['report']['nominal_wait_result_sha256'] != NOMINAL_WAIT_RESULT_SHA):
        raise ValueError('same completed learned and nominal/reactive pilots required')
    # These existing checkers reauthenticate full closed artifacts and reconstruct
    # receipts/consumed packets. They do not run controller inference or physics.
    actual = chained_wait.completed_child(sources, LEARNED_RESULT_SHA,
        completed['timing_waiter_result_sha256'])
    if actual != completed:
        raise ValueError('completed controller replay receipt must reconstruct exactly')
    replay = run.read_json(chained_wait.job.OUTPUT, 'result.json')
    if run.digest(chained_wait.job.OUTPUT/'result.json') != completed['replay_result_sha256']:
        raise ValueError('same authenticated completed controller replay result required')
    bound = prefix.boundary(replay['report'])
    if reactive_wait.completed_child(sources, NOMINAL_WAIT_RESULT_SHA) != reactive['report']:
        raise ValueError('completed reactive native verification must reconstruct exactly')
    run.verify(sources)
    run.verify_artifacts(chained_wait.OUTPUT, chained_ids)
    run.verify_artifacts(reactive_wait.OUTPUT, reactive_ids)
    return dict(chained_wait_result_sha256=chained_wait_result_sha,
        controller_replay_result_sha256=completed['replay_result_sha256'],
        timing_waiter_result_sha256=completed['timing_waiter_result_sha256'],
        learned_result_sha256=LEARNED_RESULT_SHA,
        reactive_wait_result_sha256=REACTIVE_WAIT_RESULT_SHA,
        reactive_result_sha256=REACTIVE_NATIVE_RESULT_SHA,
        nominal_wait_result_sha256=NOMINAL_WAIT_RESULT_SHA,
        prefix_report=replay['report'], boundary=bound,
        original_owners_ended=True, complete_predecessor_artifact_rosters_reauthenticated=True,
        complete_consumed_controller_rows_and_public_packets_reconstructed=True,
        completed_reactive_physical_prefix_reconstructed=reactive['report']['actual_physical_prefix_reconstructed'],
        predecessor_scientific_success_required=False, controller_replay_reexecuted=False,
        future_native_execution_started=False, future_native_physical_prefix_reconstructed=False,
        hardware_resources_admitted=False, native_execution_serialization_checked=False,
        navigation_recovered=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
