"""Authenticate the completed prospective prefix before a fresh longer native run."""
from scripts import extended_return_budget_native_prefix_development as prefix
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit

replay = prefix.replay
run = replay.run
old_inputs = replay.inputs.native_inputs
native = old_inputs.native
SOURCE = 'scripts/extended_return_budget_native_inputs_development.py'
TEST = 'lewm/tests/test_extended_return_budget_native_inputs_development.py'
PREFIX_ARTIFACTS = {'launch.json','context_decisions.jsonl.gz','state_checks.json',
    'identities.json','resource_monitor.jsonl','report.json'}


def prepared_sources(seeds=()):
    # Source preparation is possible before the prospective replay is launched.
    return replay.inputs.prepared_sources((replay.SOURCE,replay.PROTOCOL,*replay.TESTS,SOURCE,TEST,*seeds))


def _bound_inputs(result_sha,sources):
    root=replay.OUTPUT
    if (type(result_sha) is not str or len(result_sha)!=64
            or any(c not in '0123456789abcdef' for c in result_sha)):
        raise ValueError('actual completed prospective prefix result SHA-256 required')
    launch=run.read_json(root,'launch.json')
    if (run.owner_live(launch['owner'])
            or launch['boot_id']!=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('original prospective prefix owner must end on the same boot')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve prospective prefix failure without a native bypass')
    expected_launch=dict(protocol=replay.PROTOCOL,baseline_navigation_ticks=4000,candidate_navigation_ticks=8000,
        maximum_prefix_observations=4004,planned_state_frames=replay.pair.expected_states(4004),
        stop_at_first_normalized_decision_difference=True,preceding_timing_and_native_owners_ended=True,
        cpu_replay_serialization_checked=True,native_execution=False,automatic_retry=False)
    if (any(type(launch.get(k)) is not type(v) or launch[k]!=v for k,v in expected_launch.items())
            or not {replay.SOURCE,replay.PROTOCOL}<=launch['source_sha256'].keys()
            or any(sources.get(p)!=sha for p,sha in launch['source_sha256'].items())
            or any(launch['source_sha256'].get(p)!=sources.get(p) for p in (replay.SOURCE,replay.PROTOCOL))):
        raise ValueError('exact prepared prospective replay definition and source ancestry required')
    run.verify_artifacts(root,{'result.json':result_sha})
    result=run.read_json(root,'result.json');launch_sha=run.digest(root/'launch.json')
    required=dict(status='EXTENDED_RETURN_BUDGET_CONTROLLER_PREFIX_V1_COMPLETE',
        original_native_result_sha256=replay.inputs.NATIVE_RESULT_SHA,
        complete_output_and_public_packets_rechecked=True,
        original_native_and_timing_inputs_reauthenticated_before_and_after=True,
        original_physical_prefix_reconstructed_before_and_after=True,
        scientific_budget_only_prefix_supported=True,changed_command_executed=False,native_execution=False,
        automatic_retry=False,navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    if (any(type(result.get(k)) is not type(v) or result[k]!=v for k,v in required.items())
            or result['source_sha256']!=launch['source_sha256']
            or set(result['artifact_sha256'])!=PREFIX_ARTIFACTS
            or result['artifact_sha256']['launch.json']!=launch_sha
            or result['completed_chained_timing_result_sha256']!=launch['input_admission']['completed_chained_timing_result_sha256']):
        raise ValueError('complete positive immutable prospective prefix result required')
    prefix_ids=result['artifact_sha256']|{'result.json':result_sha}
    run.verify_artifacts(root,prefix_ids)
    if run.canonical(run.read_json(root,'report.json'))!=run.canonical(result['report']):
        raise ValueError('complete saved prospective report required')
    bound=prefix.boundary(result['report'])
    replay.pair.require_admission(launch['input_admission'])
    old_launch=old_inputs.native_launch()
    if run.owner_live(old_launch['owner']) or old_inputs.worker_live():
        raise ValueError('both original native owners must remain ended')
    if (native.OUTPUT/'failure.json').exists() or (native.OUTPUT/'failure.json').is_symlink():
        raise ValueError('preserve original native operational failure')
    old_sha=replay.inputs.NATIVE_RESULT_SHA
    run.verify_artifacts(native.OUTPUT,{'result.json':old_sha})
    old=run.read_json(native.OUTPUT,'result.json');record=old_inputs.require_result(old,old_launch)
    native_ids=old['artifact_sha256']|{'result.json':old_sha}
    run.verify_artifacts(native.OUTPUT,native_ids)
    if (record['status']!=native.WORKER_STATUS or record['collection']['decisions']!=4014
            or record['collection']['schedule_terminal']!='MISSION_TICK_BUDGET_EXHAUSTED'
            or record['model_state_sha256']!=replay.pair.MODEL_SHA
            or launch['input_admission']['original_context_sha256']!=native_ids[native.CASE[0]+'/context_decisions.jsonl.gz']
            or launch['input_admission']['complete_raw_artifact_roster_sha256']!=run.fingerprint(native_ids)
            or run.canonical(run.read_json(native.OUTPUT,native.CASE[0]+'_worker_terminal.json'))!=run.canonical(record)):
        raise ValueError('same original completed native worker, model and complete input population required')
    require_raw_audit(record,run.read_json(native.OUTPUT,native.CASE[0]+'_audit.json'),learned=True)
    if any(sources.get(p)!=sha for p,sha in old_launch['source_sha256'].items()):
        raise ValueError('complete immutable original native source ancestry required')
    run.verify(sources)
    receipt=dict(controller_prefix_result_sha256=result_sha,controller_prefix_launch_sha256=launch_sha,
        original_native_result_sha256=old_sha,completed_chained_timing_result_sha256=result['completed_chained_timing_result_sha256'],
        prefix_report=result['report'],boundary=bound,prefix_artifact_sha256=prefix_ids,
        original_native_artifact_sha256=native_ids,model_state_sha256=replay.pair.MODEL_SHA,
        original_owners_ended=True,complete_predecessor_artifact_rosters_reauthenticated=True,
        initial_public_prefix_reconstruction_required=False,completed_public_prefix_audit_reused=True,
        prior_controller_or_physics_execution_repeated=False,
        prior_native_physical_prefix_reconstruction_repeated=False,
        native_execution_started=False,navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)
    return receipt,launch


def admit(result_sha,sources):
    # The completed prefix result already binds its full public-packet and
    # saved-output audit. Reuse it after checking all bound input identities.
    receipt,_=_bound_inputs(result_sha,sources)
    return receipt


def verify_bound(receipt,sources):
    # Rehash the immutable evidence without repeating its completed audit.
    actual,_=_bound_inputs(receipt['controller_prefix_result_sha256'],sources)
    if run.canonical(actual)!=run.canonical(receipt):
        raise ValueError('complete initially admitted native/prefix identities changed')
