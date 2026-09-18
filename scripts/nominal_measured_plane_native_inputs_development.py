"""Admit an ended learned episode and the completed nominal intervention."""
from scripts import nominal_measured_plane_native_prefix_development as prefix
from scripts import run_go2_measured_plane_dispatch_recovery_v1 as learned
from scripts.startup_source_inventory_development import discover_sources

run = prefix.run
job = prefix.replay.job
LEARNED_LAUNCH_SHA = '93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb'
LEARNED_OWNER = dict(pid=2916106, created=1789162140.42, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/run_go2_measured_plane_dispatch_recovery_v1.py'])


def completed_prefix():
    root = prefix.replay.OUTPUT
    run.verify_artifacts(root, {'launch.json': prefix.LAUNCH_SHA, 'result.json': prefix.RESULT_SHA})
    launch = run.read_json(root, 'launch.json'); result = run.read_json(root, 'result.json')
    if ((root/'failure.json').exists() or (root/'failure.json').is_symlink()
            or run.owner_live(launch['owner'])
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or result['status'] != 'MEASURED_PLANE_FORECAST_SOURCE_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != prefix.LAUNCH_SHA
            or result['original_raw_inputs_reauthenticated_before_and_after'] is not True):
        raise ValueError('exact ended completed forecast-source prefix required')
    prefix.boundary(result['report'])
    run.verify(result['source_sha256']); run.verify_artifacts(root, result['artifact_sha256'])
    if run.read_json(root, 'report.json') != result['report']:
        raise ValueError('same saved complete forecast-source report required')
    prefix.replay.check_output(result['report'])
    return result


def learned_launch():
    run.verify_artifacts(learned.OUTPUT, {'launch.json': LEARNED_LAUNCH_SHA})
    launch = run.read_json(learned.OUTPUT, 'launch.json')
    if (launch['owner'] != LEARNED_OWNER
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('exact original learned parent owner and boot required')
    run.verify(launch['source_sha256'])
    return launch


def prepared_sources(seeds):
    replay = completed_prefix(); launch = learned_launch()
    inherited = dict(launch['source_sha256'])
    for name, sha in replay['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('source ancestry conflict')
        inherited[name] = sha
    sources = discover_sources(seeds, inherited); run.verify(sources)
    return sources


def require_learned_result(result, launch):
    if (result['status'] != 'MEASURED_PLANE_DISPATCH_RECOVERY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != LEARNED_LAUNCH_SHA
            or len(result['conditions']) != 1 or result['automatic_retry'] is not False
            or result['original_failed_waiters_preserved'] is not True
            or result['controller_completion_sha256'] != learned.original.prefix.COMPLETION_SHA
            or any(result[k] is not False for k in
                ('navigation_qualified', 'real_time_qualified', 'hardware_qualified', 'goal_achieved'))):
        raise ValueError('complete exact fresh learned episode without selecting success required')
    record = result['conditions'][0]
    if result['measured_round_trip_successes'] != int(record['verified_round_trip']):
        raise ValueError('actual learned physical outcome accounting required')
    return record


def admit(result_sha, sources):
    launch = learned_launch()
    if run.owner_live(LEARNED_OWNER): raise ValueError('learned native parent must end before nominal execution')
    root = learned.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed learned execution; no automatic bypass')
    run.verify_artifacts(root, {'result.json': result_sha})
    result = run.read_json(root, 'result.json'); record = require_learned_result(result, launch)
    ids = result['artifact_sha256'] | {'result.json': result_sha}
    run.verify_artifacts(root, ids)
    name = learned.original.CASE[0]
    required = [name+'/'+n for n in learned.original.pipeline.artifacts(2, record['collection'])]
    required += [name+s for s in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json')]
    if any(n not in ids for n in required):
        raise ValueError('entire original learned raw artifact roster required')
    if (run.read_json(root, name+'_worker_terminal.json') != record
            or run.read_json(root/name, 'result.json') != record['collection']
            or any(ids.get(path) != sha for path, sha in record['artifact_sha256'].items())
            or record['worker_log_sha256'] != ids[name+'_worker.log']):
        raise ValueError('all complete learned worker receipts and artifact bindings required')
    audit = run.read_json(root, name+'_audit.json')
    learned.original.require_worker(record, audit)
    if (run.read_json(root, name+'_readout.json') != record['readout']
            or run.read_json(root, name+'_prefix_comparison.json') != record['prefix_comparison']):
        raise ValueError('same original learned readout and physical prefix required')
    if record['collection']['decisions'] < prefix.FRAMES:
        raise ValueError('learned episode must contain the already verified four-observation intervention')
    proof = completed_prefix()
    if any(sources.get(path) != sha for path, sha in launch['source_sha256'].items()):
        raise ValueError('all learned sources required in new frozen source union')
    run.verify(sources)
    return dict(learned_result_sha256=result_sha, learned_launch_sha256=LEARNED_LAUNCH_SHA,
        learned_artifact_sha256=ids, learned_owner_ended=True,
        learned_complete_raw_audit_verified_from_frozen_worker=True,
        learned_scientific_success_required=False, learned_raw_audit_reexecuted=False,
        forecast_prefix_result_sha256=prefix.RESULT_SHA, prefix_report=proof['report'],
        forecast_prefix_artifact_sha256=proof['artifact_sha256'],
        model_state_sha256=prefix.replay.job.MODEL_SHA, native_execution=False,
        navigation_qualified=False, goal_achieved=False)
