"""Authenticate the exact completed chained native input for a future replay."""
import psutil

from scripts import run_go2_measured_plane_chained_maze02_v1 as native
from scripts.startup_source_inventory_development import discover_sources

run = native.run
SOURCE = 'scripts/measured_plane_chained_full_history_inputs_development.py'
TEST = 'lewm/tests/test_measured_plane_chained_full_history_inputs_development.py'
LAUNCH_SHA = '0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff'
BOOT_ID = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
OWNER = dict(pid=2992412, created=1789193084.73, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/run_go2_measured_plane_chained_maze02_v1.py', '--chained-wait-result-sha256',
    '0a41c3177c2696c86d4b8d21a56ed67baba936e8999184a105f7b804462b494b'])
WORKER_PID = 2994743
WORKER_CREATED = 1789194027.81


def native_launch():
    run.verify_artifacts(native.OUTPUT, {'launch.json': LAUNCH_SHA})
    launch = run.read_json(native.OUTPUT, 'launch.json')
    if (launch['owner'] != OWNER or launch['boot_id'] != BOOT_ID
            or run.Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT_ID):
        raise ValueError('exact original chained native launch owner and boot required')
    run.verify(launch['source_sha256'])
    return launch


def worker_live():
    try:
        return psutil.Process(WORKER_PID).create_time() == WORKER_CREATED
    except psutil.NoSuchProcess:
        return False


def prepared_sources(seeds=()):
    launch = native_launch()
    sources = discover_sources((SOURCE, TEST, *seeds), launch['source_sha256'])
    run.verify(sources)
    return sources


def require_result(result, launch):
    expected = dict(status='MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE',
        chained_wait_result_sha256=launch['input_admission']['chained_wait_result_sha256'],
        controller_replay_result_sha256=launch['input_admission']['controller_replay_result_sha256'],
        learned_result_sha256=native.inputs.LEARNED_RESULT_SHA,
        reused_layout_executions=1, new_independent_layout_executions=0,
        measured_plane_constrained_estimator=True, chained_anchor_reacquisition_enabled=True,
        original_bridge_allowance_unchanged=True, single_pass_timing_change_adopted=False,
        automatic_retry=False, model_training=False, navigation_qualified=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    if (any(type(result.get(k)) is not type(v) or result[k] != v for k,v in expected.items())
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256'].get('launch.json') != LAUNCH_SHA
            or type(result['conditions']) is not list or len(result['conditions']) != 1):
        raise ValueError('complete exact chained result and unchanged scientific scope required')
    record = result['conditions'][0]
    success = record['verified_round_trip']
    if (type(success) is not bool or type(result['measured_round_trip_successes']) is not int
            or result['measured_round_trip_successes'] != int(success)):
        raise ValueError('actual physical outcome accounting required without selecting success')
    return record


def admit(result_sha, sources):
    launch = native_launch()
    if run.owner_live(OWNER) or worker_live():
        raise ValueError('original chained parent and worker must end before replay input admission')
    root = native.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed chained execution without bypass')
    if (type(result_sha) is not str or len(result_sha) != 64
            or any(c not in '0123456789abcdef' for c in result_sha)):
        raise ValueError('actual completed chained result SHA-256 required')
    run.verify_artifacts(root, {'result.json': result_sha})
    result = run.read_json(root, 'result.json')
    record = require_result(result, launch)
    ids = result['artifact_sha256'] | {'result.json': result_sha}
    run.verify_artifacts(root, ids)
    name, layout = native.CASE[:2]
    required = {name+'/'+leaf for leaf in native.pipeline.artifacts(layout, record['collection'])}
    required.update(name+suffix for suffix in (
        '_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json'))
    required.update(('launch.json', 'resource_monitor.jsonl'))
    if not required <= ids.keys():
        raise ValueError('complete original chained raw artifact roster required')
    if (run.canonical(run.read_json(root, name+'_worker_terminal.json')) != run.canonical(record)
            or run.canonical(run.read_json(root/name, 'result.json')) != run.canonical(record['collection'])
            or any(ids.get(path) != sha for path,sha in record['artifact_sha256'].items())
            or record['worker_log_sha256'] != ids[name+'_worker.log']):
        raise ValueError('exact chained worker receipts and artifact bindings required')
    audit = run.read_json(root, name+'_audit.json')
    # Reconstruct the original physical prefix again, without rerunning either
    # controller or the complete native sensor/physics audit.
    native.require_worker(record, audit)
    if (run.canonical(run.read_json(root, name+'_readout.json')) != run.canonical(record['readout'])
            or run.canonical(run.read_json(root, name+'_prefix_comparison.json')) != run.canonical(record['prefix_comparison'])):
        raise ValueError('same complete native readout and reconstructed physical prefix required')
    from scripts.measured_plane_chained_full_history_timing_development import state_frames
    count = record['collection']['decisions']
    state_frames(count)
    if any(sources.get(path) != sha for path,sha in launch['source_sha256'].items()):
        raise ValueError('every immutable native source required in the new source union')
    run.verify(sources)
    run.verify_artifacts(root, ids)
    return dict(learned_result_sha256=result_sha, learned_launch_sha256=LAUNCH_SHA,
        native_case=name, frames=count, native_owner_ended=True, original_worker_ended=True,
        complete_raw_audit_verified=True, original_scientific_success_required=False,
        original_schedule_terminal=record['collection']['schedule_terminal'],
        original_verified_round_trip=record['verified_round_trip'],
        complete_raw_artifact_roster_sha256=run.fingerprint(ids),
        original_context_sha256=ids[name+'/context_decisions.jsonl.gz'],
        model_state_sha256=record['model_state_sha256'],
        original_physical_prefix_reconstructed=True, original_controller_replayed=False,
        original_full_raw_audit_reexecuted=False, native_execution=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False,
        goal_achieved=False)
