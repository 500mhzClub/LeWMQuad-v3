"""Admission for a fresh measured-plane native episode after the existing queue.

The queue receipt is scheduling evidence. Its frozen waiter already checks the
queue's raw outcomes; this module does not rerun unrelated training or scenes.
The new experiment independently runs and audits its own complete episode.
"""
import json
from scripts import measured_plane_native_prefix_development as prefix
from scripts import await_go2_chained_anchor_maze02_native_v1 as queued
from scripts.startup_source_inventory_development import discover_sources

job,run = prefix.job,prefix.run
SOURCE = 'scripts/measured_plane_native_inputs_development.py'
TEST = 'lewm/tests/test_measured_plane_native_inputs_development.py'
QUEUE_LAUNCH_SHA = 'cf6703e83197d6c75df35b2b53834a47a53a293a06c64737ce94ade2ac0b87c1'
QUEUE_OWNER = dict(pid=2845479,created=1789129072.88,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',
    'scripts/await_go2_chained_anchor_maze02_native_v1.py'])


def completed_prefix():
    name = str(prefix.completed.OUTPUT.relative_to(run.ROOT))
    run.verify({name:prefix.COMPLETION_SHA})
    proof = json.loads(prefix.completed.OUTPUT.read_text())
    if (proof['status'] != 'MEASURED_PLANE_CONTROLLER_PREFIX_COMPLETION_VERIFIED'
            or proof['result_sha256'] != prefix.RESULT_SHA
            or proof['original_launch_sha256'] != prefix.completed.LAUNCH_SHA
            or proof['original_owner_ended'] is not True or proof['complete_output_stream_checked'] is not True
            or proof['actual_consumed_raw_packets_reconstructed'] != prefix.FRAMES):
        raise ValueError('exact full measured-plane controller completion required')
    prefix.boundary(proof['report'])
    run.verify(proof['source_sha256'])
    run.verify_artifacts(job.OUTPUT,proof['artifact_sha256'] | {'result.json':prefix.RESULT_SHA})
    prefix.completed.ended(run.read_json(job.OUTPUT,'launch.json'))
    return proof


def prepared_sources(seeds=()):
    proof = completed_prefix()
    run.verify_artifacts(queued.OUTPUT,{'launch.json':QUEUE_LAUNCH_SHA})
    queue = run.read_json(queued.OUTPUT,'launch.json')
    inherited = dict(proof['source_sha256'])
    for name,sha in queue['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('source ancestry conflict')
        inherited[name] = sha
    sources = discover_sources((SOURCE,TEST,*seeds,str(prefix.completed.OUTPUT.relative_to(run.ROOT))),inherited)
    run.verify(sources)
    return sources


def queue_identity(result,launch):
    if (result['status'] != 'CHAINED_ANCHOR_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != QUEUE_LAUNCH_SHA
            or launch['waiter_pid'] != QUEUE_OWNER['pid']
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or result['automatic_retry'] is not False
            or any(result[k] is not False for k in ('navigation_qualified','real_time_qualified','hardware_qualified','goal_achieved'))):
        raise ValueError('exact complete existing native queue waiter required')
    report = result['report']
    if (report['complete_native_worker_and_artifact_roster_verified'] is not True
            or report['actual_physical_prefix_reconstructed'] is not True
            or report['scientific_success_required'] is not False):
        raise ValueError('complete original native verification without selecting scientific success required')
    return report


def admit_queue(result_sha,sources):
    if run.owner_live(QUEUE_OWNER): raise ValueError('existing native queue owner must end first')
    root = queued.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve failed existing queue; do not bypass or retry')
    run.verify_artifacts(root,{'launch.json':QUEUE_LAUNCH_SHA,'result.json':result_sha})
    launch = run.read_json(root,'launch.json'); result = run.read_json(root,'result.json')
    report = queue_identity(result,launch)
    if any(sources.get(k) != v for k,v in launch['source_sha256'].items()):
        raise ValueError('all existing queue sources required in frozen source union')
    ids = result['artifact_sha256'] | {'result.json':result_sha}
    run.verify_artifacts(root,ids)
    if run.read_json(root,'native_completion.json') != report:
        raise ValueError('same complete native waiter report required')
    run.verify_artifacts(queued.native.OUTPUT,{'result.json':report['native_result_sha256']})
    child = run.read_json(queued.native.OUTPUT,'result.json')
    if (child['status'] != 'CHAINED_ANCHOR_MAZE02_PILOT_V1_COMPLETE'
            or child['measured_round_trip_successes'] != report['measured_round_trip_successes']
            or any(sources.get(k) != v for k,v in child['source_sha256'].items())):
        raise ValueError('same completed child identity attested by the original waiter required')
    run.verify(sources)
    return dict(waiter_result_sha256=result_sha,waiter_artifact_sha256=ids,
        native_result_sha256=report['native_result_sha256'],original_queue_owner_ended=True,
        original_queue_completion_receipt_verified=True,queue_scientific_success_required=False,
        unrelated_queue_raw_audits_reexecuted=False,queue_outcomes_used_as_new_navigation_evidence=False)


def admit(result_sha,sources):
    proof = completed_prefix()
    queue = admit_queue(result_sha,sources)
    # Reuse the already admitted, ended original worker for the paired prefix.
    worker_name = str(job.worker.OUTPUT.relative_to(run.ROOT))
    run.verify({worker_name:job.WORKER_ADMISSION_SHA})
    worker = json.loads(job.worker.OUTPUT.read_text())
    if (worker['status'] != 'EXTENDED_BUDGET_COMPLETED_WORKER_ADMITTED'
            or worker['model_state_sha256'] != job.MODEL_SHA
            or worker['original_worker_ended'] is not True or worker['model_state_unchanged'] is not True
            or run.owner_live(worker['original_worker'])):
        raise ValueError('same ended worker and assigned unchanged model required')
    run.verify_artifacts(job.native.OUTPUT,worker['artifact_sha256'])
    run.verify(sources)
    return dict(controller_completion_sha256=prefix.COMPLETION_SHA,controller_result_sha256=prefix.RESULT_SHA,
        prefix_report=proof['report'],prefix_artifact_sha256=proof['artifact_sha256'] | {'result.json':prefix.RESULT_SHA},
        worker_admission_sha256=job.WORKER_ADMISSION_SHA,worker_artifact_sha256=worker['artifact_sha256'],
        model_state_sha256=job.MODEL_SHA,queue=queue,full_training_ancestry_reexecuted=False,
        native_execution=False,navigation_qualified=False,goal_achieved=False)
