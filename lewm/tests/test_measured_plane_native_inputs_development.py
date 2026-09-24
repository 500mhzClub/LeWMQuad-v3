"""Queue completion is required; a successful predecessor outcome is not."""
from copy import deepcopy
import pytest
from scripts import measured_plane_native_inputs_development as inputs


def example():
    launch = dict(source_sha256={'source':'hash'},waiter_pid=inputs.QUEUE_OWNER['pid'],
        boot_id=inputs.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    result = dict(status='CHAINED_ANCHOR_MAZE02_NATIVE_WAIT_V1_COMPLETE',
        source_sha256=deepcopy(launch['source_sha256']),artifact_sha256={'launch.json':inputs.QUEUE_LAUNCH_SHA},
        automatic_retry=False,navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,
        goal_achieved=False,report=dict(complete_native_worker_and_artifact_roster_verified=True,
            actual_physical_prefix_reconstructed=True,scientific_success_required=False,
            measured_round_trip_successes=0))
    return result,launch


def test_negative_scientific_result_does_not_block_completed_queue():
    result,launch=example()
    assert inputs.queue_identity(result,launch)['measured_round_trip_successes'] == 0


@pytest.mark.parametrize('fault',['incomplete','source','launch','owner','boot','retry','qualification','audit','prefix','selection'])
def test_unfinished_or_changed_queue_evidence_rejected(fault):
    result,launch=example()
    if fault == 'incomplete': result['status']='RUNNING'
    elif fault == 'source': result['source_sha256']={}
    elif fault == 'launch': result['artifact_sha256']['launch.json']='different'
    elif fault == 'owner': launch['waiter_pid']+=1
    elif fault == 'boot': launch['boot_id']='different'
    elif fault == 'retry': result['automatic_retry']=True
    elif fault == 'qualification': result['navigation_qualified']=True
    elif fault == 'audit': result['report']['complete_native_worker_and_artifact_roster_verified']=False
    elif fault == 'prefix': result['report']['actual_physical_prefix_reconstructed']=False
    else: result['report']['scientific_success_required']=True
    with pytest.raises(ValueError): inputs.queue_identity(result,launch)


def test_live_original_owner_blocks_before_opening_any_result(monkeypatch):
    monkeypatch.setattr(inputs.run,'owner_live',lambda owner:True)
    monkeypatch.setattr(inputs.run,'verify_artifacts',lambda *a:pytest.fail('opened result while owner live'))
    with pytest.raises(ValueError,match='end first'): inputs.admit_queue('unavailable',{})
