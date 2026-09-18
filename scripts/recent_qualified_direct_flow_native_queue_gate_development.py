"""Preserve the original queue and contact pilot before the isolated maze3 run."""
import re
from scripts import await_go2_supervised_commitment_contact_native_v1 as predecessor
from scripts.supervised_commitment_contact_queue_gate_development import (
    verify_queue_completion, require_native_idle)
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import verify

WAIT_LAUNCH_SHA='4631acdc04f710c93f84d161ee428b7c8003ff7d663e4e75f0880697401f529b'


def predecessor_sources():
    verify_artifacts(predecessor.OUTPUT, {'launch.json':WAIT_LAUNCH_SHA})
    sources=read_json(predecessor.OUTPUT,'launch.json')['source_sha256']
    verify(sources)
    return sources


def verify_completed_predecessor(expected_sha,sources):
    if type(expected_sha) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha) is None:
        raise ValueError('explicit completed contact waiter SHA required before native launch')
    inherited=predecessor_sources()
    if any(sources.get(k)!=v for k,v in inherited.items()):
        raise ValueError('complete original waiter sources must remain unchanged')
    root=predecessor.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('predecessor terminal failure retained; no bypass')
    verify_artifacts(root,{'result.json':expected_sha})
    result=read_json(root,'result.json')
    if (result['status']!='SUPERVISED_COMMITMENT_CONTACT_NATIVE_WAIT_V1_COMPLETE'
            or result['source_sha256']!=inherited or result['automatic_retry'] is not False
            or result['artifact_sha256'].get('launch.json')!=WAIT_LAUNCH_SHA):
        raise ValueError('complete fixed original waiter required')
    verify_artifacts(root,result['artifact_sha256'])
    report=predecessor.authenticate_native(sources)
    if report!=result['report'] or read_json(root,'native_completion.json')!=report:
        raise ValueError('original contact verifier and completed receipt must agree')
    queue=verify_queue_completion(report['queue_result_sha256'],sources)
    if read_json(root,'queue_completion.json')!=queue:
        raise ValueError('completed original ordered queue receipt must agree')
    verify(sources)
    verify_artifacts(root,result['artifact_sha256']|{'result.json':expected_sha})
    return dict(wait_result_sha256=expected_sha,wait_launch_sha256=WAIT_LAUNCH_SHA,
        native_completion=report,queue_completion=queue,
        original_contact_verifier_reexecuted=True,scientific_success_required=False)
