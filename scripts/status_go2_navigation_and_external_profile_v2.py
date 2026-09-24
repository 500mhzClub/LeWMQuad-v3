"""Small read-only status view; no raw/profile reads or outcome qualification."""
from datetime import datetime, timezone
import hashlib
import json

from scripts import status_go2_active_navigation_development_v1 as previous

PROFILE=previous.BASE/'go2_body_projected_external_sampling_v2_attempt_001'
RECORDS={
    'parent':(PROFILE/'launch.json','6d0a33721486c96a8ff3c20a7c0ae2248087e2face8f42cc502c86884dbd69f1'),
    'child':(PROFILE/'child_execution.json','78981e6c2e26386dc693147f3e21eb89f01e1a934d4e2c935da2b7d765f33cd7'),
    'profiler':(PROFILE/'profiler_execution.json','3ac3a4e79dac89bd66ee7807c2d6f03de6af7d5eb7f9b6fb8dddcf204caea826'),
    'watcher':(previous.ROOT/'docs/go2_body_projection_external_profile_v2_completion_watch_execution_2026-09-11.json',
        '5c48f38827389851aa7129290c2e827d3822d113c34f99d156b429b29a78f537'),
}


def snapshot():
    owners={}
    for name,(path,sha) in RECORDS.items():
        previous.ordinary(path)
        if path.stat().st_size>2_000_000:raise ValueError('bounded original execution record required')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('original execution identity changed')
        record=json.loads(data)
        owners[name]=previous.owner_status(record['owner'],record['boot_id'])
    old=previous.snapshot()
    native=old['extended_budget_native']
    collected=previous.ordinary(previous.NATIVE_ROOT/
        'no_rgb_direct_extended_budget_anchored_maze_02/result.json').exists()
    return dict(utc=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,
        profile=dict(owners=owners,last_complete_frame=previous.last_complete(PROFILE/'comparison.jsonl','frame'),
            terminal_presence_unverified={name:previous.ordinary(PROFILE/name).exists() for name in
                ('child_result.json','result.json','failure.json','child_failure.json')},
            sampling_mode='nonblocking'),
        native=native | dict(collection_receipt_present_unverified=collected),
        resources=old['resources'],runtime_artifact_authentication_performed=False,
        profile_read=False,raw_model_admission_performed=False,navigation_outcome_verified=False)


if __name__=='__main__':print(json.dumps(snapshot(),indent=2,allow_nan=False))
