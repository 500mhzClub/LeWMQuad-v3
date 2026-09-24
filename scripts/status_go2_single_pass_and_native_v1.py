"""Read-only owner and bounded progress inspection; no outcome qualification."""
from datetime import datetime, timezone
import hashlib
import json

from scripts import status_go2_active_navigation_development_v1 as previous

REPLAY = previous.BASE/'go2_single_pass_body_projected_late_history_v1_attempt_001'
LAUNCH_SHA = '86e325f68d0f7d5f389e911009c9ccdf1b6c3d24291e9a2c974af838ed1307d8'


def snapshot():
    path = previous.ordinary(REPLAY/'launch.json')
    if path.stat().st_size > 2_000_000:
        raise ValueError('bounded original execution record required')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LAUNCH_SHA:
        raise ValueError('original replay launch identity changed')
    launch = json.loads(data)
    old = previous.snapshot()
    return dict(utc=datetime.now(timezone.utc).isoformat(), diagnostic_only=True,
        paired_replay=dict(owner=previous.owner_status(launch['owner'], launch['boot_id']),
            last_complete_frame=previous.last_complete(REPLAY/'comparison.jsonl', 'frame'),
            terminal_presence_unverified={name: previous.ordinary(REPLAY/name).exists()
                for name in ('result.json', 'failure.json')}),
        native=old['extended_budget_native'], resources=old['resources'],
        completion_receipt_present_unverified=previous.ordinary(previous.ROOT/
            'docs/go2_single_pass_body_projected_completion_verification_2026-09-11.json').exists(),
        raw_model_admission_performed=False, runtime_artifact_authentication_performed=False,
        navigation_outcome_verified=False)


if __name__ == '__main__': print(json.dumps(snapshot(), indent=2, allow_nan=False))
