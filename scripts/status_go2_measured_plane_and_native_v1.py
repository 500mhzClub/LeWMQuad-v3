"""Bounded status for the exact observer diagnostic and existing native owner."""
from datetime import datetime, timezone
import hashlib
import json

from scripts import status_go2_active_navigation_development_v1 as previous

REPLAY = previous.BASE/'go2_measured_plane_observer_history_v1_attempt_001'
LAUNCH_SHA = '8f09edbb77d103e3fe37e6f021da16be810a1696b588dea2264e98489f30afe1'


def snapshot():
    path = previous.ordinary(REPLAY/'launch.json')
    if path.stat().st_size > 8*1024**2:
        raise ValueError('bounded fixed observer launch required')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LAUNCH_SHA:
        raise ValueError('original observer launch identity changed')
    launch = json.loads(data)
    old = previous.snapshot()
    return dict(utc=datetime.now(timezone.utc).isoformat(), diagnostic_only=True,
        observer_history=dict(owner=previous.owner_status(launch['owner'], launch['boot_id']),
            last_complete_progress_frame=previous.last_complete(REPLAY/'progress.jsonl', 'frame'),
            maximum_frames=launch['maximum_frames'],
            terminal_presence_unverified={n: previous.ordinary(REPLAY/n).exists()
                for n in ('result.json','failure.json')}),
        native=old['extended_budget_native'], resources=old['resources'],
        completion_receipt_present_unverified=previous.ordinary(previous.ROOT/
            'docs/go2_measured_plane_observer_history_completion_2026-09-11.json').exists(),
        raw_model_admission_performed=False, runtime_artifact_authentication_performed=False,
        navigation_outcome_verified=False)


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2, allow_nan=False))
