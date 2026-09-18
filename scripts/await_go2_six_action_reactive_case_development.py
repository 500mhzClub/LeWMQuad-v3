"""Append one never-started six-action reactive case to the current serial queue."""
import json
import os
from pathlib import Path
import time

import psutil

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
PREDECESSOR = BASE/'go2_stop_conditioned_independent_00_current_planning_seed_2026091001_full_jepa_v1_attempt_001'
OUTPUT = BASE/'go2_stop_conditioned_independent_00_six_action_reactive_no_model_v1_attempt_001'
QUEUE_PID = 3196768
QUEUE_CREATED = 1789286098.74


def live():
    try:
        owner = psutil.Process(QUEUE_PID)
        return abs(owner.create_time()-QUEUE_CREATED)<.01 and owner.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False


def main():
    assert live() and not OUTPUT.exists() and not OUTPUT.is_symlink()
    owner = psutil.Process()
    receipt = dict(status='SIX_ACTION_REACTIVE_CASE_QUEUED', pid=owner.pid,
        created=owner.create_time(), waiting_for_pid=QUEUE_PID, waiting_for_created=QUEUE_CREATED,
        preceding_root=str(PREDECESSOR), output=str(OUTPUT), layout_index=0,
        mode='six_action_reactive', model_name=None, automatic_retry=False,
        serial_collection_and_audit=True, start_requires_scientific_success=False,
        start_requires_operational_completion=True)
    with Path('docs/go2_six_action_reactive_queue_2026-09-13.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt), flush=True)
    while live(): time.sleep(5)
    if (PREDECESSOR/'failure.json').exists(): raise RuntimeError('preceding queue failed; preserve evidence')
    result = json.loads((PREDECESSOR/'result.json').read_text())
    assert result['status'] == 'STOP_CONDITIONED_INDEPENDENT_CASE_COMPLETE'
    assert result['assignment'] == dict(layout_index=0, mode='current_planning',
        model_name='seed_2026091001_full_jepa', training_seed=2026091001,
        condition='jepa', variant='full')
    python = str(Path('.generated/venvs/genesis_rocm_0_4_6_v1/bin/python').absolute())
    os.execv(python, [python, '-B', 'scripts/run_go2_six_action_reactive_independent_case_v1.py'])


if __name__ == '__main__': main()
