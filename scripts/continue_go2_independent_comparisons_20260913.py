"""Execute the seven never-started assignments after the stopped wait chain.

The supervised run's collection and raw audit finished, but its final inherited
source check failed. Preserve that failure. Each assignment below gets its first
native attempt, serially including its audit, with no automatic retry.
"""
import json
from pathlib import Path
import subprocess
import sys

import psutil

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
STANDARD = 'scripts/run_go2_stop_conditioned_independent_case_v1.py'
CONTACT = 'scripts/run_go2_commitment_contact_independent_case_v1.py'
CASES = (
    (0, 'frozen_reference', 'seed_2026091001_full_direct'),
    (0, 'nominal', 'seed_2026091001_full_jepa'),
    (0, 'commitment_contact', 'seed_2026091001_full_jepa'),
    (0, 'commitment_contact', 'seed_2026091001_full_supervised_rollout'),
    (1, 'frozen_reference', 'seed_2026091001_full_jepa'),
    (1, 'frozen_reference', 'seed_2026091001_full_supervised_rollout'),
    (0, 'current_planning', 'seed_2026091001_full_jepa'),
)


def main():
    for layout, mode, model in CASES:
        root = BASE/f'go2_stop_conditioned_independent_{layout:02d}_{mode}_{model}_v1_attempt_001'
        if root.exists() or root.is_symlink():
            raise ValueError(f'preserve existing attempt: {root}')
    owner = psutil.Process()
    receipt = dict(pid=owner.pid, created=owner.create_time(), cases=CASES,
        automatic_retry=False, serial_collection_and_audit=True,
        predecessor_failure_preserved=True)
    with Path('docs/go2_independent_comparisons_continuation_v2_2026-09-13.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt), flush=True)
    for layout, mode, model in CASES:
        script = CONTACT if mode == 'commitment_contact' else STANDARD
        args = [sys.executable, '-B', script, '--layout', str(layout), '--model', model]
        if mode != 'commitment_contact': args += ['--mode', mode]
        print('STARTING', layout, mode, model, flush=True)
        subprocess.run(args, check=True)


if __name__ == '__main__':
    main()
