"""Count deployment start-action coverage in the actual dense predictor fit."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts.prepare_go2_short_pulse_training_development import ROOTS
from scripts import train_go2_frozen_vjepa_native_adaptation_development as fit

RESULT = Path('docs/go2_dense_start_action_coverage_2026-09-17.json')


def main():
    assert not RESULT.exists()
    samples_path = fit.OUTPUT / 'samples.json'
    samples = json.loads(samples_path.read_text())
    stats_path = fit.reference.CACHE / 'proprio_v1/proprio_norm_stats.json'
    stats = json.loads(stats_path.read_text())
    raw = np.asarray([r['control'] for r in samples]) * stats['control_std'] + stats['control_mean']
    quiet = np.max(np.abs(raw), axis=(1, 2, 3)) < 1e-6
    actions = np.asarray([r['action'] for r in samples]).reshape(-1, 5, 2)
    moving = np.max(np.abs(actions), axis=(1, 2)) > 1e-6
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    rows = []
    for name in ACTIONS:
        tape = np.asarray(apply_safety_limits_single(
            [candidate_commands(name)[0]] * 5, (0., 0., 0.), limits)[0])[:, [0, 2]]
        matches = quiet & np.all(np.abs(actions - tape) < 1e-6, axis=(1, 2))
        environments = Counter()
        identifiers = []
        for r, match in zip(samples, matches):
            if not match:
                continue
            spec = json.loads((ROOTS[r['source']] / r['trial'] / 'specification.json').read_text())
            assert spec['data_role'] == 'train'
            key = json.dumps(dict(geometry=spec['geometry'], appearance_seed=spec['appearance_seed']), sort_keys=True)
            environments[key] += 1
            identifiers.append(r['sample_id'])
        rows.append(dict(action=name, applied_tape=tape.tolist(), count=int(matches.sum()),
                         environments=[dict(specification=json.loads(k), count=v) for k, v in environments.items()],
                         sample_ids=identifiers))
    report = dict(status='COMPLETE', samples_sha256=fit.digest(samples_path),
                  normalization_sha256=fit.digest(stats_path), total_samples=len(samples),
                  quiet_history_samples=int(quiet.sum()), quiet_to_moving=int((quiet & moving).sum()),
                  quiet_to_hold=int((quiet & ~moving).sum()), actions=rows,
                  interpretation='Exact applied-tape coverage after all-zero past commands; does not establish equal visual/body-state coverage or causality of navigation errors.',
                  training_only=True, new_training=False, new_navigation=False)
    fit.save(RESULT, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'actions'}))
    for r in rows:
        print(r['action'], r['count'], 'environment groups', len(r['environments']))


if __name__ == '__main__':
    main()
