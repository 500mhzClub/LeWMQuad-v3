#!/usr/bin/env python3
"""Read-only replay of current surface observer on ten audited RGBD packets.

No physics rerun, writes, rescoring of prior metrics or controller intervention.
Printed counts are development diagnostics, not a moving-navigation endpoint.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.depth_local_surfaces_development import LocalSurfaceHistory
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.run_go2_single_sample_rgbd_observation_development_v1 import (
    OUTPUT, preflight, digest, verify_bindings)

EVIDENCE = {
    'launch.json': '088587ce5db2e7aec2e885b6f7bef0f74428ad6a38a540ede6c1114274e909e5',
    'result.json': '0a54ba38f1d98c30c6fb9b49a1dd753c63e50bb65dc841886fe1a77b508a6146',
    'raw_artifact_audit.json': '47d945747f6a0bf8799c5aa6f6b822e3c7e6afa028561e886380c63a3fdfcb55'}


def main():
    if len(sys.argv) != 1: raise ValueError('fixed read-only development replay')
    sources, inputs, gait, native = preflight()
    verify_bindings({str((OUTPUT/p).relative_to(ROOT)): h for p, h in EVIDENCE.items()})
    launch = json.loads((OUTPUT/'launch.json').read_text())
    audit = json.loads((OUTPUT/'raw_artifact_audit.json').read_text())
    if (launch['source_sha256'] != sources or audit['status'] != 'PASS'
            or not audit['all_depth_checks_pass']):
        raise ValueError('completed audited single-sample source identity required')
    results = json.loads((OUTPUT/'result.json').read_text())
    rows = []
    for trial in results['trials']:
        directory = OUTPUT/trial['scene_id']
        verify_bindings({str((directory/p).relative_to(ROOT)): h for p, h in trial['artifact_sha256'].items()})
        history = LocalSurfaceHistory()
        for index in range(5):
            policy, depth = load_rgbd_observation(directory, index)
            observed = history.observe(depth, policy, now_ns=policy['sensor_state']['decision_ns'])
            row = {'scene_id': trial['scene_id'], 'observation_index': index,
                'valid_columns': sum(observed['valid_columns']),
                'unknown_column_runs': observed['unknown_column_runs'],
                'unmodelled_valid_columns': observed['unmodelled_valid_columns'],
                'surface_segments': observed['surface_segments'],
                'depth_discontinuities': observed['depth_discontinuities'],
                'translation_constraint_geometry': observed['translation_constraint_geometry']}
            rows.append(row)
        if len(history.snapshot()) != 4: raise AssertionError('four causal observations retained')
    print(json.dumps({'status': 'REPLAY_COMPLETE', 'packets': len(rows),
        'prior_bindings': [len(sources), len(inputs), len(gait), len(native)],
        'observer_sha256': digest(ROOT/'lewm/depth_local_surfaces_development.py'),
        'checker_sha256': digest(Path(__file__)), 'rows': rows,
        'scope': 'stationary sensor-only local surface replay; no arrival, motion, clearance or navigation success'},
        allow_nan=False))


if __name__ == '__main__': main()
