#!/usr/bin/env python3
"""Posthoc numerical/sampling diagnosis, without changing the frozen scan."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'lewm_genesis', ROOT / 'lewm_worlds'):
    sys.path.insert(0, str(path))
from lewm.gyro_integration_diagnostic_development import METHODS, integrate_rates, orientation_errors
from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_active_exit_scan_development_v1 import OUTPUT as SOURCE, verify_bindings, digest, write_json
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding

OUTPUT = ROOT / '.generated/go2_active_scan_orientation_diagnostic_development_v2_attempt_001'
NEW_SOURCES = ('lewm/gyro_integration_diagnostic_development.py',
               'lewm/tests/test_gyro_integration_diagnostic_development.py',
               'scripts/diagnose_go2_active_scan_orientation_development_v1.py',
               'docs/go2_active_scan_orientation_diagnostic_development_v1_2026-09-05.md',
               'scripts/diagnose_go2_active_scan_orientation_development_v2.py',
               'lewm/tests/test_active_scan_orientation_accounting_development.py',
               'docs/go2_active_scan_orientation_diagnostic_development_v2_2026-09-05.md')


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def physical_array_digest(raw):
    bindings = array_binding(raw)
    return hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    if len(sys.argv) != 1 or OUTPUT.exists() or OUTPUT.resolve() != OUTPUT:
        raise ValueError('fixed fresh orientation diagnosis root required')
    launch = json.loads((SOURCE / 'launch.json').read_text())
    result = json.loads((SOURCE / 'result.json').read_text())
    audit = json.loads((SOURCE / 'raw_artifact_audit.json').read_text())
    if (result['status'] != 'COMPLETE' or result['completed_trials'] != 16 or result['planned_trials'] != 16
            or len(result['trials']) != 16 or audit['status'] != 'PASS' or audit['audited_trials'] != 16
            or audit['study_result_sha256'] != digest(SOURCE / 'result.json')
            or result['launch_sha256'] != digest(SOURCE / 'launch.json')):
        raise ValueError('complete source-bound original scan and raw audit required')
    sources = launch['source_sha256'] | {p: digest(ROOT / p) for p in NEW_SOURCES}
    inputs = launch['input_sha256'] | launch['gait_sha256']
    failure = ROOT / '.generated/go2_active_scan_orientation_diagnostic_development_v1_attempt_001'
    failure_bindings = {str((failure / 'launch.json').relative_to(ROOT)): '0bbb0a612f655f4b5386db74596d9d43013fa2f3b375c4a707e556124b0e3437',
                        str((failure / 'result.json').relative_to(ROOT)): '334df1fa095b62a76b4d9bd5bb07274a15e867e185ffc52b065f7843927f0005'}
    verify_bindings(failure_bindings)
    original = json.loads((failure / 'launch.json').read_text())
    verify_bindings(original['source_sha256'] | original['input_sha256'])
    failed = json.loads((failure / 'result.json').read_text())
    if failed['status'] != 'FAIL' or failed['trials'] or failed['error'] != 'TypeError("unhashable type: \'dict\'")':
        raise ValueError('exact empty accounting-failure predecessor required')
    inputs.update(failure_bindings)
    inputs.update({str((SOURCE / name).relative_to(ROOT)): digest(SOURCE / name)
                   for name in ('launch.json', 'result.json', 'raw_artifact_audit.json')})
    for row in result['trials']:
        for name in ('physics_trace.npz', 'ideal_sensor_samples.npz', 'scan_decisions.json'):
            inputs[str((SOURCE / row['scene_id'] / name).relative_to(ROOT))] = row['artifact_sha256'][name]
    verify_bindings(sources | inputs)
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.json', {'source_sha256': sources, 'input_sha256': inputs,
               'rates_hz': [50, 500], 'methods': METHODS, 'scope': 'posthoc diagnostic; 500 Hz is privileged evaluation only'})
    rows, groups = [], {}
    try:
        for member in result['trials']:
            directory = SOURCE / member['scene_id']
            raw = read_npz(directory / 'physics_trace.npz')
            sensors = read_npz(directory / 'ideal_sensor_samples.npz')
            decisions = json.loads((directory / 'scan_decisions.json').read_text())
            groups.setdefault(physical_array_digest(raw), []).append(member['scene_id'])
            if not decisions:
                rows.append({'scene_id': member['scene_id'], 'decisions': 0, 'methods': {}, 'unavailable_reason': 'no_control_decisions'})
                continue
            start = member['prefix_terminal_sample_index']
            sample_indices = np.array([d['pre_sample_index'] for d in decisions])
            decision_ns = np.array([d['decision_ns'] for d in decisions])
            initial = rotation_xyzw(raw['base_pose_world'][start, 3:])
            truth = np.stack([initial.T @ rotation_xyzw(raw['base_pose_world'][i, 3:]) for i in sample_indices])
            live = np.array([d['controller']['rotation_initial_body_from_current_body'] for d in decisions])
            modes = {}
            for rate in (50, 500):
                if rate == 50:
                    times, rates = sensors['measured_ns'], sensors['gyro_values']
                else:
                    times = np.rint(raw['timestamp_s'] * 1e9).astype(np.int64)
                    rotations = np.stack([rotation_xyzw(p[3:]) for p in raw['base_pose_world']])
                    rates = np.einsum('nji,nj->ni', rotations, raw['base_twist_world'][:, 3:])
                selected = (times >= decision_ns[0]) & (times <= decision_ns[-1])
                times, rates = times[selected], rates[selected]
                lookup = np.searchsorted(times, decision_ns)
                if not np.array_equal(times[lookup], decision_ns) or np.any(np.diff(times) != 1_000_000_000 // rate):
                    raise ValueError('exact original decision sample coverage required')
                for method in METHODS:
                    predicted = integrate_rates(times, rates, method)[lookup]
                    replay = float(np.abs(predicted - live).max()) if rate == 50 and method == 'midpoint' else None
                    if replay is not None and replay > 1e-12:
                        raise ValueError('original live midpoint replay changed')
                    modes[f'{rate}:{method}'] = {**orientation_errors(predicted, truth), 'samples': len(times),
                                                'live_replay_max_abs_difference': replay, 'deployment_packet_data': rate == 50}
            rows.append({'scene_id': member['scene_id'], 'decisions': len(decisions), 'methods': modes})
            print(json.dumps({'event': 'scan_orientation_diagnosed', 'completed': len(rows), 'total': 16}), flush=True)
        verify_bindings(sources | inputs)
        write_json(OUTPUT / 'result.json', {'status': 'COMPLETE', 'trials': rows, 'physical_array_identity_groups': groups,
                   'launch_sha256': digest(OUTPUT / 'launch.json'), 'scope': 'reused-scan numerical diagnostic, no revised task result or controller deployment'})
        print(json.dumps({'status': 'COMPLETE', 'trials': 16, 'unique_physical_array_groups': len(groups)}), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'result.json', {'status': 'FAIL', 'error': repr(error), 'trials': rows,
                   'launch_sha256': digest(OUTPUT / 'launch.json')})
        raise


if __name__ == '__main__':
    main()
