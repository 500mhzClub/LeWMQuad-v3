"""Read-only scientific score aggregation AFTER the fixed 36-fit study.

Requires the actual terminal result SHA. Prints JSON; no artifact output,
training, checkpoint deserialization, raw data loading or experiment launch.
Hash authentication is not an independent audit of training or raw scoring.
"""
import argparse
import json

from lewm.independent_pulse_scientific_readout_development import summarize, SEEDS, ROLES, VARIANTS, CONDITIONS
from scripts.navigation_artifact_root_development import verify_artifacts, artifact_path
from scripts.run_go2_independent_pulse_matched_study_v1 import OUTPUT, definition, identity

DEFINITION_SHA256 = '3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b'


def read_result(result_sha256):
    verify_artifacts(OUTPUT, {'result.json': result_sha256})
    if (OUTPUT / 'failure.json').exists() or (OUTPUT / 'failure.json').is_symlink():
        raise ValueError('terminal study failure cannot supply a complete comparison')
    def read(name):
        path = artifact_path(OUTPUT, name)
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError('bounded explicit study metadata required')
        return json.loads(path.read_text())
    terminal = read('result.json')
    roster = [f'seed_{s}_{v}_{c}' for s in SEEDS for v in VARIANTS for c in CONDITIONS]
    if not (terminal['status'] == 'MATCHED_DEVELOPMENT_COMPARISON_COMPLETE'
            and terminal['completed_fits'] == roster and terminal['fits'] == 36
            and terminal['optimizer_updates'] == 43200 and terminal['seeds'] == list(SEEDS)
            and terminal['objective_conditions'] == list(CONDITIONS) and terminal['input_variants'] == list(VARIANTS)
            and all(terminal[k] is False for k in ('final_evaluation', 'checkpoint_selection_performed',
                'navigation_qualified', 'hardware_qualified', 'goal_achieved'))):
        raise ValueError('complete original unpromoted 36-fit factorial required')
    bindings = terminal['output_sha256']
    names = ['launch.json'] + [f'seed_{s}_{r}_scores.json' for s in SEEDS for r in ROLES]
    if not set(names) <= set(bindings):
        raise ValueError('launch and all nine role score bindings required')
    verify_artifacts(OUTPUT, bindings)
    launch = read('launch.json'); current = definition()
    if not (identity(current) == identity(launch['definition']) == launch['definition_sha256'] == DEFINITION_SHA256):
        raise ValueError('unchanged original frozen experiment definition required')
    scores = {s: {r: read(f'seed_{s}_{r}_scores.json') for r in ROLES} for s in SEEDS}
    report = summarize(scores)
    verify_artifacts(OUTPUT, bindings)
    verify_artifacts(OUTPUT, {'result.json': result_sha256})
    if identity(definition()) != DEFINITION_SHA256:
        raise ValueError('experiment source/config identity changed during score aggregation')
    report['authenticated_study_result_sha256'] = result_sha256
    report['authenticated_study_definition_sha256'] = DEFINITION_SHA256
    report['authenticated_score_sha256'] = {n: bindings[n] for n in names if n != 'launch.json'}
    report['study_output_hashes_verified'] = True
    report['independent_training_or_raw_scoring_audit_performed'] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(read_result(args.result_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
