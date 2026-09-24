"""Complete paired descriptive readout after six-fit ledger/raw-score admission."""
import argparse
import json
from scripts.family_transition_model_admission_development import admit
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as INPUT, VARIANTS, CONDITIONS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

OUTPUT = BASE/'go2_family_transition_fit_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_family_transition_fit_readout_v1_2026-09-08.md'
METRICS = ('position_error_m', 'yaw_error_rad', 'contact_brier')


def summarize(scores):
    if set(scores) != {f'{v}_{c}' for v in VARIANTS for c in CONDITIONS}:
        raise ValueError('all six matched primary methods required')
    rows = []; contrasts = []
    for role in ('train', 'geometry_transfer'):
        reference = None
        for method, roles in scores.items():
            r = roles[role]
            population = [(x['scope'], x['cluster'], x['motion_targets'], x['contact_targets'], x['contact_positives']) for x in r['clusters']]
            if reference is None: reference = population
            if population != reference or r['role'] != role:
                raise ValueError('identical complete target populations required for every method')
            for scope in ('all', 'initial', 'moving'):
                cells = [x for x in r['clusters'] if x['scope'] == scope]
                if len(cells) != 2 or len({x['cluster'] for x in cells}) != 2:
                    raise ValueError('both independent parameter clusters must remain explicit')
                rows.append(dict(method=method, role=role, scope=scope, cells=cells,
                    macro={m: sum(x[m] for x in cells)/2 if all(x[m] is not None for x in cells) else None for m in METRICS}))
        pairs = [(f'{v}_jepa', f'{v}_supervised_rollout', 'added_latent_prediction_objective') for v in VARIANTS]
        pairs += [(f'{v}_supervised_rollout', f'{v}_direct', 'recursive_prediction_package') for v in VARIANTS]
        pairs += [(f'full_{c}', f'no_rgb_{c}', 'predictor_RGB_information') for c in CONDITIONS]
        for left, right, question in pairs:
            for scope in ('all', 'initial', 'moving'):
                a = next(r for r in rows if (r['method'], r['role'], r['scope']) == (left, role, scope))
                b = next(r for r in rows if (r['method'], r['role'], r['scope']) == (right, role, scope))
                contrasts.append(dict(role=role, scope=scope, left=left, right=right, question=question,
                    left_minus_right={m: a['macro'][m]-b['macro'][m] if a['macro'][m] is not None and b['macro'][m] is not None else None for m in METRICS}))
    return dict(rows=rows, contrasts=contrasts, optimization_seeds=1,
        parameter_clusters_per_role=2, mirrored_layouts_and_windows_independent=False,
        confidence_interval=None, p_value=None, learned_benefit_established=False,
        navigation_qualified=False, probability_calibration_established=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--study-result-sha256', required=True); args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive scientific readout')
    _, admission = admit(args.study_result_sha256)
    original = read_json(INPUT, 'launch.json'); terminal = read_json(INPUT, 'result.json')
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_family_transition_fits_v1.py',
        'lewm/tests/test_family_transition_fit_readout_development.py'), terminal['source_sha256'])
    definition = original | dict(source_sha256=sources); verify(definition)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, protocol=PROTOCOL, admission=admission,
        study_result_sha256=args.study_result_sha256, native_execution=False, model_training=False))
    try:
        scores = {f'{v}_{c}': {r: read_json(INPUT, f'seed_2026091001_{v}_{c}_{r}_scores.json')
            for r in ('train', 'geometry_transfer')} for v in VARIANTS for c in CONDITIONS}
        report = summarize(scores)
        verify(definition); verify_artifacts(INPUT, {'result.json': args.study_result_sha256, **terminal['artifact_sha256']})
        write_json(OUTPUT/'result.json', dict(status='FAMILY_TRANSITION_PAIRED_READOUT_COMPLETE', **report,
            source_sha256=sources, study_result_sha256=args.study_result_sha256,
            launch_sha256=digest(OUTPUT/'launch.json'), checkpoint_selection_performed=False, goal_achieved=False))
        print('FAMILY_TRANSITION_READOUT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
        print(json.dumps([r for r in report['rows'] if r['scope']=='all'], indent=2), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FAMILY_TRANSITION_READOUT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
