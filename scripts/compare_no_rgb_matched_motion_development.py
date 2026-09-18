"""Paired full/no-RGB motion errors on the unchanged validation windows."""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm.seeded_motion_correction_development import registry
from scripts import fit_closed_loop_motion_residual_development as original
from scripts.fit_no_rgb_matched_motion_residual_development import ROOT, SEEDS, METHODS
from scripts.navigation_artifact_root_development import create_output, validate_root


def read(root, name):
    return json.loads((root/name).read_text())


def main():
    full_models = registry()
    rows = []; no_rgb_models = {}; identities = {}
    for seed in SEEDS:
        for method in METHODS:
            full = full_models[f'seed_{seed}_full_{method}']
            roots = dict(full=validate_root(original.BASE/full['root_name']),
                no_rgb=validate_root(original.BASE/ROOT.format(seed=seed, condition=method)))
            reports = {k:read(root, 'result.json') for k,root in roots.items()}
            no_rgb = reports['no_rgb']
            if (any(r['status'] != 'COMPLETE' for r in reports.values())
                    or no_rgb['base_model'] != f'seed_{seed}_no_rgb_{method}'
                    or not no_rgb['original_jepa_windows_targets_and_groups_exact']
                    or not no_rgb['model_rgb_zero_checked_every_forward']):
                raise ValueError('complete matched no-RGB assignments required')
            for variant, root in roots.items():
                fit_hash = hashlib.sha256((root/'residual_fit.npz').read_bytes()).hexdigest()
                expected = full['fit_sha256'] if variant == 'full' else no_rgb['fit_sha256']
                if fit_hash != expected:
                    raise ValueError('frozen fit identity changed')
                identities[root.name] = dict(fit_sha256=fit_hash,
                    result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
            # Targets and groups are compared directly, not inferred from counts.
            for recording in (*original.TRAIN, original.VALIDATION):
                if read(roots['full'], recording+'.json') != read(roots['no_rgb'], recording+'.json'):
                    raise ValueError('full/no-RGB windows or groups differ')
                with np.load(roots['full']/(recording+'.npz'), allow_pickle=False) as a, \
                        np.load(roots['no_rgb']/(recording+'.npz'), allow_pickle=False) as b:
                    if any(not np.array_equal(a[k], b[k]) for k in ('target', 'valid')):
                        raise ValueError('full/no-RGB targets or masks differ')
            for group in reports['full']['results']:
                for horizon, values in reports['full']['results'][group].items():
                    paired = no_rgb['results'][group][horizon]
                    for stage in ('base', 'corrected'):
                        if values[stage]['count'] != paired[stage]['count']:
                            raise ValueError('paired scoring population differs')
                        rows.append(dict(seed=seed, method=method, group=group,
                            horizon_ms=int(horizon), stage=stage, count=values[stage]['count'],
                            full_rmse_mm=values[stage]['rmse_m']*1000,
                            no_rgb_rmse_mm=paired[stage]['rmse_m']*1000,
                            no_rgb_minus_full_rmse_mm=(paired[stage]['rmse_m']-values[stage]['rmse_m'])*1000))
            no_rgb_models[no_rgb['base_model']] = dict(seed=seed, condition=method,
                input_variant='no_rgb', root_name=roots['no_rgb'].name,
                fit_sha256=no_rgb['fit_sha256'], model_state_sha256=no_rgb['model_state_sha256'],
                prediction_head=no_rgb['prediction_head'])
    primary = [r for r in rows if r['group']=='moving' and r['horizon_ms']==700]
    means = []
    for method in METHODS:
        for stage in ('base', 'corrected'):
            cells = [r for r in primary if r['method']==method and r['stage']==stage]
            means.append(dict(method=method, stage=stage, training_seeds=len(cells),
                full_rmse_mm=float(np.mean([r['full_rmse_mm'] for r in cells])),
                no_rgb_rmse_mm=float(np.mean([r['no_rgb_rmse_mm'] for r in cells])),
                no_rgb_lower_error_seed_count=sum(r['no_rgb_minus_full_rmse_mm']<0 for r in cells)))
    output = create_output(original.BASE/'go2_no_rgb_matched_motion_comparison_v1_attempt_001')
    result = dict(rows=rows, moving_700ms_seed_means=means, artifact_identities=identities,
        all_full_no_rgb_windows_targets_and_groups_equal=True, neural_weights_changed=False,
        camera_based_pose_history_retained=True, validation_recordings=1,
        overlapping_windows_are_not_independent_replications=True,
        prospective_navigation_tested=False, rgb_navigation_benefit_established=False)
    with (output/'result.json').open('x') as f:
        json.dump(result, f, indent=2)
    model_registry = Path('docs/go2_no_rgb_navigation_models_2026-09-15.json')
    with model_registry.open('x') as f:
        json.dump(dict(models=no_rgb_models, all_nine_frozen_no_rgb_models=True), f, indent=2)
    print(json.dumps(dict(output=str(output), moving_700ms_seed_means=means,
        registry_sha256=hashlib.sha256(model_registry.read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
