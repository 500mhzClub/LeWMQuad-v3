"""Descriptive crossed-seed/layout analysis of the fixed prediction study.

No fitting, checkpoint choice, artifact access or significance test. Input
scores must come from the complete authenticated experiment. This is a second
reduction of saved scores, NOT independent reconstruction from raw predictions.
"""
from copy import deepcopy
import math
from statistics import fmean

SEEDS = (2026091101, 2026091102, 2026091103)
ROLES = ('train', 'selection', 'development_eval')
VARIANTS = ('full', 'no_rgb', 'latest_packet_only', 'no_candidate_command')
CONDITIONS = ('direct', 'supervised_rollout', 'jepa')
PREDICTION_METRICS = ('position_error_m', 'yaw_error_rad', 'contact_brier')
HAZARD_METRICS = ('contact_brier', 'minimum_score_contact_fraction',
    'avoidable_contact_regret', 'hazard_concordance')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def primary(variant, condition):
    return variant + '_' + condition + ('_direct_outcomes' if condition == 'direct' else '_rollout_outcomes')


def contrasts():
    """Every fixed contrast; no choosing an auxiliary head or winning seed."""
    rows = []
    def add(question, left, right):
        rows.append(dict(question=question, left=left, right=right))
    for variant in VARIANTS:
        add('added_latent_prediction_objective', primary(variant, 'jepa'), primary(variant, 'supervised_rollout'))
        add('recursive_prediction_package', primary(variant, 'supervised_rollout'), primary(variant, 'direct'))
        add('combined_JEPA_package_not_latent_objective_isolation', primary(variant, 'jepa'), primary(variant, 'direct'))
    for condition in CONDITIONS:
        for baseline in ('action_time', 'zero_motion_empirical_contact'):
            add('full_model_against_simple_control', primary('full', condition), baseline)
        for variant, question in (('no_rgb', 'RGB_information'),
                ('latest_packet_only', 'earlier_packet_information_not_all_memory'),
                ('no_candidate_command', 'prospective_command_information_not_all_action')):
            add(question, primary('full', condition), primary(variant, condition))
    return rows


def _number(value):
    require(value is None or (type(value) in (int, float) and math.isfinite(value)),
        'finite scalar metric or explicit unavailable value required')
    return value


def paired_cells(left, right, *, seeds, layouts, metric, higher_is_better=False):
    """Seeds repeat on the SAME layouts: preserve the crossed matrix, not N=9.

    A complete-population macro is unavailable if any cell is unavailable.
    The observed-subset descriptive mean is separate and never called benefit.
    Raw left-minus-right is retained; negative oriented difference is better.
    """
    require(len(seeds) == len(set(seeds)) and len(layouts) == len(set(layouts))
        and bool(seeds) and bool(layouts), 'unique nonempty crossed units required')
    require(set(left) == set(right) == set(seeds)
        and all(set(left[s]) == set(right[s]) == set(layouts) for s in seeds),
        'exact same seeds and planned layouts required')
    cells = []
    for seed in seeds:
        for layout in layouts:
            a, b = _number(left[seed][layout]), _number(right[seed][layout])
            delta = None if a is None or b is None else _number(a - b)
            cells.append(dict(seed=seed, layout_id=layout, left=a, right=b,
                left_minus_right=delta,
                oriented_difference=None if delta is None else (-delta if higher_is_better else delta)))
    observed = [r['oriented_difference'] for r in cells if r['oriented_difference'] is not None]
    complete = len(observed) == len(cells)
    def reduction(rows):
        values = [r['oriented_difference'] for r in rows if r['oriented_difference'] is not None]
        return dict(planned_cells=len(rows), observed_cells=len(values),
            complete_population_mean=fmean(values) if len(values) == len(rows) else None,
            observed_subset_mean=fmean(values) if values else None)
    direction = ('INCOMPLETE_NO_FULL_POPULATION_DIRECTION' if not complete else
        'ALL_CELLS_LOWER' if all(v < 0 for v in observed) else
        'ALL_CELLS_HIGHER' if all(v > 0 for v in observed) else
        'ALL_CELLS_EXACTLY_TIED' if all(v == 0 for v in observed) else 'MIXED_OR_PARTLY_TIED')
    return dict(metric=metric, raw_direction='left minus right',
        higher_raw_metric_is_better=higher_is_better, negative_oriented_difference_is_better=True,
        cells=cells, complete=complete, **reduction(cells),
        by_seed={str(s): reduction([r for r in cells if r['seed'] == s]) for s in seeds},
        by_layout={l: reduction([r for r in cells if r['layout_id'] == l]) for l in layouts},
        observed_lower_cells=sum(v < 0 for v in observed), observed_higher_cells=sum(v > 0 for v in observed),
        observed_exact_ties=sum(v == 0 for v in observed), descriptive_direction=direction,
        independent_layout_units=len(layouts), repeated_optimization_seeds=len(seeds),
        confidence_interval=None, p_value=None, practical_benefit_established=False)


def _expected_heads():
    return {'action_time', 'zero_motion_empirical_contact'} | {
        v + '_' + c + '_' + h for v in VARIANTS for c in CONDITIONS
        for h in (('direct_outcomes',) if c == 'direct' else ('direct_outcomes', 'rollout_outcomes'))}


def _layout_rows(rows, layouts):
    require(isinstance(rows, list) and len(rows) == len(layouts)
        and [r['layout_id'] for r in rows] == list(layouts),
        'complete ordered planned layout rows required')
    return {r['layout_id']: r for r in rows}


def _scopes(prediction):
    # Every available head must expose the same exact horizon/stratum roster.
    result = None
    for value in prediction['metrics'].values():
        scopes = [('all',)] + [('by_actual_offset_ns', k) for k in sorted(value['by_actual_offset_ns'])]
        scopes += [('strata', field, k) for field in sorted(value['strata']) for k in sorted(value['strata'][field])]
        require(result is None or result == scopes, 'same horizon and stratum population for every head required')
        result = scopes
    require(result is not None, 'at least one scored head required to identify population scopes')
    return result


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def summarize(scores):
    """Consume exact seed -> role -> saved *_scores.json objects.

    Does not trust saved macro or paired-difference fields: recomputes these
    from per-layout values. Artifact, training and raw-scoring authentication
    remain separate requirements. Incomplete heads/layout outcomes stay visible.
    """
    require(isinstance(scores, dict) and set(scores) == set(SEEDS)
        and all(set(scores[s]) == set(ROLES) for s in SEEDS), 'all three seeds and all three roles required')
    expected_primary = {v + '_' + c: primary(v, c) for v in VARIANTS for c in CONDITIONS}
    populations, role_scopes, row_ids, hazard_groups, denominators = {}, {}, {}, {}, {}
    for seed in SEEDS:
        for role in ROLES:
            report = scores[seed][role]; p = report['prediction']; h = report['matched_contact']
            require(report['seed'] == seed and report['primary_heads'] == expected_primary
                and report['no_best_seed_or_checkpoint_selection'] is True,
                'fixed seed and primary-head mapping required')
            require(p['role'] == h['role'] == role and p['resubstitution'] == h['resubstitution'] == (role == 'train'),
                'exact scoring role and resubstitution identity required')
            require(all(p[k] is False and h[k] is False for k in
                ('checkpoint_selection_performed', 'navigation_qualified', 'goal_achieved'))
                and p['final_evaluation'] is False and h['horizon_ns'] == 2_000_000_000,
                'development-only unselected outcomes at the fixed hazard horizon required')
            for part in (p, h):
                require(set(part['metrics']).isdisjoint(part['unavailable_heads'])
                    and set(part['metrics']) | set(part['unavailable_heads']) == _expected_heads(),
                    'complete exact requested head roster, including unavailable heads, required')
            population = p['population']; layouts = sorted(population['layouts'])
            require(population['role'] == role and len(layouts) == (6 if role == 'train' else 3)
                and h['planned_layouts'] == len(layouts), 'fixed six/three/three layout roles required')
            ids = p['scored_row_indices']
            require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
                and ids == sorted(set(ids)), 'unique ordered scored row identities required')
            scopes = _scopes(p)
            if role in populations:
                require(populations[role] == population and row_ids[role] == ids
                    and role_scopes[role] == scopes and hazard_groups[role] == h['groups'],
                    'same exact population, rows, scopes and contact availability across seeds required')
            populations[role] = deepcopy(population); row_ids[role] = list(ids)
            role_scopes[role] = scopes; hazard_groups[role] = deepcopy(h['groups'])
            for head in p['metrics']:
                for scope in scopes:
                    rows = _at(p['metrics'][head], scope)['layouts']
                    _layout_rows(rows, layouts)
                    counts = [{k: r[k] for k in ('layout_id', 'windows', 'motion_count',
                        'contact_count', 'contact_positives')} for r in rows]
                    key = (role, 'prediction', tuple(scope))
                    require(key not in denominators or denominators[key] == counts,
                        'same observed target counts for every method and seed required')
                    denominators[key] = counts
            for head in h['metrics']:
                rows = h['metrics'][head]['layouts']; _layout_rows(rows, layouts)
                counts = [{k: r[k] for k in ('layout_id', 'planned_groups', 'scored_groups',
                    'contrastive_groups')} for r in rows]
                key = (role, 'matched_contact')
                require(key not in denominators or denominators[key] == counts,
                    'same contact group denominators for every method and seed required')
                denominators[key] = counts
    require(sum(len(p['layouts']) for p in populations.values()) ==
        len({l for p in populations.values() for l in p['layouts']}), 'layout leakage between roles forbidden')
    require(sum(len(ids) for ids in row_ids.values()) == len({i for ids in row_ids.values() for i in ids}),
        'scored row leakage between roles forbidden')
    roles = {}
    for role in ROLES:
        layouts = sorted(populations[role]['layouts']); analyses = []
        for section, scopes, metrics in (
                ('prediction', role_scopes[role], PREDICTION_METRICS),
                ('matched_contact', [()], HAZARD_METRICS)):
            for scope in scopes:
                for contrast in contrasts():
                    results = {}
                    for metric in metrics:
                        values = []
                        for side in ('left', 'right'):
                            by_seed = {}
                            for seed in SEEDS:
                                part = scores[seed][role][section]; name = contrast[side]
                                rows = None if name not in part['metrics'] else _layout_rows(
                                    _at(part['metrics'][name], scope)['layouts'], layouts)
                                by_seed[seed] = {l: None if rows is None else rows[l][metric] for l in layouts}
                            values.append(by_seed)
                        results[metric] = paired_cells(*values, seeds=SEEDS, layouts=layouts,
                            metric=metric, higher_is_better=metric == 'hazard_concordance')
                    analyses.append(dict(section=section, scope=list(scope), **contrast, metrics=results))
        roles[role] = dict(population=populations[role], resubstitution=role == 'train',
            scored_row_indices=row_ids[role], analyses=analyses,
            contact_group_availability=hazard_groups[role],
            per_seed_missing_evidence={str(s): dict(
                prediction=deepcopy(scores[s][role]['prediction']['unavailable_heads']),
                matched_contact=deepcopy(scores[s][role]['matched_contact']['unavailable_heads']),
                baseline_missing_cells=deepcopy(scores[s][role]['baseline_missing_cells'])) for s in SEEDS})
    return dict(schema='independent_pulse_scientific_readout.v1', roles=roles,
        contrast_count=len(contrasts()), seeds=list(SEEDS),
        interpretation='Descriptive crossed optimization-seed/layout differences; not independent N=9 on development_eval.',
        missingness='No complete-population macro or direction when any seed/layout metric is unavailable.',
        score_aggregation_only=True, source_artifacts_verified_by_interface=False,
        raw_prediction_reconstruction_performed=False, probability_calibration_established=False,
        predictive_JEPA_benefit_established=False, online_rollout_benefit_established=False,
        navigation_qualified=False, hardware_qualified=False, final_evaluation=False, goal_achieved=False)
