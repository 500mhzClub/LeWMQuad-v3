"""Report both fixed frontier probes alongside the preceding exposed-maze pilot."""
from collections import Counter
import json
import numpy as np

from lewm.contact_score_ablation_development import DISABLED_LOGIT
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path, read, summarize
from scripts.compare_go2_contact_score_ablation_development import behavior
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS


def main():
    output = path('go2_arrival_conditioned_frontier_layout00_summary_v1_attempt_001')
    if output.exists(): raise ValueError('preserve the completed comparison')
    summaries = {}; launches = {}; behaviors = {}; views = {}; hashes = {}
    for mode in ('learned', 'disabled'):
        for revised in (False, True):
            label = ('Closer view' if revised else 'Original') + f' / {mode} contact'
            if revised:
                attempt = '001' if mode == 'learned' else '002'
                name = f'go2_arrival_conditioned_frontier_contact_{mode}_noise_2mm_native_layout00_4800_v1_attempt_{attempt}'
            else:
                name = f'go2_contact_score_ablation_pose_command_xy_{mode}_noise_2mm_native_layout00_4800_v1_attempt_001'
            root = path(name); launch = read(root, 'launch.json')
            if (launch['contact_score_mode'] != mode or launch['forecast_xy_source'] != 'pose_command'
                    or launch['pose_command_fit_sha256'] != FIT_SHA256 or launch['layout_index'] != 0
                    or bool(launch.get('frontier_exclusion_requires_observed_pose_arrival')) != revised):
                raise ValueError('recorded treatment differs from assigned probe')
            plans = [r for r in read(root, 'planning.json') if 'selection' in r]
            for plan in plans:
                c = plan['motion_correction']
                a = np.asarray(c['applied_prediction_after_contact_ablation'])
                u = np.asarray(c['upstream_prediction_for_contact_ablation'])
                if (c['contact_score_mode'] != mode or c['fit_sha256'] != FIT_SHA256
                        or a.shape != (6,8,5) or not np.isfinite(a).all()
                        or not np.array_equal(a[:,:,:4], u[:,:,:4])
                        or not np.array_equal(a[:,:,:2], c['corrected_forecast_xy_m'])
                        or not np.allclose(a[:,:,:2], c['pose_command_forecast_xy_m'], atol=1e-7, rtol=0)):
                    raise ValueError('actual forecast treatment differs')
                expected = u[:,:,4] if mode == 'learned' else np.full((6,8), DISABLED_LOGIT)
                if not np.array_equal(a[:,:,4], expected): raise ValueError('contact channel differs')
                if not np.allclose(np.exp(-np.logaddexp(0., -a[:,6,4])),
                        [r['predicted_contact_by_commit_end'] for r in plan['selection']['candidates']],
                        atol=1e-12, rtol=0): raise ValueError('scored contact differs')
            v = read(root, 'frontier_visits.json'); events = v['events']
            deferred = [e for e in events if e.get('exclusion_deferred_until_arrival')]
            arrived = [e for e in events if revised and not e['exclusion_deferred_until_arrival']]
            for e in events if revised else ():
                distance = np.linalg.norm(np.asarray(e['view_completion_map_xy_m'])-e['target_xy_m'])
                if (abs(distance-e['completion_distance_to_target_m']) > 1e-12
                        or bool(distance > .10) != e['exclusion_deferred_until_arrival']
                        or (distance > .10 and e['excluded_cells'] != 0)):
                    raise ValueError('actual view-retirement receipt differs')
            summaries[label] = summarize(root); launches[label] = launch
            behaviors[label] = behavior(root, plans)
            views[label] = dict(completed_views=len(events), deferred_exclusions=len(deferred),
                views_completed_after_arrival=len(arrived), excluded_cells=len(v['excluded_cells']),
                deferred_completion_distances_m=[e['completion_distance_to_target_m'] for e in deferred],
                arrived_completion_distances_m=[e['completion_distance_to_target_m'] for e in arrived],
                planning_reasons=dict(Counter(r.get('reason','SELECTED') for r in read(root,'planning.json'))),
                selected_route_statuses=dict(Counter(r['route_status'] for r in plans)),
                pipeline_faults=read(root,'pipeline_faults.json'), events=events)
            hashes[label] = launch['source_sha256'] | launch['extra_sources']
    labels = list(launches); reference = launches[labels[0]]
    differences = {label:[k for k in SHARED_FIELDS if launch.get(k)!=reference.get(k)]
        for label,launch in launches.items()}
    if any(differences.values()): raise ValueError(f'shared scientific settings changed: {differences}')
    common = set.intersection(*(set(v) for v in hashes.values()))
    if any(len({v[k] for v in hashes.values()}) != 1 for k in common):
        raise ValueError('shared predecessor runtime source changed')
    new_a, new_b = (hashes[f'Closer view / {mode} contact'] for mode in ('learned','disabled'))
    changed = [k for k in new_a.keys() & new_b.keys() if new_a[k] != new_b[k]]
    if any(not k.endswith('/run_go2_arrival_conditioned_frontier_development.py')
            and k != 'scripts/run_go2_arrival_conditioned_frontier_development.py' for k in changed):
        raise ValueError('unexpected source change between the two new probes')
    report = dict(layout_index=0, conditions=summaries, behavior_metrics=behaviors,
        frontier_views=views, common_predecessor_runtime_sources=len(common),
        new_probe_source_differences=changed,
        source_difference_reason='Only sequential CPU assignment and disabled startup-recovery output name changed.',
        comparison='arrival_conditioned_frontier_probes_with_prior_exposed_layout_reference',
        figure_title='Frontier-view development probes and earlier references — layout 00',
        original_policy_rerun=False, asynchronous_trajectories_not_identical=True,
        disabled_startup_failure_preserved=True, previous_pilot_not_replaced=True,
        statistical_advantage_established=False, host_real_time_qualified=False,
        hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({label:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=s['independent_arrival_evaluation']['disallowed_contact_samples']) for label,s in summaries.items()}))


if __name__ == '__main__': main()
