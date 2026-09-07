"""Account for terminal censoring and compare only shared admitted observations."""
import numpy as np

from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_finite_rgbd_motion_errors_development_v1 import OUTPUT, ARMS, fixed_members
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def analyze():
    launch = read_json(OUTPUT, 'launch.json'); verify(launch)
    result = read_json(OUTPUT, 'result.json')
    if result['status'] != 'FINITE_RGBD_MOTION_ERROR_DIAGNOSTIC_COMPLETE': raise ValueError('completed fixed diagnostic required')
    sources = discover_sources(('scripts/analyze_go2_finite_rgbd_motion_errors_development_v1.py',), launch['source_sha256'])
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    bound |= {str((OUTPUT/n).relative_to(ROOT)): digest(OUTPUT/n) for n in ('launch.json', 'result.json')}
    verify_bindings(sources|bound)
    all_predictions = {}; reports = {}; checked_rows = queries = 0
    expected_names = [c.name for c in fixed_members()]
    for arm in ARMS:
        p = read_json(OUTPUT, arm+'_predictions.json'); e = read_json(OUTPUT, arm+'_evaluation.json')
        assert p['exact_nominal_frames'] == len(p['rows']) == 54
        all_predictions[arm] = p; arm_report = {}
        evaluated = {r['member']: {a['frame']: a for a in r['admitted_rows']} for r in e['details']}
        for name in expected_names:
            failure = None; admitted = []; terminal = ignored = 0; delta = []; differences = []; categorical = []
            footprint_counts = []; gap_errors = []
            for i, row in enumerate(p['rows']):
                assert row['observation_index'] == i and row['measured_ns'] == 1_500_000_000+i*100_000_000
                assert list(row['members']) == expected_names
                m = row['members'][name]; nominal = row['members']['nominal']; checked_rows += 1
                assert m['selects_command'] is False and m['measured_ns'] == row['measured_ns']
                if failure is not None:
                    assert m['status'] == 'NOT_REINVOKED_AFTER_SHADOW_FAILURE' and m['failure'] == failure and m['state'] is None
                    ignored += 1; continue
                if m['status'] == 'TERMINAL_SHADOW_FAILURE':
                    assert m['state'] is None and m['failure']['measured_ns'] == row['measured_ns']
                    failure = m['failure']; terminal += 1; continue
                assert m['status'] == 'SHADOW_OBSERVATION_COMPLETE' and m['state'] is not None
                admitted.append(i); assert i in evaluated[name]
                state = m['state']; assert state['fusion']['usable_under_declared_proxy_budget']
                if nominal['state'] is not None:
                    a = np.asarray(state['fusion']['position_initial_body_m'])
                    b = np.asarray(nominal['state']['fusion']['position_initial_body_m'])
                    delta.append(float(np.linalg.norm(a-b)))
                    differences.append(evaluated[name][i]['position_error_m']-evaluated['nominal'][i]['position_error_m'])
                    def signature(s):
                        motion = s['point_motion']
                        return (s['fusion']['depth_rank'], s['fusion']['kind'], motion['status'],
                            *(motion.get(k) for k in ('lifted_matches','inliers','previous_grid_cells','current_grid_cells')))
                    if signature(state) != signature(nominal['state']): categorical.append(i)
                if 'surface_query' in m:
                    q = m['surface_query']; queries += 1
                    assert not q['navigation_qualified'] and not q['contact_permitted'] and not q['supplied_error_bounds_validated']
                    footprint_counts.append(sum(q['floor_coverage'].values()))
                    gap_errors.extend(evaluated[name][i]['pose_only_reference_gap_error_m'])
            summary = e['members'][name]
            assert (summary['admitted'], summary['terminal_failures'], summary['not_reinvoked']) == (len(admitted), terminal, ignored)
            assert len(admitted)+terminal+ignored == 54 and set(admitted) == set(evaluated[name])
            arm_report[name] = dict(shared_admitted_rows=len(delta),
                maximum_position_response_from_nominal_m=max(delta, default=None),
                mean_position_error_difference_on_shared_rows_m=float(np.mean(differences)) if differences else None,
                categorical_difference_frames=categorical, queried_footprint_counts=footprint_counts,
                maximum_pose_only_reference_gap_error_m=max(gap_errors, default=None))
        reports[arm] = arm_report
    # A matched RGB intervention must preserve failures, not just successful means.
    neutral = all_predictions['neutral']['rows']
    for arm in ARMS:
        rows = all_predictions[arm]['rows']
        for a, b in zip(rows, neutral, strict=True):
            blank = a['members']['blank_rgb']; reference = b['members']['nominal']
            assert blank['status'] == reference['status']
            if blank['state'] is not None: assert blank['state']['fusion'] == reference['state']['fusion']
            else: assert blank['failure'] == reference['failure']
    verify(launch); verify_bindings(sources|bound)
    return dict(status='FINITE_MEMBER_TERMINAL_AND_SHARED_ROW_ACCOUNTING_COMPLETE',
        source_sha256=sources, input_sha256=bound, checked_member_rows=checked_rows,
        checked_surface_queries=queries, comparisons=reports,
        all_blank_rgb_fusion_and_failure_histories_match_neutral_nominal=True,
        exact_correspondence_identity_observable=False, matched_physics_is_one_layout=True,
        finite_population_is_not_continuous_error_bound=True, navigation_qualified=False)


if __name__ == '__main__':
    target = OUTPUT/'shared_row_analysis.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh exclusive accounting only')
    result = analyze(); write_json(target, result); print(result['status'], flush=True)
