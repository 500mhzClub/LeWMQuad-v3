"""Fresh modality-scoped collection; no predecessor reclassification."""
import json
from scripts.independent_layout_batch_development import BATCHES, INVENTORY_IDS, load_inventory, inventory_bindings
from scripts.ordered_dynamic_pilot_development import artifacts as episode_artifacts
from scripts.navigation_artifact_root_development import BASE, validate_root
from scripts.run_go2_successive_choice_maze_development_v1 import digest

PROTOCOL = 'docs/go2_independent_rgb_body_collection_v1_2026-09-06.md'
BATCH_BUDGET = 8 * 1024**3
EPISODE_ALLOWANCE = 256 * 1024**2
RESERVE = 40 * 1024**3


def output_root(batch):
    if batch not in BATCHES:
        raise ValueError('exact declared layout batch required')
    output = BASE / f'go2_independent_rgb_body_collection_v1_{batch}_attempt_001'
    validate_root(output, must_exist=output.exists())
    return output


def validate_launch(launch, inventory, batch):
    ids = list(inventory.episode_ids(batch))
    if launch['batch'] != batch or launch['output_root'] != str(output_root(batch)):
        raise ValueError('fresh batch/output binding required')
    if launch['planned_trials'] != ids or len(ids) != 120:
        raise ValueError('all120 fixed cases required')
    expected = {c: inventory.specification(c) for c in ids}
    if json.dumps(launch['conditions'], sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('unchanged fixed inventory required')
    if (launch['inventory_sha256'] != INVENTORY_IDS['inventory.json']
            or launch['role'] != expected[ids[0]]['data_role']
            or launch['maximum_batch_bytes'] != BATCH_BUDGET
            or launch['episode_storage_allowance_bytes'] != EPISODE_ALLOWANCE
            or launch['minimum_free_bytes'] != RESERVE
            or launch['eligibility_contract'] != 'RGB_BODY_TERMINAL_COVERAGE_V1'
            or launch['model_training'] is not False):
        raise ValueError('fixed role, eligibility and resource contract required')


def commit_episode(output, spec, result):
    names = [spec['trial'] + '/' + p for p in episode_artifacts(spec, result)]
    present = [p for p in names if (output / p).is_file()]
    return dict(trial=spec['trial'], result=result, artifact_sha256={p: digest(output / p) for p in present},
        absent_expected_artifacts=sorted(set(names) - set(present)),
        artifact_bytes=sum((output / p).stat().st_size for p in present))


def eligibility(coverage, window, footprints):
    """Independent prediction-data and depth-navigation claims, after raw audit."""
    if len(footprints) != coverage['paired_frames']:
        raise ValueError('complete per-frame measurement population required')
    hard = [i for i, r in enumerate(footprints) if not r['score']['stable_interior_metric_pass']
            or r['score']['near_occlusion_failure']]
    reasons = []
    if not coverage['candidate_acquisition_complete']: reasons.append(coverage['classification'])
    if window is None or not window['history_ready']: reasons.append('NO_COMPLETE_CAUSAL_DEPARTURE_HISTORY')
    if not footprints: reasons.append('NO_RECORDED_SENSOR_FRAMES')
    if hard: reasons.append('HARD_MEASUREMENT_FAILURE')
    return dict(contract='RGB_BODY_TERMINAL_COVERAGE_V1', rgb_body_prediction_eligible=not reasons,
        exclusion_reasons=reasons, hard_measurement_failed_frames=hard,
        strict_depth_failed_frames=[i for i, r in enumerate(footprints)
            if not r['score']['original_strict_score']['passes_sampled_physical_visibility']],
        boundary_depth_repaired=False, depth_navigation_qualified=False,
        training_performed=False, final_evaluation=False)
