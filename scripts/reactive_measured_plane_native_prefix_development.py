"""Require actual execution of the already observed first reactive difference."""
from types import SimpleNamespace

from scripts import replay_go2_measured_plane_reactive_prefix_v1 as replay
from scripts import nominal_measured_plane_native_prefix_development as original
from scripts.extended_budget_anchored_maze_development import _bind

run = replay.run
FRAMES, INTERVENTION, PHYSICS_SAMPLES = original.FRAMES, original.INTERVENTION, original.PHYSICS_SAMPLES
LAUNCH_SHA = 'ed9e51645c2e4176c46c4f330b1a7513b1b1e22afc8408114d8ac1ffc8b8cc5a'
RESULT_SHA = '87defc11bf27970855e5b2c1ce17f6839ffed6d8641fa1cc998e2afedab9176b'


def boundary(report):
    if (report['frames'] != FRAMES or report['learned_forecasts'] != 1
            or report['actual_model_forward_calls'] != [1, 0]
            or report['model_state_unchanged'] is not True
            or report['model_state_sha256'] != replay.job.MODEL_SHA
            or report['reactive_model_instantiated'] is not False
            or report['reactive_residual_instantiated'] is not False
            or report['fully_nonpredictive_arm'] is not True
            or report['predictive_feasibility_gates_matched'] is not False
            or report['isolated_prediction_ranking_ablation'] is not False
            or report['native_execution'] is not False
            or report['retrospective_navigation_outcomes_inferred'] is not False):
        raise ValueError('exact completed four-frame reactive comparison required')
    old, new = report['boundary_baseline'], report['boundary_candidate']
    expected = replay.compare(old, new, old, frame=INTERVENTION)
    if (report['boundary_comparison'] != expected or expected['requested_command_changed'] is not True
            or expected['terminal_boundary'] is not False):
        raise ValueError('exact nonterminal first reactive intervention required')
    for decision, action, requested in ((old, 'left_arc', [.16, 0., .45]), (new, 'forward', [.2, 0., 0.])):
        if (decision['terminal'] is not None or decision['failure'] is not None
                or decision['new_selection']['action'] != action or decision['requested_command'] != requested):
            raise ValueError('complete observed left-arc versus reactive forward intervention required')


def learned_reference(decision):
    if (decision['controller'] != 'measured_plane_residual_continuation_controller_v1'
            or decision['measured_plane_constrained_estimator'] is not True):
        raise ValueError('original complete learned decision without normalization required')
    return decision


executed_boundary = _bind(original.executed_boundary, boundary=boundary)
packets = original.packets
# Retain all original raw physics arrays, command endpoints, complete public
# packets and full decisions. The reactive replay saved the original learned
# decision directly, so its baseline undergoes no source-label normalization.
_compare = _bind(original.compare, boundary=boundary, executed_boundary=executed_boundary,
    replay=SimpleNamespace(OUTPUT=replay.OUTPUT, learned_reference=learned_reference))


def compare(prior, current, report):
    result = _compare(prior, current, report)
    if result.pop('both_arms_predictive') is not True:
        raise ValueError('original shared physics comparison receipt changed')
    return result | dict(both_arms_predictive=False, fully_nonpredictive_candidate=True,
        reactive_is_whole_method_comparison=True, isolated_prediction_ranking_ablation=False,
        future_constraint_gates_matched=False)


def completed_prefix():
    root = replay.OUTPUT
    run.verify_artifacts(root, {'launch.json': LAUNCH_SHA, 'result.json': RESULT_SHA})
    launch = run.read_json(root, 'launch.json'); result = run.read_json(root, 'result.json')
    if ((root/'failure.json').exists() or (root/'failure.json').is_symlink()
            or run.owner_live(launch['owner'])
            or launch['boot_id'] != run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or result['status'] != 'MEASURED_PLANE_REACTIVE_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or result['original_raw_inputs_reauthenticated_before_and_after'] is not True):
        raise ValueError('exact ended completed reactive prefix required')
    boundary(result['report']); run.verify(result['source_sha256'])
    run.verify_artifacts(root, result['artifact_sha256'])
    if run.read_json(root, 'report.json') != result['report']:
        raise ValueError('same saved complete reactive report required')
    replay.check_output(result['report'])
    return result
