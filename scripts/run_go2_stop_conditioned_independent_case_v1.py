"""Run one explicitly assigned independent development case, then audit it once."""
import argparse
import contextlib
import time

import numpy as np
import psutil
import torch

from scripts import run_go2_stop_conditioned_settling_maze02_v1 as reference
from scripts import stop_conditioned_independent_maze_pipeline_development as pipeline
from scripts.all_phase_fit_execution_development import ROSTER
from scripts.all_phase_planner_model_admission_development import load_assigned

run = reference.run
native = reference.previous
MODEL_LAUNCH_ROOT = run.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
MODEL_LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
SOURCE = 'scripts/run_go2_stop_conditioned_independent_case_v1.py'
PROTOCOL = 'docs/go2_stop_conditioned_independent_comparison_preparation_2026-09-12.md'
SOURCES = (SOURCE, PROTOCOL,
    'lewm/stop_conditioned_comparators_development.py',
    'scripts/stop_conditioned_comparator_pipeline_development.py',
    'scripts/stop_conditioned_independent_maze_pipeline_development.py',
    'lewm/tests/test_stop_conditioned_comparators_development.py',
    'lewm/tests/test_stop_conditioned_independent_maze_pipeline_development.py',
    'lewm/tests/test_stop_conditioned_independent_case_development.py',
    'lewm/independent_round_trip_layouts_development.py',
    'lewm/independent_round_trip_evaluation_development.py',
    'scripts/independent_round_trip_session_development.py')


def assignment(layout, mode, model_name):
    if type(layout) is not int or not 0 <= layout < 8:
        raise ValueError('one of the eight fixed independent layouts required')
    pipeline.comparators.previous.require_mode(mode)
    if mode == 'reactive':
        if model_name is not None:
            raise ValueError('reactive case has no high-level model')
        row = dict(name=None, seed=None, condition=None, variant=None)
    else:
        rows = [r for r in ROSTER if r['name'] == model_name]
        if len(rows) != 1:
            raise ValueError('explicit model from the existing eighteen-model roster required')
        row = dict(rows[0])
    return dict(layout_index=layout, mode=mode, model_name=row['name'],
        training_seed=row['seed'], condition=row['condition'], variant=row['variant'])


def model_identity(model):
    if model is None:
        return None
    if any(m.training for m in model.modules()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged evaluation model without gradients required')
    return native.state_digest(model.state_dict())


def main(layout, mode, model_name):
    if not __debug__:
        raise ValueError('assertions required')
    case = assignment(layout, mode, model_name)
    name = f'independent_{layout:02d}_{mode}_{model_name or "no_model"}'
    output = run.BASE/f'go2_stop_conditioned_{name}_v1_attempt_001'
    run.validate_root(output, must_exist=False)
    if output.exists() or output.is_symlink():
        raise ValueError('preserve existing attempt; no automatic retry or resume')
    baseline = run.read_json(reference.OUTPUT, 'launch.json')
    if run.owner_live(baseline['owner']):
        raise ValueError('finish the current stopping-rule trial before independent execution')
    completed = run.read_json(reference.OUTPUT, 'result.json')
    if completed['status'] != 'STOP_CONDITIONED_SETTLING_MAZE02_V1_COMPLETE':
        raise ValueError('completed stopping-rule trial required; retain negative outcomes')
    # A predecessor manifest records historical development files, including
    # experiments this controller does not execute. Bind this new run to its
    # current sources; retain changes from the predecessor as explicit evidence.
    sources = {p: run.digest(run.ROOT/p) for p in dict.fromkeys(
        (*baseline['source_sha256'], *SOURCES))}
    source_changes = {p: dict(predecessor_sha256=old, launch_sha256=sources[p])
        for p, old in baseline['source_sha256'].items() if sources[p] != old}
    run.verify(sources)
    environment = run.read_json(native.OUTPUT, 'launch.json')
    environment = environment | dict(source_sha256={
        p: sources[p] for p in environment['source_sha256']})
    native.original.require_environment(environment)
    native.original.old.verify_ordered_launch(environment)
    hardware = run.hardware(); reference.guarded.resources.admission(hardware)
    run.cv2.setNumThreads(1); run.cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model = None; treatment = {}; correction_sha = None
    if mode != 'reactive':
        run.verify_artifacts(MODEL_LAUNCH_ROOT, {'launch.json': MODEL_LAUNCH_SHA})
        admission = run.read_json(MODEL_LAUNCH_ROOT, 'launch.json')['input_admission']['correction_admission']
        model, condition, variant = load_assigned(admission, model_name)
        if (condition, variant) != (case['condition'], case['variant']):
            raise ValueError('actual model must match the assigned treatment')
        treatment = dict(model=model, condition=condition, variant=variant)
        correction_sha = admission['correction_result_sha256']
    before = model_identity(model); owner = psutil.Process()
    launch = dict(**case, case=name, source_sha256=sources, hardware=hardware,
        source_changes_from_stopping_trial=source_changes,
        model_state_sha256=before, correction_result_sha256=correction_sha,
        model_admission_launch_sha256=MODEL_LAUNCH_SHA if model is not None else None,
        stopping_trial_result_sha256=run.digest(reference.OUTPUT/'result.json'),
        scene_specification=pipeline.layouts.specification(layout),
        public_mission=pipeline.layouts.public_mission(layout), navigation_ticks=8000,
        owner=dict(pid=owner.pid, created=owner.create_time(), command=owner.cmdline()),
        native_scene_workers=1, physics_paused_during_compute=True,
        automatic_retry=False, independent_layout_development_execution=True,
        real_time_qualified=False, hardware_qualified=False)
    run.create_output(output); native.bounded_json(output/'launch.json', launch)
    print('INDEPENDENT_CASE_LAUNCHED', str(output), run.digest(output/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        with (output/'worker.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            collection = pipeline.collect(layout, sources[PROTOCOL], output=output,
                geometry=native.geometry_factory(native.URDF), episode_name=name, mode=mode, **treatment)
            if model_identity(model) != before:
                raise ValueError('collection changed model weights')
            audit = pipeline.audit(layout, collection, sources[PROTOCOL], input_root=output,
                robot_geometry=native.geometry_factory(native.URDF), episode_name=name, mode=mode, **treatment)
            native.bounded_json(output/(name+'_audit.json'), audit)
            with np.load(output/name/'physics_trace.npz', allow_pickle=False) as raw:
                readout = native.original.case_readout(audit, collection, raw['physics_contact'])
            native.bounded_json(output/(name+'_readout.json'), readout)
            resources = native.results.resource_audit.check(output, name, collection)
            if model_identity(model) != before:
                raise ValueError('audit changed model weights')
            run.verify(sources)
            native.original.require_environment(environment)
            native.original.old.verify_ordered_launch(environment)
        names = [name+'/'+p for p in pipeline.artifacts(layout, collection, mode=mode)]
        names += pipeline.resource_artifacts(name)+['launch.json', 'worker.log', name+'_audit.json', name+'_readout.json']
        native.bounded_json(output/'result.json', dict(status='STOP_CONDITIONED_INDEPENDENT_CASE_COMPLETE',
            assignment=case, collection=collection, readout=readout, resource_audit=resources,
            source_sha256=sources, artifact_sha256={p:run.digest(output/p) for p in names},
            model_state_sha256=before, verified_round_trip=audit['verified_round_trip'],
            independent_layout_development_execution=True, reused_development_layout=False,
            real_time_qualified=False, hardware_qualified=False, goal_achieved=False,
            automatic_retry=False, wall_s=time.perf_counter()-started))
        print('INDEPENDENT_CASE_COMPLETE', run.digest(output/'result.json'), audit['verified_round_trip'], flush=True)
    except BaseException as error:
        native.bounded_json(output/'failure.json', dict(reason=repr(error), automatic_retry=False,
            original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=int, required=True)
    parser.add_argument('--mode', choices=pipeline.comparators.previous.MODES, required=True)
    parser.add_argument('--model')
    args = parser.parse_args()
    main(args.layout, args.mode, args.model)
