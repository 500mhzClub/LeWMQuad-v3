"""Fresh development trial of the stop-conditioned arrival boundary.

Reuse the physical collector and its complete raw audit once. Do not repeat
the predecessor's prefix reconstruction or infer a new trajectory from it.
"""
import contextlib
import json
import time

import numpy as np
import psutil
import torch

from lewm.stop_conditioned_settling_development import StopConditionedSettlingController
from scripts import run_go2_extended_return_budget_maze02_v1 as previous
from scripts import resource_guarded_extended_return_maze_development as guarded

run = previous.run
base = guarded.pipeline
OUTPUT = run.BASE/'go2_stop_conditioned_settling_maze02_v1_attempt_001'
CASE = 'no_rgb_direct_stop_conditioned_settling_maze_02'
SOURCE = 'scripts/run_go2_stop_conditioned_settling_maze02_v1.py'
PROTOCOL = 'docs/go2_extended_return_settling_diagnosis_2026-09-12.md'
NEW_SOURCES = (SOURCE, PROTOCOL, 'lewm/stop_conditioned_settling_development.py',
    'lewm/tests/test_stop_conditioned_settling_development.py',
    'lewm/tests/test_stop_conditioned_settling_pipeline_development.py',
    'docs/go2_extended_return_post_audit_comparison_interruption_2026-09-12.json')
PREDECESSOR_AUDIT = {
    previous.CASE[0]+'_audit.json': '043154f80da4f3bd02bbe9828ddabc4c7f1a108380b64d788df76dc0706948c1',
    previous.CASE[0]+'_readout.json': '95ffaa095fe5d5062dd7404c21dcc744a961df709d58c1ec8b931720104b70fd',
}


def controller_factory(guard):
    def create(*args, **kwargs):
        return guarded.CheckedController(StopConditionedSettlingController(*args, **kwargs), guard)
    return create


execute = base.bind(guarded._execute, controller_factory=controller_factory)


def collect(*args, **kwargs):
    return execute(base.collect, args, kwargs, root=kwargs['output'],
        episode=kwargs['episode_name'], phase='collection')


def audit(*args, **kwargs):
    return execute(base.audit, args, kwargs, root=kwargs['input_root'],
        episode=kwargs['episode_name'], phase='audit')


def main():
    if not __debug__:
        raise ValueError('assertions required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('preserve the existing attempt; no automatic retry or resume')
    predecessor = run.read_json(previous.OUTPUT, 'launch.json')
    if run.owner_live(predecessor['owner']):
        raise ValueError('existing experiment owner must end before fresh execution')
    # The complete scientific audit is available. Its later paired-prefix
    # comparison was deliberately interrupted; do not call that attempt complete.
    run.verify_artifacts(previous.OUTPUT, PREDECESSOR_AUDIT)
    sources = predecessor['source_sha256'] | {p: run.digest(run.ROOT/p) for p in NEW_SOURCES}
    run.verify(sources)
    previous.original.require_environment(predecessor)
    previous.original.old.verify_ordered_launch(predecessor)
    hardware = run.hardware()
    guarded.resources.admission(hardware)
    run.cv2.setNumThreads(1)
    run.cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = previous.assigned_model()
    before = previous.state_digest(model.state_dict())
    owner = psutil.Process()
    launch = dict(case=CASE, layout_index=2, condition='direct', variant='no_rgb',
        navigation_ticks=8000, controller='stop_conditioned_settling_controller_v1',
        predecessor_raw_audit_sha256=PREDECESSOR_AUDIT,
        predecessor_final_comparison_completed=False,
        source_sha256=sources, model_state_sha256=before,
        input_sha256=predecessor['input_sha256'], native_sha256=predecessor['native_sha256'],
        native_scene_sha256=predecessor['native_scene_sha256'],
        native_geometry_sha256=predecessor['native_geometry_sha256'],
        renderer_environment=predecessor['renderer_environment'], hardware=hardware,
        owner=dict(pid=owner.pid, created=owner.create_time(), command=owner.cmdline()),
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        native_scene_workers=1, physics_paused_during_compute=True,
        scheduling_reason='one fresh case; retain matched CPU numerics and avoid competing native/audit jobs',
        zero_request_boundary_required_before_dwell=True, automatic_retry=False,
        independent_layout_development_execution=False, real_time_qualified=False,
        hardware_qualified=False, repeated_predecessor_prefix_reconstruction=False)
    run.create_output(OUTPUT)
    previous.bounded_json(OUTPUT/'launch.json', launch)
    print('STOP_CONDITIONED_SETTLING_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        with (OUTPUT/'worker.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            definition = sources[PROTOCOL]
            result = collect(2, definition, output=OUTPUT, model=model,
                geometry=previous.geometry_factory(previous.URDF), episode_name=CASE,
                condition='direct', variant='no_rgb')
            if previous.state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
                raise ValueError('collection changed the assigned model')
            raw_audit = audit(2, result, definition, input_root=OUTPUT, model=model,
                robot_geometry=previous.geometry_factory(previous.URDF), episode_name=CASE,
                condition='direct', variant='no_rgb')
            previous.bounded_json(OUTPUT/(CASE+'_audit.json'), raw_audit)
            with np.load(OUTPUT/CASE/'physics_trace.npz', allow_pickle=False) as raw:
                readout = previous.original.case_readout(raw_audit, result, raw['physics_contact'])
            previous.bounded_json(OUTPUT/(CASE+'_readout.json'), readout)
            resources = previous.results.resource_audit.check(OUTPUT, CASE, result)
            run.verify(sources)
            previous.original.require_environment(predecessor)
            previous.original.old.verify_ordered_launch(predecessor)
            if previous.state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
                raise ValueError('audit changed the assigned model')
        names = [CASE+'/'+n for n in base.artifacts(2, result)]
        names += guarded.resource_artifacts(CASE)
        names += ['launch.json', 'worker.log', CASE+'_audit.json', CASE+'_readout.json']
        identities = {n: run.digest(OUTPUT/n) for n in names}
        previous.bounded_json(OUTPUT/'result.json', dict(
            status='STOP_CONDITIONED_SETTLING_MAZE02_V1_COMPLETE',
            collection=result, readout=readout, resource_audit=resources,
            source_sha256=sources, artifact_sha256=identities, model_state_sha256=before,
            model_state_unchanged=True, verified_round_trip=raw_audit['verified_round_trip'],
            reused_layout_executions=1, new_independent_layout_executions=0,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False,
            goal_achieved=False, automatic_retry=False, wall_s=time.perf_counter()-started))
        print('STOP_CONDITIONED_SETTLING_COMPLETE', run.digest(OUTPUT/'result.json'),
            raw_audit['verified_round_trip'], flush=True)
    except BaseException as error:
        previous.bounded_json(OUTPUT/'failure.json', dict(status='STOP_CONDITIONED_SETTLING_FAILED',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()
