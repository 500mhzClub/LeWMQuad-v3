"""One prospective layout-00 six-action reactive development case, without a model."""
import argparse
import contextlib
import time

import numpy as np
import psutil
import torch

from scripts import run_go2_stop_conditioned_independent_case_v1 as previous
from scripts import six_action_reactive_pipeline_development as pipeline

run = previous.run
native = previous.native
SOURCE = 'scripts/run_go2_six_action_reactive_independent_case_v1.py'
PROTOCOL = 'docs/go2_six_action_reactive_experiment_2026-09-13.md'
SOURCES = previous.SOURCES + (SOURCE, PROTOCOL,
    'lewm/six_action_reactive_controller_development.py',
    'lewm/tests/test_six_action_reactive_controller_development.py',
    'scripts/six_action_reactive_pipeline_development.py')
CASE = 'independent_00_six_action_reactive_no_model'
OUTPUT = run.BASE/f'go2_stop_conditioned_{CASE}_v1_attempt_001'


def main():
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('preserve existing attempt')
    stopping = run.read_json(previous.reference.OUTPUT, 'result.json')
    assert stopping['status'] == 'STOP_CONDITIONED_SETTLING_MAZE02_V1_COMPLETE'
    baseline = run.read_json(previous.reference.OUTPUT, 'launch.json')
    sources = {p:run.digest(run.ROOT/p) for p in dict.fromkeys((*baseline['source_sha256'], *SOURCES))}
    changes = {p:dict(predecessor_sha256=old, launch_sha256=sources[p])
        for p,old in baseline['source_sha256'].items() if sources[p] != old}
    environment = run.read_json(native.OUTPUT, 'launch.json')
    environment = environment | dict(source_sha256={p:sources[p] for p in environment['source_sha256']})
    run.verify(sources); native.original.require_environment(environment)
    native.original.old.verify_ordered_launch(environment)
    hardware = run.hardware(); previous.reference.guarded.resources.admission(hardware)
    run.cv2.setNumThreads(1); run.cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    owner = psutil.Process()
    assignment = previous.assignment(0, 'reactive', None) | dict(mode='six_action_reactive')
    launch = dict(**assignment, case=CASE, source_sha256=sources,
        source_changes_from_stopping_trial=changes, hardware=hardware,
        scene_specification=pipeline.original.layouts.specification(0),
        public_mission=pipeline.original.layouts.public_mission(0), navigation_ticks=8000,
        owner=dict(pid=owner.pid, created=owner.create_time(), command=owner.cmdline()),
        native_scene_workers=1, physics_paused_during_compute=True,
        model_state_sha256=None, high_level_model_loaded=False, automatic_retry=False,
        action_bank_matches_learned_planner=True, fully_nonpredictive_controller=True,
        reactive_is_whole_method_comparison=True, independent_layout_development_execution=True,
        real_time_qualified=False, hardware_qualified=False)
    run.create_output(OUTPUT); native.bounded_json(OUTPUT/'launch.json', launch)
    print('SIX_ACTION_REACTIVE_LAUNCHED', str(OUTPUT), flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'worker.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            collection = pipeline.collect(0, sources[PROTOCOL], output=OUTPUT,
                geometry=native.geometry_factory(native.URDF), episode_name=CASE, mode='reactive')
            audit = pipeline.audit(0, collection, sources[PROTOCOL], input_root=OUTPUT,
                robot_geometry=native.geometry_factory(native.URDF), episode_name=CASE, mode='reactive')
            native.bounded_json(OUTPUT/(CASE+'_audit.json'), audit)
            with np.load(OUTPUT/CASE/'physics_trace.npz', allow_pickle=False) as raw:
                readout = native.original.case_readout(audit, collection, raw['physics_contact'])
            native.bounded_json(OUTPUT/(CASE+'_readout.json'), readout)
            resources = native.results.resource_audit.check(OUTPUT, CASE, collection)
            run.verify(sources); native.original.require_environment(environment)
            native.original.old.verify_ordered_launch(environment)
        names = [CASE+'/'+p for p in pipeline.artifacts(0, collection, mode='reactive')]
        names += pipeline.resource_artifacts(CASE)+['launch.json', 'worker.log', CASE+'_audit.json', CASE+'_readout.json']
        native.bounded_json(OUTPUT/'result.json', dict(status='SIX_ACTION_REACTIVE_INDEPENDENT_CASE_COMPLETE',
            assignment=assignment, collection=collection, readout=readout, resource_audit=resources,
            source_sha256=sources, artifact_sha256={p:run.digest(OUTPUT/p) for p in names},
            high_level_model_loaded=False, verified_round_trip=audit['verified_round_trip'],
            fully_nonpredictive_controller=True, action_bank_matches_learned_planner=True,
            reactive_is_whole_method_comparison=True, automatic_retry=False,
            real_time_qualified=False, hardware_qualified=False, goal_achieved=False,
            wall_s=time.perf_counter()-started))
        print('SIX_ACTION_REACTIVE_COMPLETE', audit['verified_round_trip'], flush=True)
    except BaseException as error:
        native.bounded_json(OUTPUT/'failure.json', dict(reason=repr(error), automatic_retry=False,
            original_evidence_preserved=True)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(); main()
