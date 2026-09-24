"""Eight prospective motion-readout recordings; no fitting or navigation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from lewm import maze_view_transfer_layouts_development as layouts
from lewm.eligible_floor_registration_development import bind
from scripts import collect_go2_maze_view_training_development as training

previous = training.previous
OUTPUT = training.OUTPUT.parent/'go2_maze_view_transfer_v1_attempt_001'
FIT_ROOT = training.OUTPUT.parent/'go2_maze_view_readout_v1_attempt_003'
TRIALS = tuple(f'maze_transfer_{i:02d}' for i in range(layouts.CASE_COUNT))
save, digest = training.save, training.digest


class TransferPhysicalInit(previous.IndependentRoundTripPhysicalInit):
    def __init__(self, spec, *, backend):
        bind(previous.IndependentRoundTripPhysicalInit.__init__,
             specification=layouts.specification, pack=layouts.pack)(self, spec, backend=backend)
        self._runtime['high_level'] = 'fixed evaluation excitation; no navigation policy'


class TransferSession(previous.RGBNavigationRetentionMixin, previous.NogilDrawingMixin,
        previous.LiveDepthNoiseMixin, previous.CompactDepthRetentionMixin,
        previous.LzmaRawDepthPairedCameraSession, TransferPhysicalInit):
    pass


def prepare():
    assert not (FIT_ROOT/'result.json').exists(), 'fix evaluation before fit results'
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free > 2*1024**3 and previous.psutil.virtual_memory().available > 16*1024**3
    specs = list(layouts.specifications())
    for spec in specs:
        assert spec['data_role'] == 'development_transfer'
        definition = layouts.pack(spec)
        np.testing.assert_allclose(definition.robot.spawn_xyz_m[:2],
                                   spec['geometry']['spawn_se2_world'][:2], atol=0, rtol=0)
        yaw = spec['geometry']['spawn_se2_world'][2]
        q = definition.robot.spawn_quat_wxyz
        assert abs(np.arctan2(2*q[0]*q[3], 1-2*q[3]**2)-yaw) < 1e-12
    tape, phases = training.schedule()
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT/'plan.json', dict(trials=TRIALS, specifications=specs,
        inventory=layouts.build_inventory(), tape=tape, phases=phases,
        source_sha256=digest(__file__), dependencies={p: digest(p) for p in
            (layouts.__file__, training.__file__, previous.__file__)},
        data_role='development_transfer', cpu_groups=[[4,5,6,7],[8,9,10,11]],
        minimum_free_bytes=1024**3, free_bytes=free, depth_arrays_retained=False,
        fixed_tape_training_not_navigation=False, closed_loop_navigation=False,
        evaluation_only=True, no_retry_or_outcome_based_selection=True,
        fit_plan_sha256=digest(FIT_ROOT/'plan.json'), fit_result_available=False,
        evaluation=dict(heads=['initial_mixed','old_data_final','maze_data_final'],
            input='actual current and future RGB features; frozen V-JEPA encoder',
            horizons_ms=[500,700], departures=list(range(10,151,10)),
            sample_rule='Fixed departures in every context; retain complete pre-contact windows and report exclusions/stops.',
            metrics=['XY RMSE in mm','yaw RMSE in degrees'],
            reporting='Per maze and horizon; turns/translations/holds separated by fixed tape at departure; aggregate rows descriptive only.',
            model_selection=False, automatic_promotion=False),
        limitations=['two same-family development mazes, not a sealed test',
            'same tape and geometry-only view-selection rule as training',
            'actual-future decoder transfer only; not forecast or navigation success']))
    print('MAZE_TRANSFER_PREPARED', len(specs), flush=True)


def run(case):
    plan = json.loads((OUTPUT/'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256']
    for path, identity in plan['dependencies'].items():
        assert digest(path) == identity
    spec = layouts.specification(case)
    assert spec == plan['specifications'][case]
    directory = OUTPUT/f'case_{case:02d}'
    directory.mkdir(exist_ok=False)
    save(directory/'launch.json', dict(pid=os.getpid(), case=case,
        cpu_affinity=previous.psutil.Process().cpu_affinity()))
    save(directory/'specification.json', spec)
    previous.torch.set_num_threads(4)
    previous.cv2.setNumThreads(1)
    previous.cv2.ocl.setUseOpenCL(False)
    session, stop = None, None
    started = time.monotonic()
    try:
        previous.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        session = TransferSession(spec, directory, noise_layout_index=case%4, noise_sigma_mm=2)
        session.install_contact_identity()
        save(directory/'actuator_identity.json', previous.configure_gains(session.ctx.build.robot,
            session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint'))
        try:
            session.settle_recorded()
            previous.admit_context_setup(session, plan['source_sha256'])
            for tick in range(len(plan['tape'])*5+1):
                if tick%5 == 0:
                    if shutil.disk_usage(OUTPUT).free < plan['minimum_free_bytes']:
                        raise RuntimeError('recording reserve reached')
                    session.sensor_packets()
                if tick == len(plan['tape'])*5:
                    break
                session.phase = 2
                session.command_policy_step(plan['tape'][tick//5])
        except previous.PhysicalStop as error:
            stop = str(error)
        result = dict(status='COMPLETE' if stop is None else 'PHYSICAL_STOP', case=case,
            trial=TRIALS[case], data_role='development_transfer',
            camera_frames=len(session.captured_pairs), physical_stop=stop,
            disallowed_contact=any(bool(s['physics_contact']) for s in session.samples),
            wall_s=time.monotonic()-started, closed_loop_navigation=False, robot_visible=True)
    except BaseException as error:
        save(directory/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if session is not None:
            try:
                session.persist(directory)
                session.persist_observations(directory)
            finally:
                session.ctx.build.scene.destroy()
        previous.shutdown_genesis()
    save(directory/'result.json', result)
    print('MAZE_TRANSFER_CASE_COMPLETE', json.dumps(result), flush=True)


def worker(index):
    for case in range(index, layouts.CASE_COUNT, 2):
        with (OUTPUT/f'case_{case:02d}_process.log').open('x') as log:
            subprocess.run([sys.executable, __file__, '--case', str(case)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('MAZE_TRANSFER_WORKER_CASE', index, case, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', type=int, choices=range(layouts.CASE_COUNT))
    group.add_argument('--worker', type=int, choices=(0,1))
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.case is not None: run(args.case)
    else: worker(args.worker)
