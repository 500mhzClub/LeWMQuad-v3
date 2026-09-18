"""Compare snapshot-rendered pixels with the owning scene at three poses."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import contextlib
import json
import time
import numpy as np

from lewm.actuator_gain_development import configure_gains
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.run_go2_stopping_projection_transfer_development import FreshCameraSession, layouts
from scripts.navigation_artifact_root_development import BASE
from scripts.run_go2_contact_attributed_execution_development_v1 import array
from scripts import snapshot_camera_renderer_development as renderer


def main():
    import cv2
    import torch
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    root=BASE/'go2_snapshot_camera_renderer_check_v1_attempt_002'; root.mkdir()
    spec=layouts.specification(1)
    rows=[]
    with (root/'worker.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
                initializer=renderer.initialize, initargs=(spec,str(root/'renderer'))) as pool:
            witness=pool.submit(renderer.ready).result()
            initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
            native=root/'native'; native.mkdir()
            session=FreshCameraSession(spec,native,noise_layout_index=1,noise_sigma_mm=2)
            session.install_contact_identity()
            configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                session.ctx.policy.env_cfg,'checkpoint')
            session.settle_recorded()
            for index,command in enumerate(([0,0,0],[.2,0,0],[0,0,.45])):
                if index:
                    for _ in range(5):session.command_policy_step(command)
                stamp=int(session.ctx.runner._sim_time_ns)
                start=time.perf_counter_ns(); qpos=array(session.ctx.build.robot.get_qpos()).copy()
                snapshot_ns=time.perf_counter_ns()-start
                reference,transforms=session._render_pair()
                start=time.perf_counter_ns(); result=pool.submit(renderer.render,qpos,stamp).result()
                receipt=dict(index=index,measured_ns=stamp,snapshot_wall_ns=snapshot_ns,
                    round_trip_wall_ns=time.perf_counter_ns()-start,rendering_wall_ns=result['rendering_wall_ns'],
                    maximum_transform_difference=float(np.max(np.abs(np.asarray(transforms)-result['transforms']))),cameras=[])
                for name,(a,d),(b,e) in zip(('primary','auxiliary'),reference,result['images']):
                    receipt['cameras'].append(dict(camera=name,rgb_equal=bool(np.array_equal(a,b)),
                        depth_equal=bool(np.array_equal(d,e)),rgb_different_pixels=int(np.any(a!=b,axis=2).sum()),
                        depth_different_pixels=int((d!=e).sum()),maximum_depth_difference_m=float(np.max(np.abs(d-e)))))
                rows.append(receipt)
            shutdown_genesis()
    report=dict(rows=rows,renderer_ready=witness,physics_advanced_in_renderer=False,
        component_test_only=True,closed_loop_navigation=False)
    with (root/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
