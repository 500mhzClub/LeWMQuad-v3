"""Whole recorded-episode equivalence and throughput; no new episode population."""
import argparse
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_direct_slot_development import load as load_direct
from scripts import render_go2_navigation_capability_pipeline_development as replay
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def main(source_root,assignment):
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    assert source_root.resolve().is_relative_to((base/'runs').resolve())
    episode=json.loads((source_root/'episode.json').read_text());assert episode['role']=='dev_tune'
    config=json.loads((source_root/'config.json').read_text());arm=config['controller']
    if arm not in ('C1','C2','C3','C4'):raise ValueError('C0 branch replay is not this check')
    root=base/'equivalence'/assignment;root.mkdir(parents=True,exist_ok=False)
    budget=Budget(base,protocol);budget.admit_persist(128*1024**2)
    spec=json.loads((source_root/'specification.json').read_text())
    frames=json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    requests=json.loads((source_root/'requests.json').read_text())
    with np.load(source_root/'native/physics_trace.npz',allow_pickle=False) as a:trace={k:a[k].copy() for k in a.files}
    models=[]
    def actual_model(receipts,controller):
        model=load_direct(protocol) if controller=='C4' else replay.source.load_dense_navigation_model('action',readout_arm='maze_view_maze_data')
        models.append(model);return model
    save(root/'config.json',dict(schema='navigation_capability_equivalence.v1',source=str(source_root),
        source_config_sha256=sha(source_root/'config.json'),controller=arm,assignment=assignment,
        implementation_only=True,new_science_episode=False,physics_device='cpu',model_device='cuda:0' if arm in ('C3','C4') else 'unused neural workload omitted',
        software=dict(torch=str(torch.__version__),hip=str(torch.version.hip)),
        devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())],
        source_bindings={p:sha(p) for p in [__file__,replay.__file__]},
        exact_native_trace_required=True,full_original_source_episode_replayed=True))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    started=time.monotonic()
    try:
        replay.source.previous.warmup();replay.source.previous.study.cohort.stable.floor.configure()
        replay.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        result=bind(replay.verify,RecordedPrediction=actual_model)(source_root,root,budget,spec,episode,trace,requests,frames)
        if not result['exact_native_trace_values']:raise ValueError('throughput optimisation did not preserve exact native trace')
        if models:
            expected=json.loads((source_root/'model_calls.json').read_text());actual=models[0].receipts
            if len(actual)!=len(expected):raise ValueError('different number of prediction calls')
            for a,b in zip(actual,expected,strict=True):
                assert a['observed_ns']==b['observed_ns']
                np.testing.assert_array_equal(a['requested_commands'],b['requested_commands'])
                np.testing.assert_array_equal(a['motion_xy_yaw'],b['motion_xy_yaw'])
            result['prediction_slot']='Recomputed frozen model on original-bitwise sensor inputs'
            result['exact_frozen_prediction_calls']=len(actual)
        result.update(wall_s=time.monotonic()-started,simulated_s=len(requests)*.02,
            peak_process_tree_rss_bytes=budget.peak_rss,peak_device_used_bytes=budget.peak_device_used,
            implementation_only=True,science_episode=False)
        save(root/'result.json',result)
    except BaseException as exc:
        save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,
            optimisation_admitted=False,wall_s=time.monotonic()-started));raise
    finally:replay.source.shutdown_genesis()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--assignment',required=True)
    a=p.parse_args()
    if '/' in a.assignment or a.assignment.startswith('.'):raise ValueError('fresh check assignment required')
    main(a.source_root,a.assignment)
