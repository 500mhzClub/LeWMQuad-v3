"""Separate higher-camera render of the already-verified C1 pilot replay."""
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from lewm import decision_headroom_json_v42_development as output
from scripts import render_go2_navigation_capability_pipeline_development as video
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def main():
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    source=base/'runs/v0_pilot_C1_dev00_ep0_attempt001'
    previous=base/'videos/pipeline_test_attempt001/replay_verification.json'
    fidelity=json.loads(previous.read_text());assert fidelity['passed'] and fidelity['exact_native_trace_values']
    root=base/'videos/pipeline_test_chase_revision002';root.mkdir(exist_ok=False)
    budget=Budget(base,protocol);budget.admit_persist(256*1024**2)
    configuration=dict(source=str(source),source_config_sha256=sha(source/'config.json'),
        original_replay_verification=dict(path=str(previous),sha256=sha(previous)),
        camera_offset_body_m=[-.6,0.,3.],camera_smoothing_previous_weight=.8,
        simulator_changed=False,controller_changed=False,source_episode_repeated=False,
        reason='Original low chase offset frequently hid the robot behind maze walls',
        implementation_bindings={p:sha(p) for p in [__file__,video.__file__]})
    save(root/'config.json',configuration)
    spec=json.loads((source/'specification.json').read_text());episode=json.loads((source/'episode.json').read_text())
    requests=json.loads((source/'requests.json').read_text())
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as a:trace={k:a[k].copy() for k in a.files}
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    video.source.previous.warmup();video.source.previous.study.cohort.stable.floor.configure()
    started=time.monotonic()
    try:
        video.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        chase=video.chase_pass(source,root,budget,spec,episode,trace,requests)
        (root/'pipeline_test_provisional.mp4').rename(root/'pipeline_test.mp4')
        save(root/'metadata.json',dict(status='PIPELINE_TEST_REPLAY_VERIFIED',capability_video=False,
            source_config_sha256=sha(source/'config.json'),episode_sha256=sha(source/'episode.json'),
            preregistration_sha256=sha(PROTOCOL),harness_sha256=json.loads((source/'config.json').read_text())['harness_sha256'],
            model_binding=protocol['harness_v0']['shared_model_and_sensor_bindings'],
            replay_verification_sha256=sha(previous),chase=chase,
            video_sha256=sha(root/'pipeline_test.mp4'),wall_s=time.monotonic()-started,
            visual_review_pending=True))
    except BaseException as exc:
        save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),publish=False,automatic_retry=False));raise
    finally:video.source.shutdown_genesis()


if __name__=='__main__':main()
