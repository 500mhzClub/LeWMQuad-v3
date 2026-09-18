"""Profile the accepted tracker prefix of the failed live layout-0 recording."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import cv2
import numpy as np
import torch
from lewm.gyro_seeded_height_floor_tracking_development import GyroSeededHeightFloorMotion
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retained-depth-cache',action='store_true')
    args=parser.parse_args()
    root=BASE/'go2_live_gyro_height_floor_noise_2mm_native_layout00_4800_v1_attempt_001'
    output=root/('retained_depth_cache_profile_v1' if args.retained_depth_cache else 'tracking_backlog_profile_v1');output.mkdir()
    recorded=json.loads((root/'poses.json').read_text())
    assert [p['frame'] for p in recorded]==list(range(len(recorded)))
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native')
    if args.retained_depth_cache:
        from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion
        motion=RetainedDepthCacheMotion()
    else:motion=GyroSeededHeightFloorMotion()
    profiler=cProfile.Profile();timings=[];started=time.monotonic()
    keys=('frame','measured_ns','position_initial_body_m','rotation_initial_body_from_current_body',
        'reference_frame','rgb_sha256','depth_sha256')
    for expected in recorded:
        frame=expected['frame'];p,d,fast,rgb,aux,now=reader.packet(frame)
        if frame>=900:profiler.enable()
        begin=time.perf_counter()
        raw=motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=aux,now_ns=now)
        elapsed=time.perf_counter()-begin
        if frame>=900:profiler.disable()
        assert raw['current_pose'] is not None and raw.get('failure') is None,frame
        assert all(raw['current_pose'][k]==expected['raw_pose'][k] for k in keys),frame
        timings.append(dict(frame=frame,tracking_wall_s=elapsed,profiled=frame>=900))
        if frame%250==0:print('PROFILE_PREFIX',frame,flush=True)
    profiler.dump_stats(str(output/'tracking.prof'))
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(40)
    (output/'cumulative.txt').write_text(stream.getvalue())
    report=dict(exact_recorded_raw_pose_witnesses=len(recorded),profiled_frame_start=900,
        profiled_frame_end=len(recorded)-1,elapsed_wall_s=time.monotonic()-started,
        actual_noisy_packet_digests_verified=True,native_physics_used=False,
        registration_or_navigation_executed=False,
        profiling_overhead_included=True,live_timing_reproduction_claimed=False,
        retained_depth_cache=args.retained_depth_cache,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__,'lewm/retained_depth_cache_tracking_development.py')})
    if args.retained_depth_cache:
        cache=motion.model.retained_depth_cache
        report['cache']=dict(entries=len(cache),maximum_entries=cache.maximum_entries,
            hits=cache.hits,misses=cache.misses,
            retained_array_bytes=sum(a.nbytes for inputs,estimated in cache.values()
                for a in (*inputs,*estimated.values())))
    (output/'frames.json').write_text(json.dumps(timings,indent=2))
    (output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True);print(stream.getvalue(),flush=True)


if __name__=='__main__':main()
