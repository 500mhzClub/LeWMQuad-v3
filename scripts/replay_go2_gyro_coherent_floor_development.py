"""Compare the paired-floor estimator on retained delivered sensor recordings."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from lewm.gyro_coherent_floor_tracking_development import GyroCoherentFloorMotion
from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion
from lewm.coherent_reference_refresh_development import CoherentReferenceRefreshMotion
from lewm.local_view_revisit_tracking_development import LocalViewRevisitMotion
from lewm.batched_consensus_tracking_development import BatchedConsensusMotion
from lewm.cached_pair_floor_tracking_development import CachedPairFloorMotion
from lewm.cached_floor_moments_development import CachedFloorMomentsMotion
from lewm.deferred_registration_copy_development import DeferredRegistrationCopyMotion
from lewm.cached_moments_deferred_copy_tracking_development import CachedMomentsDeferredCopyMotion
from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration
from lewm.partial_floor_height_development import read_pose
from lewm.physical_execution_development import rotation_xyzw
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.diagnose_go2_depth_noise_failures_development import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',required=True)
    parser.add_argument('--variant',choices=('baseline','coherent','coherent_refresh','local_view_revisit','batched_consensus','cached_pair_floor','deferred_registration_copy','cached_floor_moments','cached_moments_deferred_copy'),required=True)
    parser.add_argument('--profile-from-frame',type=int)
    parser.add_argument('--run-label', help='distinct ordinary label for a matched replay repetition')
    args=parser.parse_args();root=path(args.root_name)
    if (root/'depth_retention.json').exists():
        raise ValueError('retired recordings cannot support this replay')
    suffix='replay_v1' if args.profile_from_frame is None else f'profile_{args.profile_from_frame}_32_v1'
    if args.run_label is not None:
        if not args.run_label.isidentifier() or args.run_label.startswith('sealed'):
            raise ValueError('ordinary identifier required for replay label')
        suffix += '_' + args.run_label
    output=root/f'gyro_coherent_floor_{args.variant}_{suffix}';output.mkdir()
    recorded={r['frame']:r['raw_pose'] for r in read(root,'poses.json')}
    count=len(read(root,'native/in_memory_camera_observations.json')['frames'])
    if args.profile_from_frame is not None and not 0<=args.profile_from_frame<=count-32:
        raise ValueError('complete 32-frame profiling window required')
    profiler=cProfile.Profile() if args.profile_from_frame is not None else None
    sources=[__file__,'lewm/gyro_coherent_floor_constraint_development.py',
        'lewm/gyro_coherent_floor_tracking_development.py',
        'lewm/gyro_consensus_pair_pose_development.py','lewm/gyro_conditioned_pair_pose_development.py']
    if args.variant in ('coherent_refresh','local_view_revisit','batched_consensus','cached_pair_floor','deferred_registration_copy','cached_floor_moments','cached_moments_deferred_copy'):
        sources += ['lewm/coherent_reference_refresh_development.py',
            'lewm/recent_anchored_reference_refresh_development.py']
    if args.variant in ('local_view_revisit','batched_consensus','cached_pair_floor','deferred_registration_copy','cached_floor_moments','cached_moments_deferred_copy'):
        sources += ['lewm/local_view_revisit_tracking_development.py']
    if args.variant in ('batched_consensus','cached_pair_floor','deferred_registration_copy','cached_floor_moments','cached_moments_deferred_copy'):
        sources += ['lewm/batched_gyro_consensus_development.py',
            'lewm/batched_consensus_tracking_development.py']
    if args.variant == 'cached_pair_floor':
        sources += ['lewm/cached_pair_floor_tracking_development.py']
    if args.variant == 'deferred_registration_copy':
        sources += ['lewm/deferred_registration_copy_development.py']
    if args.variant == 'cached_floor_moments':
        sources += ['lewm/cached_floor_moments_development.py']
    if args.variant == 'cached_moments_deferred_copy':
        sources += ['lewm/cached_floor_moments_development.py',
            'lewm/deferred_registration_copy_development.py',
            'lewm/cached_moments_deferred_copy_tracking_development.py']
    save(output,'launch.json',dict(root_name=root.name,variant=args.variant,frames=count,
        run_label=args.run_label,
        profile_from_frame=args.profile_from_frame,profile_frames=32 if profiler is not None else 0,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        estimator_only=True,native_execution=False,original_noise_recipe=True))
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native')
    motion={'baseline': RetainedDepthCacheMotion, 'coherent': GyroCoherentFloorMotion,
        'coherent_refresh': CoherentReferenceRefreshMotion,
        'local_view_revisit': LocalViewRevisitMotion,
        'batched_consensus': BatchedConsensusMotion,
        'cached_pair_floor': CachedPairFloorMotion,
        'deferred_registration_copy': DeferredRegistrationCopyMotion,
        'cached_floor_moments': CachedFloorMomentsMotion,
        'cached_moments_deferred_copy': CachedMomentsDeferredCopyMotion}[args.variant]()
    registration=RobustHeightFloorRegistration();rows=[];failure=None;matched=0;started=time.monotonic()
    fields=('frame','measured_ns','reference_frame','position_initial_body_m',
        'rotation_initial_body_from_current_body','rgb_sha256','depth_sha256')
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(count):
            try:
                p,d,fast,rgb,a,now=reader.packet(frame)
                motion_started=time.perf_counter_ns()
                profiled=profiler is not None and args.profile_from_frame<=frame<args.profile_from_frame+32
                if profiled:
                    raw=profiler.runcall(motion.observe,p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=a,now_ns=now)
                else:
                    raw=motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=a,now_ns=now)
                motion_wall_ns=time.perf_counter_ns()-motion_started
                if raw['current_pose'] is None:
                    save(output,'terminal_raw_snapshot.json',raw)
                    raise ValueError('raw tracking failed')
                if frame in recorded:
                    identical=all(raw['current_pose'][k]==recorded[frame][k] for k in fields)
                    matched+=identical
                    if args.variant=='baseline' and not identical:
                        raise ValueError('baseline raw pose differs from recorded original')
                registration_started=time.perf_counter_ns()
                evidence=registration.observe(p,d,a,raw,now_ns=now)
                registration_wall_ns=time.perf_counter_ns()-registration_started
                read_pose(evidence,identity=(0,0,0),now_ns=now)
                row=dict(frame=frame,raw_pose=raw['current_pose'],
                    call_profile_enabled=profiled,
                    motion_wall_ns=motion_wall_ns,registration_wall_ns=registration_wall_ns,
                    registered_position_m=evidence['current_pose']['position_initial_body_m'],
                    pair_constraint=getattr(motion.model,'last_coherent_floor_constraint',None),
                    local_view_revisit_attempt=getattr(motion.model,'last_revisit_attempt',None))
                if args.variant == 'cached_pair_floor':
                    row['pair_fit_cache'] = motion.model.last_pair_fit_cache
                if args.variant in ('cached_floor_moments','cached_moments_deferred_copy'):
                    cache = motion.model.floor_moments
                    row['cloud_moments_cache'] = dict(hits=cache.hits, misses=cache.misses,
                        pruned_reductions=cache.pruned_reductions, entries=len(cache.entries))
                rows.append(row);sink.write(json.dumps(row)+'\n');sink.flush()
            except Exception as error:
                failure=dict(frame=frame,reason=repr(error));break
            if frame%300==0:print('COHERENT_FLOOR_REPLAY_FRAME',frame,flush=True)
    if profiler is not None:profiler.dump_stats(str(output/'motion_calls.prof'))
    frames=read(root,'native/in_memory_camera_observations.json')['frames']
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world']
    origin=physics[frames[0]['physical_sample_index']];R0=rotation_xyzw(origin[3:])
    errors=[float(np.linalg.norm((physics[frames[r['frame']]['physical_sample_index'],:3]-origin[:3])@R0
        -r['registered_position_m'])) for r in rows]
    result=dict(variant=args.variant,recorded_frames=count,accepted_frames=len(rows),failure=failure,
        local_view_revisit_attempts=sum(r.get('local_view_revisit_attempt') is not None for r in rows),
        local_view_revisit_selections=sum(bool((r.get('local_view_revisit_attempt') or {}).get('selected')) for r in rows),
        reference_refresh_frames=[r['frame'] for r in rows if
            r['raw_pose'].get('promotion_reason') == 'accepted_anchor_recent_reference_age'],
        matching_recorded_raw_poses=matched,recorded_raw_poses=len(recorded),
        registered_position_error_m=None if not errors else dict(median=float(np.median(errors)),
            maximum=max(errors),final=errors[-1]),elapsed_s=time.monotonic()-started,
        replay_call_wall_time_ms={stage:dict(median=float(np.median([r[stage]/1e6 for r in rows])),
            p95=float(np.percentile([r[stage]/1e6 for r in rows],95)),
            maximum=max(r[stage]/1e6 for r in rows))
            for stage in ('motion_wall_ns','registration_wall_ns')} if rows else None,
        replay_call_timing_excludes_packet_io=True,
        profiled_frames=[r['frame'] for r in rows if r['call_profile_enabled']],
        profiled_call_times_include_profiler_overhead=profiler is not None,
        native_state_loaded_only_after_estimation=True,closed_loop_navigation_test=False)
    save(output,'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
