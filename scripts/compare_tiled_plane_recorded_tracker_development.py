"""Compare original and tiled trackers on the collected outbound/return history.

Runs only tracking, with no model, map, planner, simulator or command execution.
Stops at the first differing public tracker result; does not discard failures.
"""
from collections import Counter
import json
import time

import cv2
import psutil
import torch

from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorVisualMotion
from lewm.tiled_plane_stop_conditioned_controller_development import TiledPlaneChainedVisualMotion
from scripts import profile_stop_conditioned_early_decisions_development as source

OUTPUT = source.trial.run.BASE/'go2_tiled_plane_tracker_recorded_comparison_v1_attempt_001'
COUNT = 4740  # Includes the current run's return-arrival observation, frame 4739.


def write(name, value):
    with (OUTPUT/name).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False); output.write('\n')


def main():
    hardware = dict(available_ram_bytes=psutil.virtual_memory().available,
        artifact_free_bytes=psutil.disk_usage(source.trial.run.BASE).free,
        cpu_percent=psutil.cpu_percent(interval=.2),
        scheduling='one short-memory tracker pair alongside the existing full audit')
    if hardware['available_ram_bytes'] < 16*1024**3 or hardware['artifact_free_bytes'] < 40*1024**3:
        raise ValueError('retain memory and artifact reserve')
    collection = json.loads((source.INPUT/'result.json').read_text())
    if collection['decisions'] != 4750 or collection['schedule_terminal'] != 'OBSERVED_ROUND_TRIP_CANDIDATE':
        raise ValueError('exact collected stopping trial required')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    OUTPUT.mkdir()
    owner = psutil.Process()
    write('launch.json', dict(owner=dict(pid=owner.pid, created=owner.create_time()),
        hardware=hardware, observations=COUNT, input=str(source.INPUT),
        source_sha256={p:source.digest(source.Path(p)) for p in (
            'scripts/compare_tiled_plane_recorded_tracker_development.py',
            'lewm/tiled_plane_stop_conditioned_controller_development.py')},
        current_trial_launch_sha256=source.digest(source.ROOT/'launch.json'),
        native_execution=False, model_loaded=False, automatic_retry=False))
    print('TRACKER_COMPARISON_STARTED', owner.pid, owner.create_time(), str(OUTPUT), flush=True)
    started_all = time.perf_counter()
    try:
        names = ['policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
            'fast_gyro_histories.npz', 'auxiliary_camera_audit.json']
        bindings = {name:source.digest(source.INPUT/name) for name in names}
        reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
        auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
        arms = [MeasuredPlaneChainedAnchorVisualMotion(), TiledPlaneChainedVisualMotion()]
        totals = [0., 0.]; choices = Counter(); chains = 0
        with (OUTPUT/'frames.jsonl').open('x') as progress:
            for frame in range(COUNT):
                leaves = [f'{stem}_{frame:04d}.{suffix}' for stem, suffix in
                    [('rgb','png'), ('depth','npz'), ('auxiliary_rgb','png'), ('auxiliary_depth','npz')]]
                frame_bindings = {name:source.digest(source.INPUT/name) for name in leaves}
                p, d, f, now = reader.packet(frame)
                image, depth = source.packets.rgb_packet(source.INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                results = [None, None]; durations = [None, None]
                for arm in ((0,1) if frame%2 == 0 else (1,0)):
                    started = time.perf_counter()
                    results[arm] = arms[arm].observe(p,d,f,now_ns=now,
                        auxiliary_depth=depth,auxiliary_rgb=image)
                    durations[arm] = time.perf_counter()-started
                    totals[arm] += durations[arm]
                if json.loads(json.dumps(results[0])) != json.loads(json.dumps(results[1])):
                    write('difference.json',dict(frame=frame,baseline=results[0],candidate=results[1]))
                    raise ValueError(f'tracker results first differ at frame {frame}')
                if frame_bindings != {name:source.digest(source.INPUT/name) for name in leaves}:
                    raise ValueError(f'input files changed at frame {frame}')
                bindings.update(frame_bindings)
                result = results[0]
                if result.get('failure') is not None or result.get('current_pose') is None:
                    write('matched_failure.json',dict(frame=frame,result=result))
                    raise ValueError(f'both trackers unavailable at frame {frame}')
                camera = result['camera_selection'].get('selected_camera')
                choices[str(camera)] += 1
                chains += bool((result.get('chained_anchor_fallback') or {}).get('accepted'))
                progress.write(json.dumps(dict(frame=frame,complete_tracker_result_equal=True,
                    baseline_s=durations[0],candidate_s=durations[1],camera=camera,
                    retained_references=len(arms[0].model.references)))+'\n')
                progress.flush()
                if frame%100 == 0:
                    print('TRACKER_COMPARISON_FRAME',frame,flush=True)
        if any(source.digest(source.INPUT/name) != bindings[name] for name in names):
            raise ValueError('sensor metadata changed')
        write('inputs.json',bindings)
        write('result.json',dict(status='RECORDED_TRACKER_COMPARISON_COMPLETE',
            observations=COUNT,complete_public_tracker_results_equal=True,
            camera_counts=dict(choices),accepted_chained_fallback_frames=chains,
            baseline_tracker_total_s=totals[0],candidate_tracker_total_s=totals[1],
            total_tracker_time_reduction_percent=100*(1-totals[1]/totals[0]),
            wall_s=time.perf_counter()-started_all,shared_host=True,profiler_used=False,
            native_execution=False,model_loaded=False,controller_execution=False,
            map_or_planner_state_compared=False,full_controller_equivalence_proven=False,
            adopted=False,automatic_retry=False))
        print('TRACKER_COMPARISON_COMPLETE', totals, flush=True)
    except BaseException as error:
        write('failure.json',dict(reason=repr(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
