"""Isolated decision latency (Andrew Knowles, 28 September 2026).

In-run latencies were measured under concurrent missions. Here a fixed sample of
logged decisions (the first 100 decisions of validation episodes 10/0, 11/0 and
12/0, fixed before measurement) is replayed deterministically through each
controller's unchanged planning step with its real prediction-slot model, one
controller at a time with nothing else running. Decisions and dispatch are
verified against the log, so the timed inputs are the logged ones. Latency is
the harness's own planning-stage wall time with physics paused.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import partial
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from lewm.navigation_capability_paired_floor_start_development import initialize_mapping
from lewm.navigation_capability_target_reference_development import install_task_cues
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

EPISODES = (10, 11, 12)
DECISIONS = 100
WARMUP = 3


def replay(base, protocol, arm, maze):
    source = owner.source
    root = base/f'runs/v4_completed_support_validation_{arm}_val{maze:02d}_ep0_attempt001'
    spec = json.loads((root/'specification.json').read_text())
    requests = json.loads((root/'requests.json').read_text())
    frames = json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames']
    plans = {r['frame']: r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r}
    model = owner.load_model(arm, protocol, base)
    budget = owner.Budget(base, protocol)
    clock = source.UntimedSimulationClock()
    controller = session = None
    work = base/'analysis/qualification_v4_2026-09-28/latency_replays'/f'{arm}_val{maze:02d}'
    work.mkdir(parents=True)
    (work/'native').mkdir()
    decided = 0
    try:
        source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'), initializer=initializer))
            registration = pool(source.previous.study.previous.reference.previous.initialize_registration)
            mapping = pool(initialize_mapping)
            pose = pool(partial(source.initialize_pose, str(work)))
            obstacles = pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result()
            assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            session = owner.make_session(spec, work/'native', full_frames=False)
            session.install_contact_identity()
            source.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
            session.settle_recorded()
            source.admit_context_setup(session, owner.sha(owner.__file__))
            cues = json.loads((root/'public_task_cues.json').read_text())
            base_runtime = source.DenseReactiveNavigationRuntime if arm == 'C2' else source.DenseNavigationRuntime
            runtime = type('StartupRecoveredRuntime', (CompletedSupportRuntimeMixin, base_runtime), {})
            controller = runtime(model, goal_initial_xy=cues['goal_initial_body_xy_m'], condition='jepa', variant='full', clock_ns=clock,
                evidence_sink=lambda *a: None, planning_delay_ticks=3, maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm == 'C1' else 'neural', registration_executor=registration,
                navigation_ticks=4800, arrival_radius_m=.02, mapping_executor=mapping, pose_executor=pose, obstacle_executor=obstacles)
            install_task_cues(controller, cues)
            session.physics_clock_callback = clock.advance
            history = deque(maxlen=4)
            for tick, expected in enumerate(requests):
                budget.check()
                now = int(session.ctx.runner._sim_time_ns)
                clock.advance(now)
                assert now == expected['simulator_ns']
                if tick % 5 == 0:
                    policy, depth, fast, aux_depth, aux_rgb, measured = session.sensor_packets()
                    history.append(policy)
                    record = session.captured_pairs[-1]['consumed_hash_record']
                    assert record['pixel_sha256'] == frames[tick//5]['pixel_sha256'], ('consumed frame differs', tick//5)
                    acquired = source.AcquiredFrame(tick//5, measured, policy, depth, fast, aux_rgb, aux_depth, tuple(history))
                    controller.submit(acquired)
                    owner.drain(controller, model, session, budget)
                    plan = plans.get(acquired.frame)
                    actual = next((r for r in reversed(controller.planning) if r['frame'] == acquired.frame and 'selection' in r), None)
                    assert (plan is None) == (actual is None) and (plan is None or plan['selection']['action'] == actual['selection']['action']), 'decision differs'
                    decided += plan is not None
                    if decided >= DECISIONS:
                        break
                request = controller.request(now_ns=clock())
                assert request['requested_command'] == expected['requested_command'] and request['reason'] == expected['reason'], 'dispatch differs'
                session.phase = 2
                np.testing.assert_array_equal(session.command_policy_step(request['requested_command']), expected['applied_command'])
            latencies = [r['wall_ns']/1e9 for r in clock.releases if r.get('stage') == 'planning']
    finally:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:
                thread.join(timeout=2.)
        clock.close()
        if session is not None:
            session.ctx.build.scene.destroy()
        source.shutdown_genesis()
    in_run = [r['wall_ns']/1e9 for r in json.loads((root/'stage_timings.json').read_text()) if r.get('stage') == 'planning'][:len(latencies)]
    return dict(episode=f'{maze}/0', verified_decisions=decided, latencies_s=latencies, in_run_same_decisions_s=in_run)


def main(out):
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    others = [p.info['cmdline'] for p in psutil.process_iter(['cmdline']) if p.info['cmdline'] and any(
        'completed_support_v4_development.py' in c or 'render_go2_capability' in c for c in p.info['cmdline'])]
    assert not others, 'isolation required: another mission or render is running'
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    owner.source.previous.warmup()
    owner.source.previous.study.cohort.stable.floor.configure()
    results = {}
    with wall.job(base, 'isolated decision latency benchmark'):
        for arm in ('C1', 'C2', 'C4', 'C3', 'C0'):
            started = time.monotonic()
            rows = [replay(base, protocol, arm, maze) for maze in EPISODES]
            pooled = np.concatenate([r['latencies_s'][WARMUP:] for r in rows])
            in_run = np.concatenate([r['in_run_same_decisions_s'][WARMUP:] for r in rows])
            results[arm] = dict(decisions=int(pooled.size), median_s=float(np.median(pooled)), p95_s=float(np.percentile(pooled, 95)),
                in_run_under_concurrency_same_decisions=dict(median_s=float(np.median(in_run)), p95_s=float(np.percentile(in_run, 95))),
                warmup_excluded_per_episode=WARMUP, episodes=[{k: v for k, v in r.items() if k not in ('latencies_s', 'in_run_same_decisions_s')} for r in rows],
                wall_s=time.monotonic()-started)
            print(arm, json.dumps({k: v for k, v in results[arm].items() if k != 'episodes'}), flush=True)
            torch.cuda.empty_cache()
    owner.save(out, dict(schema='navigation_capability_isolated_latency.v1', sample=dict(episodes=[f'{m}/0' for m in EPISODES],
        first_decisions=DECISIONS, warmup_excluded=WARMUP), isolation='One controller at a time; no mission or render running (checked)',
        definition='Harness planning-stage wall time per decision, physics paused; real prediction-slot model; replay verified against the log',
        results=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
