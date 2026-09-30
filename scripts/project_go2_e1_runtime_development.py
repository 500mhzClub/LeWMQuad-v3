"""E1 running-time and storage projection (Andrew's rulings of 29 Sep 2026).

Missions are scheduled with the same greedy rule as the capability cohort runners: at most five
concurrent owners and at most two C3 owners. Each mission's owner wall time is drawn from the
measured wall times of that controller:
- C1, C3 and C4 from the fresh check, which ran the versions that enter E1 on fresh mazes;
- C2 and C0 from capability validation, the only runs of those controllers on the frozen harness.
The makespan is bootstrapped over those draws.

Training for the extra seeds, and each seed's acceptance measures, run as serial GPU jobs
before that seed's missions, with measured durations. The projection is the median makespan
plus training. The cap is the projection plus the stated margin. The same function
re-projects at the pilot and at each seed boundary from E1's own completed missions.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm import e1_running_time_budget_development as e1_budget
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
MAZES, SEEDS, C0_MAZES = 60, 3, 10
WORKERS, C3_LANES = 5, 2
MARGIN = .20
DRAWS, DRAW_SEED = 1000, 2026093001
GIB = 1024**3
RESERVE_BYTES = 12*GIB
SOURCES = {'C1': 'c3v2_check_C1_chk*_ep0_attempt001', 'C3': 'c3v2_check_C3_chk*_ep0_attempt001',
           'C4': 'c3v2_check_C4_chk*_ep0_attempt001', 'C2': 'v4_completed_support_validation_C2_val*_ep0_attempt001',
           'C0': 'v4_completed_support_validation_C0_val*_ep0_attempt001'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(pattern):
    rows = []
    for directory in sorted((BASE/'runs').glob(pattern)):
        result = directory/'result.json'
        if result.exists():
            wall = json.loads(result.read_text())['wall_s']
            rows.append(dict(run=directory.name, wall_s=wall, bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())))
    return rows


def makespan(missions):
    """The cohort runners' rule: whenever an owner is free, launch the earliest pending mission whose lane has room."""
    queues = {}
    for index, (arm, seconds) in enumerate(missions):
        queues.setdefault(arm, []).append((index, seconds))
    heads = {arm: 0 for arm in queues}
    running, now, c3 = [], 0., 0
    while running or any(heads[a] < len(q) for a, q in queues.items()):
        while len(running) < WORKERS:
            eligible = [(queues[a][heads[a]][0], a) for a in queues if heads[a] < len(queues[a]) and (a != 'C3' or c3 < C3_LANES)]
            if not eligible:
                break
            _, arm = min(eligible)
            heapq.heappush(running, (now+queues[arm][heads[arm]][1], arm))
            heads[arm] += 1
            c3 += arm == 'C3'
        now, arm = heapq.heappop(running)
        c3 -= arm == 'C3'
    return now


def plan(seeds):
    """Mission order per block: C3 and C4 interleaved per maze, with the CPU controllers alongside in the first block."""
    blocks = []
    for seed in seeds:
        order = []
        for maze in range(MAZES):
            order += [('C3', maze), ('C4', maze)]
            if seed == 1:
                order += [('C1', maze), ('C2', maze)] + ([('C0', maze)] if maze < C0_MAZES else [])
        blocks.append((seed, order))
    return blocks


def project(samples, seeds, training_s, draws=DRAWS):
    rng = np.random.default_rng(DRAW_SEED)
    per_block = {}
    for seed, order in plan(seeds):
        spans = [makespan([(arm, rng.choice(samples[arm])) for arm, _ in order]) for _ in range(draws)]
        per_block[seed] = dict(missions=len(order), median_h=float(np.median(spans))/3600, p90_h=float(np.quantile(spans, .9))/3600,
                               training_and_acceptance_h=training_s.get(seed, 0.)/3600)
    median = sum(b['median_h']+b['training_and_acceptance_h'] for b in per_block.values())
    p90 = sum(b['p90_h']+b['training_and_acceptance_h'] for b in per_block.values())
    return dict(blocks=per_block, projected_h=median, projected_p90_h=p90)


def main(pair):
    output.install(BASE)
    rows = {arm: measured(pattern) for arm, pattern in SOURCES.items()}
    samples = {arm: np.asarray([r['wall_s'] for r in v]) for arm, v in rows.items()}
    # Serial GPU jobs per extra seed, from measured wall times of this programme's fits.
    predictor = json.loads((owner.REPO/'.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/result.json').read_text())['wall_s']
    acceptance = 2712.  # 13:16:34 to 14:01:46 BST on 29 September (c3v2_acceptance_v1 log and result times)
    if pair == 'v2':
        readout = json.loads((BASE/'c3v2_readout_fit_v1/result.json').read_text())['wall_s']
        c4 = json.loads((BASE/'c4v2_fit_v1/result.json').read_text())['gpu_owner_wall_s_this_fit']
    else:
        readout = json.loads((BASE.parent/'go2_maze_view_readout_v1_attempt_003/result.json').read_text())['wall_s']
        c4 = json.loads((BASE/'c4_fit_attempt002/result.json').read_text())['gpu_owner_wall_s']
    per_seed = predictor+readout+c4+acceptance
    # Queued 2x2 analysis readout fits (Andrew, 29 Sep): (a) actual-future features for C3-v2's data, at the v1
    # readout fit's measured rate scaled to its frame count; (b) predicted features for C3-v1's 8,414 contexts, at
    # the C3-v2 fit's measured rate; plus one acceptance-evaluator pass. Serial, charged to block 2.
    v1_fit = json.loads((BASE.parent/'go2_maze_view_readout_v1_attempt_003/result.json').read_text())['wall_s']
    v2_fit = json.loads((BASE/'c3v2_readout_fit_v1/result.json').read_text())['wall_s']
    analysis = v1_fit*(9662+2928)/9662 + v2_fit*8414/11182 + acceptance
    training = {1: 0., 2: per_seed+analysis, 3: per_seed}
    result = project(samples, (1, 2, 3), training)
    used = e1_budget.running_hours(BASE)
    cap = (result['projected_h']+used)*(1+MARGIN)
    size = {arm: float(np.mean([r['bytes'] for r in v])) for arm, v in rows.items()}
    counts = dict(C3=MAZES*SEEDS, C4=MAZES*SEEDS, C1=MAZES, C2=MAZES, C0=C0_MAZES)
    storage = {arm: counts[arm]*size[arm] for arm in counts}
    free = __import__('shutil').disk_usage(BASE).free
    per_seed_bytes = MAZES*(size['C3']+size['C4'])
    report = dict(schema='e1_runtime_projection.v1', pair=pair, margin=MARGIN, e1_running_hours_already_used=used,
        measured_missions={arm: dict(n=len(v), median_wall_s=float(np.median(samples[arm])), mean_wall_s=float(np.mean(samples[arm])),
                                     max_wall_s=float(np.max(samples[arm])), mean_bytes=size[arm], source=SOURCES[arm]) for arm, v in rows.items()},
        extra_seed_gpu_jobs_s=dict(predictor=predictor, readout=readout, c4=c4, acceptance=acceptance, total=per_seed),
        analysis_fits_s=analysis,
        projection=result, running_time_cap_h=cap,
        storage=dict(projected_bytes=sum(storage.values()), by_controller=storage, per_seed_c3_c4_bytes=per_seed_bytes,
                     free_bytes_now=free, reserve_bytes=RESERVE_BYTES, headroom_after_e1_bytes=free-RESERVE_BYTES-sum(storage.values())),
        scheduler=dict(workers=WORKERS, c3_lanes=C3_LANES, draws=DRAWS, draw_seed=DRAW_SEED), script_sha256=sha(__file__))
    root = BASE/'e1_projection'
    root.mkdir(exist_ok=True)
    path = root/f'projection_{pair}_{len(samples["C3"])}c3_with_analysis_fits.json'
    owner.save(path, report)
    print(json.dumps({k: report[k] for k in ('pair', 'e1_running_hours_already_used', 'measured_missions', 'extra_seed_gpu_jobs_s', 'projection', 'running_time_cap_h', 'storage')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=('v1', 'v2'), required=True)
    main(p.parse_args().pair)
