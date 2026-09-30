"""E1 exploratory-arm running-time cap and combined storage check (declared 30 Sep 2026, a435cbcc).

The exploratory arm is C3-v3 and C4-v3, one seed, on the same 60 sealed mazes. It needs no
training. Its mission wall times come from the exploratory safety check (C3-v3 and C4-v3 on
the 10 round safety-check mazes). The scheduler and bootstrap are those of the primary
projection. The cap is the median projection plus 20%, plus any `E1X` running time already
used.

Storage: the primary projection plus this arm must leave at least 15 GiB above the 12-GiB
reserve, measured against RecoveryStorage's free space now.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm import e1_running_time_budget_development as e1_budget
from scripts import project_go2_e1_runtime_development as primary
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = primary.BASE
MARGIN, GIB = .20, 1024**3


def main():
    output.install(BASE)
    rows = {arm: primary.measured(f'e1x_safety_{arm}_m*_ep0_attempt001') for arm in ('C3', 'C4')}
    samples = {arm: np.asarray([r['wall_s'] for r in v]) for arm, v in rows.items()}
    rng = np.random.default_rng(primary.DRAW_SEED)
    order = [(arm, maze) for maze in range(primary.MAZES) for arm in ('C3', 'C4')]
    spans = [primary.makespan([(arm, rng.choice(samples[arm])) for arm, _ in order]) for _ in range(primary.DRAWS)]
    used = e1_budget.running_hours(BASE, prefix='E1X ')
    projected = float(np.median(spans))/3600
    cap = (projected+used)*(1+MARGIN)
    sizes = {arm: float(np.mean([r['bytes'] for r in v])) for arm, v in rows.items()}
    exploratory_bytes = primary.MAZES*(sizes['C3']+sizes['C4'])
    primary_projection = json.loads((BASE/'e1_projection/projection_v2_10c3_with_analysis_fits.json').read_text())
    primary_bytes = primary_projection['storage']['projected_bytes']
    free = shutil.disk_usage(BASE).free
    headroom = free-12*GIB-primary_bytes-exploratory_bytes
    result = dict(schema='e1x_runtime_projection.v1', declaration_commit='a435cbcc',
                  measured_missions={arm: dict(n=len(v), median_wall_s=float(np.median(samples[arm])), mean_wall_s=float(np.mean(samples[arm])),
                                               mean_bytes=sizes[arm]) for arm, v in rows.items()},
                  missions=len(order), projected_h=projected, projected_p90_h=float(np.quantile(spans, .9))/3600,
                  e1x_running_hours_already_used=used, running_time_cap_h=cap,
                  storage=dict(free_bytes_now=free, reserve_bytes=12*GIB, primary_projected_bytes=primary_bytes,
                               exploratory_projected_bytes=exploratory_bytes, headroom_above_reserve_after_both_bytes=headroom,
                               headroom_above_reserve_after_both_gib=headroom/GIB, meets_15_gib=headroom >= 15*GIB),
                  primary_cap_h=primary_projection['running_time_cap_h'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    owner.save(BASE/'e1_projection/projection_exploratory_arm.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
