"""PRELIMINARY: split-conformal bounds on the centre-position forecast error, per controller (Andrew, 2 October 2026).

Calibration for the calibrated-margin experiment
(docs/go2_navigation_calibrated_margin_experiment_plan_2026-10-02.md). Written and committed
before any evaluation run.

Score: e_f per executed moving decision = the largest error over the 800-ms check horizon
between the controller's applied forecast centre path and the true centre path, both anchored
at the true pose (scripts/analyse_go2_forecast_sensitivity_error_budget_development.py).
Stratum: near-wall decisions, where the checked path's true clearance is below 0.60 m.
Decisions issued by recovery interventions (back-ups, deadlock escapes, terminal spin breaks,
latch-timeout releases) are excluded.

Calibration data: the preliminary run, recovery on (cohort prelim_on), mazes 50-89 of
prelim_test_v1, disjoint from the evaluation mazes 30-49. Possible shift: calibration ran with
recovery on and the nominal disc; evaluation runs with recovery off and the inflated disc, so
closed-loop behaviour differs. Validity is checked on the evaluation runs by realised exceedance
(the share of near-wall decisions whose e_f exceeds the bound).

Bound: the ceil((n+1)(1-alpha))-th smallest near-wall score, alpha = 0.05 (p95) and 0.01
(p99). This is a per-decision, marginal guarantee under exchangeability, not a per-mission
guarantee: a mission makes hundreds of near-wall decisions, so at p95 several exceedances per
mission are expected. p99 is the safety-relevant level. Decisions within a mission are
correlated; a maze-block bootstrap (B = 2000) gives an interval for each bound.

Usage: calibrate_go2_forecast_margins_development.py --out docs/go2_navigation_calibrated_margins_2026-10-02.json [--workers 2]
"""
import argparse
import datetime
import json
import math
from multiprocessing import Pool
import re
from pathlib import Path

import numpy as np

from scripts.analyse_go2_forecast_sensitivity_error_budget_development import NEAR_WALL_M, mission
from scripts.diagnose_go2_forecast_sensitivity_failures_development import BASE

COHORT, FIRST_MAZE, CONTROLLERS = 'prelim_on', 50, ('C1', 'C3', 'C4')
ALPHAS = {'p95': .05, 'p99': .01}
BOOTSTRAP, SEED = 2000, 20261002


def conformal(scores, alpha):
    s = np.sort(scores)
    k = math.ceil((len(s)+1)*(1-alpha))
    return float(s[min(k, len(s))-1])


def main(out, workers):
    config = json.loads((BASE/'dev_cohorts'/COHORT/'config.json').read_text())
    jobs = [(COHORT, job[4]) for job in config['plan'] if job[0] in CONTROLLERS and job[2] >= FIRST_MAZE
            and (BASE/'runs'/job[4]/'episode_evaluation.json').exists()]
    with Pool(workers) as pool:
        results = pool.map(mission, jobs, chunksize=1)
    report = dict(schema='go2_calibrated_forecast_margins.v1', created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                  label='PRELIMINARY development mode',
                  calibration=dict(cohort=COHORT, mazes=f'{FIRST_MAZE}-89 (prelim_test_v1)', recovery='on', coverage_rule_fix=False,
                                   evaluation_mazes='30-49 (disjoint)', near_wall_true_path_clearance_below_m=NEAR_WALL_M,
                                   excluded='decisions issued by recovery interventions',
                                   possible_shift='calibration recovery on with the nominal disc; evaluation recovery off with the inflated disc'),
                  guarantee='per decision, marginal under exchangeability; not per mission. p99 is the safety-relevant level.',
                  controllers={})
    rng = np.random.default_rng(SEED)
    for controller in CONTROLLERS:
        missions = [r for r in results if r['controller'] == controller]
        per = [np.array([d['e_f'] for d in r['decisions'] if d['true_clearance'] < NEAR_WALL_M and not d['intervention']]) for r in missions]
        scores = np.concatenate(per)
        excluded = sum(d['intervention'] for r in missions for d in r['decisions'] if d['true_clearance'] < NEAR_WALL_M)
        entry = dict(missions=len(missions), mazes=sorted(int(re.search(r'prelim_test(\d+)', r['assignment']).group(1)) for r in missions),
                     near_wall_decisions=int(len(scores)), excluded_intervention_decisions=int(excluded),
                     all_decisions=int(sum(len(r['decisions']) for r in missions)))
        for level, alpha in ALPHAS.items():
            boots = []
            for _ in range(BOOTSTRAP):
                pick = rng.integers(0, len(per), len(per))
                boots.append(conformal(np.concatenate([per[i] for i in pick]), alpha))
            entry[level] = dict(margin_m=round(conformal(scores, alpha), 4), alpha=alpha,
                                block_bootstrap_95_m=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)])
        report['controllers'][controller] = entry
        print(controller, {k: v for k, v in entry.items() if k != 'mazes'})
    Path(out).write_text(json.dumps(report, indent=1)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    main(a.out, a.workers)
