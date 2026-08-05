# Go2 planner-oracle assay V1 result

Date: 2026-08-04

Status: terminal development-only result; preregistered H1 gate passed. The
conditional H2 run is not authorized or needed.

## Result

The existing nine-primitive receding-horizon seam can convert a correct
candidate ranking into materially better local control. Across all 24 paired
development scenes, privileged kinematic endpoint scoring improved progress by
0.4320 m over the deterministic shuffled-score intervention. The 10,000-draw
whole-scene bootstrap 95% interval was [0.3773, 0.4833] m, entirely above zero.

| policy | success | mean progress (m) | mean final distance (m) | mean path efficiency |
|---|---:|---:|---:|---:|
| bearing (privileged ceiling) | 24/24 | 0.9000 | 0.3000 | 0.9140 |
| oracle MPC | 14/24 | 0.8151 | 0.3849 | 0.7007 |
| random | 0/24 | 0.4968 | 0.7032 | 0.5592 |
| shuffled oracle scores | 0/24 | 0.3831 | 0.8169 | 0.3674 |
| hold | 0/24 | 0.0000 | 1.2000 | 0.0000 |

All 24 scenes were complete and none was skipped. `oracle_mpc` had exactly zero
first-action regret on all 224 decisions. The shuffled arm had positive
tie-aware regret on 247/288 decisions (85.76%), mean regret 0.06824 m, and a
nonidentity selected score source on 251/288 decisions. These checks establish
that the intervention changed useful action rankings rather than merely
renaming candidates.

## Preregistered gate

Every criterion passed:

- 24 complete paired scenes, no skips;
- oracle maximum first-action regret <= 1e-9 m (observed 0);
- shuffled regret-positive rate >= 0.25 (observed 0.8576);
- shuffled mean regret >= 0.02 m (observed 0.06824 m);
- oracle progress advantage >= 0.15 m (observed 0.4320 m);
- bootstrap lower 95% bound > 0 (observed 0.3773 m);
- oracle progress greater than hold and random.

Bearing superiority and kinematic fall/safety metrics were deliberately excluded
from the gate. Bearing is already saturated on this visible-target task, and
kinematic execution cannot support a physical safety claim.

## Interpretation

This reverses the possible diagnosis that planner mechanics or the primitive
bank necessarily prevent useful action-conditioned control. There is large,
causal score-ranking headroom at H1, so no H2 rescue experiment is warranted.

It does not show that any learned visual representation captures that ranking,
nor does it establish obstacle-rich global navigation, physical locomotion,
persistent memory, held-out generalization, or safety. The next justified test
is one learned scorer—frozen dense DINOv2 tokens with a temporal
action-conditioned predictor—against persistence and action/history controls.
No new rendered data is required for that test.

## Bound artifacts

- Preregistration:
  `docs/lewm_go2_planner_oracle_assay_v1_preregistration_2026-08-04.md`
  (`6338a16a655314777b99fd8eb765ffc1399f3b4e088c55a4f18aef7c9cd49fad`)
- Raw result:
  `.generated/oracle_mpc_assay_v1/full_development_24scene_h1_seed7.json`
  (`1e733c8ccc9b20b255ab966f4d3902691120881d4c78ba7a28c1377140eb7537`)
- Analysis:
  `.generated/oracle_mpc_assay_v1/full_development_24scene_h1_seed7_analysis.json`
  (`2aef8f5951c0273d35b44b6cec849e337d60b9d78c5f9a3487d808bec6fc7078`)
- Benchmark source:
  `scripts/benchmark_lewm_closed_loop_mpc.py`
  (`596ed8bae689573da1d1ca74c915fa365d9775e6701e6b34f4e46ff454e29867`)
- Analyzer source:
  `scripts/analyze_go2_planner_oracle_assay_v1.py`
  (`4f979e4ad8bbd20b7e05e0461624741f2884ea3ffce27312a83748751b054cd3`)
