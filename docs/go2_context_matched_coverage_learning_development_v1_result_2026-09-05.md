# Context-matched action coverage: completed and audited

The fixed18-model comparison and full audit are complete. Broader observed
action coverage improves longer-horizon contact prediction and offline ranking,
but does not establish a JEPA-specific advantage or safer maze navigation.
Short-horizon motion improves mainly in the direct heads; retention and contact
prediction are mixed. The primary half-second switched-contact endpoint has
zero positive labels and cannot measure hazard discrimination.

Training63480 terminated COMPLETE18, exit0, after2,426 s. Audit93989 terminated
PASS18, exit0. It checked exact schedules, source/input bindings, initialization,
all1,200 update records per model, checkpoint identities and exact intact/shuffled
prediction replay, plus independent scalar primary metrics and offline choices.
It did not retrain optimization. All models, negative results and source bytes
are retained; no early stopping, best-seed selection, retry or discarded run.
The729-test/60-file focused suite passes. No study remains running.

Root: `.generated/go2_context_matched_coverage_learning_development_v1_attempt_001`.

- Launch SHA: `1ddaaee56a42b78b1b8fb0425e15dabad7609b6fdc243820e6a2506fcd154ddf`.
- Result SHA: `f969915b6ac0fe74675c11545a5399dd1df1a01e2369c81bc9329b84154d2534`.
- Schedule SHA: `0f03cbc05941ff6bd9e9bc5d5a7ccf31c63c0d32101acdf540a435115a4c49ff`.
- Audit SHA: `707f37e4dc3225a77f26f1f1f9ca3013f684fed19a21eb991f4a019cc7b2e56b`.

The [fixed protocol](go2_context_matched_coverage_learning_development_v1_2026-09-05.md),
29 recursively bound training source/test paths and two new audit paths are
now immutable evidence. All15 prescribed paired contrasts, with every seed,
layout, interval and endpoint, are in the
[complete result](../.generated/go2_context_matched_coverage_learning_development_v1_attempt_001/result.json).
Tables below summarize the principal outcomes, not a selected best seed.

## What was actually compared

Retain all914 original windows and384 new switches:866 train/432 development
validation,16/8 layouts. Limited and expanded data arms share every sampled
current context, layout weighting, action quantile, initialization and update
budget. Only future-action/trajectory support changes at eligible moving states.
Each data arm has direct-only, supervised recurrent prediction, and JEPA training,
with three fixed seeds. No new architecture or sensor-estimator correction was
inserted into training. This context-balanced schedule differs from the older
temporal study; do not attribute historical model differences solely to coverage.

Validation has128 switched actions from32 moving contexts. All128 are contact-free
at0.5 s; at3 s there are37 contacts and91 strictly precontact motion targets.
The32 continuation alternatives also have zero0.5-s contacts, and8 contacts at3 s.
The old-later264-window population has8 first-half-second contact labels and108
positive labels among1,090 known contact horizons across all offsets/horizons.
No release-only or unknown future is used to fill a target.

The absence of switched0.5-s contacts is a substantive design limitation, not
a failed integrity check. Brier scores there measure false alarms only. Every
method's0.5-s offline chosen-contact count is necessarily zero because every
candidate is actually contact-free at that horizon. This cannot support a claim
that a ranker distinguished safe from hazardous actions.

## Prediction results

Values average the three seeds and eight layout means. Position error is metres;
Brier is dimensionless. `Supervised latent` and `JEPA latent` are recurrent-head
predictions, not evidence of online multi-step planning.

| Data | Head | Switch0.5-s position | Switch0.5-s Brier | Switch3-s position | Switch3-s Brier |
|---|---|---:|---:|---:|---:|
| Limited | Direct-only | 0.03884 | 0.001396 | 0.06364 | 0.29793 |
| Limited | Supervised direct | 0.04092 | 0.001803 | 0.07611 | 0.29631 |
| Limited | Supervised latent | 0.04376 | 0.000416 | 0.25109 | 0.36586 |
| Limited | JEPA direct | 0.04306 | 0.009856 | 0.06196 | 0.32309 |
| Limited | JEPA latent | 0.06913 | 0.003755 | 0.11915 | 0.37974 |
| Expanded | Direct-only | 0.02965 | 0.003602 | 0.05888 | 0.19547 |
| Expanded | Supervised direct | 0.02960 | 0.003411 | 0.06402 | 0.20866 |
| Expanded | Supervised latent | 0.04431 | 0.000828 | 0.12227 | 0.19514 |
| Expanded | JEPA direct | 0.02877 | 0.015111 | 0.05149 | 0.24998 |
| Expanded | JEPA latent | 0.06572 | 0.011930 | 0.11193 | 0.25599 |

Expanded coverage improves half-second position error in all three direct heads.
For direct-only, the paired decrease is0.00918 m (descriptive layout-bootstrap
interval[-0.01143,-0.00623]). Its3-s switched-contact Brier decreases0.10246
([-0.13969,-0.06945]). All five heads improve that3-s Brier endpoint, but all
increase their mean false-alarm Brier at0.5 s.

For JEPA latent, coverage changes half-second position error by-0.00341 m
([-0.01077,0.00300]) and false-alarm Brier by+0.00817. Its3-s Brier improves
by0.12375 ([-0.16569,-0.07971]). Thus the intended short-horizon predictive
benefit is not uniformly established, even though longer-horizon coverage helps.

Within expanded data, JEPA latent is worse than matched supervised latent at
0.5-s position by0.02142 m ([0.01661,0.02715]) and at3-s Brier by0.06085
([0.03203,0.09222]). Its slightly lower3-s motion error is mixed across seeds
and layouts. JEPA direct's half-second motion advantage over supervised direct
is only0.00083 m, with an interval crossing zero, while its contact Brier is
worse. These comparisons do not support promotion of JEPA over the matched
supervised model. All intervals reuse8 development layouts and have no
multiplicity correction; they are not confirmatory significance tests.

## Offline choices, stopping, and stronger simple controls

For each model/head, the3-s diagnostic ranks five actual actions for three
direction cues in32 moving contexts. There are96 correlated choices per seed;
counts below sum288 choices across the three fixed seeds, not288 independent
physical navigation episodes.

| Head | Limited contacts /288 | Expanded contacts /288 | Limited → expanded mean regret | Limited → expanded stop fraction |
|---|---:|---:|---:|---:|
| Direct-only | 46 | 7 | 1.6242 → 0.3811 | 0.424 → 0.625 |
| Supervised direct | 36 | 11 | 1.2917 → 0.5287 | 0.441 → 0.733 |
| Supervised latent | 66 | 23 | 2.2086 → 0.8596 | 0.118 → 0.410 |
| JEPA direct | 40 | 7 | 1.4638 → 0.4440 | 0.497 → 0.892 |
| JEPA latent | 80 | 30 | 2.6962 → 1.1418 | 0.181 → 0.472 |

All five coverage contrasts improve this offline ranking/contact endpoint,
with more stopping. JEPA latent's coverage regret decrease is1.5544
([-2.2517,-1.0621]); its chosen-contact fraction decreases0.17361
([-0.25,-0.11806]). Within expanded data, however, the latent heads have more
chosen contacts and regret than their own direct heads. JEPA latent minus
supervised latent regret is+0.2822 with a broad interval crossing zero.

Crucially, always-stop has zero contacts and mean regret0.2430—lower than every
learned head here. Both empirical action-mean controls also always select stop.
That does not make always-stop a navigator: it cannot explore or complete the
task. It means this offline cost alone is insufficient evidence of useful
navigation. A physical follow-up must count goal progress, stalls and completion
alongside contacts, not reward stopping by omission.

The expanded training-only action/offset mean predicts switched positions at
0.03172 m/0.04132 m for0.5/3 s and3-s Brier0.16422, below every learned3-s
Brier here. Fixed command kinematics gives0.03794 m/0.05175 m motion error but
chooses contacting actions in35/96 cases at3 s when it ignores contact risk.
Its failure and the always-stop degeneracy are complementary controls. The
empirical mean is not a scene-conditioned or optimally frequency-matched model.

## Retention and sensor controls

Expanded-minus-limited old-later all-horizon position/Brier changes are:

| Head | Position change | Brier change |
|---|---:|---:|
| Direct-only | +0.00190 m | +0.00125 |
| Supervised direct | -0.00320 m | +0.00243 |
| Supervised latent | +0.00522 m | +0.00502 |
| JEPA direct | -0.00052 m | +0.00863 |
| JEPA latent | +0.00132 m | +0.00112 |

Retention is not uniformly improved; in particular, improved switch coverage
comes with worse old-later Brier in every mean contrast. Complete per-seed,
per-layout and old-initial endpoints are preserved in the full result.

Shuffle controls retain376 of432 complete cross-layout cells and preserve past
action, proposed action and offset. All switched contexts are eligible. For
expanded data, RGB shuffling raises3-s switch Brier from0.1955 to0.2365
(direct-only),0.1951 to0.2240 (supervised latent), and0.2560 to0.2837 (JEPA
latent). Body shuffling changes those means by less than0.0006. This supports
some scene-dependent RGB use in the expanded arms, not a noncollapse or body
irrelevance proof: matching past action and offset can leave similar body states
across the same ideal-physics scenes, making the body intervention weak. The
limited latent heads even slightly improve under RGB shuffle on this stratum.

## Decision toward the scientific goal

Preserve a bounded data-coverage benefit, an unestablished JEPA-specific benefit,
the degenerate short-horizon contact endpoint, and mixed retention. Do not
promote these models into a maze-success claim or overwrite the earlier physical
panel where the older JEPA half-second head had a bounded contact advantage.
The two results differ in training schedules, visited states, horizons and
offline versus executed decisions.

Next:

1. Advance the actual RGB/body exit/place/arrival component and temporal memory
   connection. The gravity-feedback improvement is separate useful sensor
   evidence; keep its remaining43-cm tail, near-field gap and appearance limits.
   Full navigation must start making observation-conditioned decisions, not
   accumulate only predictive or geometric scores.
2. Before a new short-horizon hazard claim, specify development contexts whose
   actual candidate outcomes include both contacts and noncontacts at that
   horizon, including later approach/switch states. Retain unavailable prefixes
   explicitly and do not manufacture labels or reopen protected material. The
   current1-s moving-prefix panel cannot answer that discrimination question.
3. A separately fixed physical coverage comparison should reuse the same
   learned/control conditions and costs, pair fresh layouts, and report progress,
   stalls, contacts, release stability and completion. Include direct, supervised
   and JEPA heads plus stopping; do not compare a new expanded model only against
   historical physical results or combine a changed controller/sensor estimator
   into the coverage contrast. Offline improvement is motivation to test, not
   permission to assume physical improvement.
4. Do not start an architecture/weight sweep merely to rescue JEPA. Establish
   scene/action outcome support and an actual observation-to-memory/controller
   task, then test prediction's incremental contribution with honest negative
   results. Independent full-maze discovery/return and real Go2 remain open.
