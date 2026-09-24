# Multi-junction route V1: result

All eight fresh routes completed every planned directed crossing without contact,
including the four-edge dead-end returns. Four met the complete declared final
arrival endpoint. The other four failed **only final heading**, not crossing,
speed, angular speed, height, attitude or contact checks. The fixed criteria and
all failures remain unchanged. See the [protocol](go2_multijunction_route_development_v1_2026-09-05.md).

| Route | Width | Planned crossings completed | Full task | Final heading error |
|---|---:|---:|---|---:|
| Left-right | 0.9 m | 3/3 | Fail: heading | 0.5255 rad |
| Left-right | 1.2 m | 3/3 | Pass | 0.3195 rad |
| Right-left | 0.9 m | 3/3 | Pass | 0.3480 rad |
| Right-left | 1.2 m | 3/3 | Pass | 0.1509 rad |
| Hairpin | 0.9 m | 3/3 | Fail: heading | 0.3805 rad |
| Hairpin | 1.2 m | 3/3 | Pass | 0.2070 rad |
| Dead-end return | 0.9 m | 4/4 | Fail: heading | 0.4575 rad |
| Dead-end return | 1.2 m | 4/4 | Fail: heading | 0.4246 rad |

The heading limit is 0.35 rad. Three routes satisfy every intermediate and final
arrival check. The narrow right-left route misses its intermediate heading
(0.5436 rad) but successfully continues and meets the final checks. Actual
continuation again limits what an instantaneous arrival proxy can establish.
All four full-task successes also satisfy the trailing 0.20-s arrival window.

This is stronger physical scope than the earlier within-corridor second boundary:
there are real branch junctions, opposite consecutive turns, a hairpin and an
explicit reverse-direction traversal. The baseline turns to drive forward during
dead-end return; this is not a newly learned reversing maneuver. No reset occurs
between edges. Width and spawn perturbation are confounded, as declared, so their
effects cannot be separated here. These eight development routes do not establish
maze-level uncertainty or final generalization performance.

## Evidence and observation interface

Collection retained 80,500 physics samples, 8,050 ideal simulated body-sensor
samples and **1,498 causal RGB/history packets**. Every route passed raw auditing:
wall identity coverage, source/gait/gain bindings, scalar native contact
reconstruction, causal controller inputs and requested/applied commands,
crossing/arrival recomputation, global clocks, ideal sensor conversion, history
prefixes, RGB pixels and proper optical frames. Command and terminal observation
coverage was verified without resetting physical or sensor state between edges.

The versioned [route observation reader](../lewm/route_rgb_dataset_development.py)
allows longer sequences while retaining strict modality, field, timestamp and
path checks. The original 46-frame probe reader remains unchanged. A junction
frame was visually inspected; physical walls and branch geometry are present.
The scenes are still low-texture synthetic environments.

The [runner](../scripts/run_go2_multijunction_route_development_v1.py) uses oracle
pose and routes; RGB/body observations are recorded, not used for action selection.
This is not visual navigation, JEPA utility, online place association or hardware
qualification. No claim is made that all local arrival contracts now work.

Output: `.generated/go2_multijunction_route_development_v1_attempt_001/`.
Result SHA-256:
`46e88ab805cd3ea5e285f03e7136070fbc95b80e3d6c55b2bd4c368f5df62801`.
Audit: [raw route audit](../scripts/audit_go2_multijunction_route_development_v1.py).
The combined explicit suite passes **325 tests**. Executed source and prior
studies remain hash-bound and unchanged; no trials were retried or discarded.

## Next scientific decision

The corrected plant can execute multi-junction routes. Do not indefinitely
postpone learning experiments while optimizing this particular heading proxy.
Retain the oracle reference and all continuous arrival metrics. If a future task
requires a precise terminal orientation, specify and test that requirement as
part of its endpoint; do not retrospectively relax this study's threshold.

Next, collect action-diverse causal data on newly generated development scenes:
different actions from the same observed prefix, and the same actions in different
scenes. Fresh deterministic prefix execution can provide matched starts without
loading any legacy snapshots; verify prefix equality rather than assume it.
Fix the scene population, training/validation roles, action tapes, horizons,
collision censoring and budgets before collection. Keep all sibling branches
and scene descendants in one role. Independent procedural layouts, not frame
splits or additional widths of these four motifs, are needed for later claims.

Use that dataset for a strong direct action-conditioned baseline and a JEPA
comparison. Evaluate executed outcomes and candidate regret as well as latent
prediction; separate predictive training from online rollout. Any oracle local
intent used in a conditional ranking test must be identical across methods and
explicitly distinguished from targets discovered by a deployed explorer. The
ultimate goal still requires learned visual control, observed online memory,
independent-maze evidence and bounded hardware evaluation.
