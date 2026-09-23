# Decision-headroom Phase 1: technical stop and proposed bounded recheck

23 September 2026. **Stage A is complete. Phase 1 did not qualify. No Phase 2
collection or comparative scoring has occurred.** The pilot stopped because
physical restoration succeeded while rendered observations did not reproduce
the source. This is an audit-tooling defect, not a JEPA performance result.

## Stage A: all four original assignments completed unchanged

| Layout | Matched head | Goal / home | Simulated seconds | Holds / plans | Translation windows | 700-ms translation XY RMSE, learned / command |
|---|---|---|---:|---:|---:|---:|
| 00 | Existing data | 0 / 0 | 480.32 | 987 / 1198 | 60 | 66.94 / 13.59 mm |
| 00 | Maze data | 0 / 0 | 480.32 | 979 / 1198 | 95 | 79.65 / 9.01 mm |
| 02 | Maze data | 0 / 0 | 480.32 | 1140 / 1198 | 4 | 86.70 / 9.34 mm |
| 02 | Existing data | 0 / 0 | 480.32 | 957 / 1198 | 102 | 68.76 / 13.41 mm |

All four exhausted their navigation budgets, with zero disallowed contacts
and pipeline faults. The coordinator and all native owners/readers exited.
No trial was duplicated, restarted or extended. The retained initial head
previously failed layout 00 and completed goal/home on layout 02; these are
reused predecessor observations. Neither matched head reproduced that success.
This is a development intervention on two exposed, deliberately selected
layouts, one run per cell. Different trajectories prevent treating differences
in aggregate motion error as an isolated readout effect. The
[complete Stage A record](go2_maze_view_readout_navigation_2026-09-23.md)
contains all metrics, identities and evidence links.

## Reference sanity evidence, including the fixture error

The original 24-case analytical qualification passed all 24 prewritten optimal
sets and all geodesic requirements, but failed two unsafe-forward assertions.
The two fixtures actually left 10/20 mm clearance, above the fixed 5-mm minimum;
their conservative swept lower bounds were 9.718/19.718 mm. Their unsafe labels
were incorrect. No evaluator weight, safety margin, geometry or saved cost was
changed to obtain a pass. No case was re-executed.

An explicit analytical label erratum records these two post-execution
corrections. It is not represented as 24 originally correct safety labels.
Both the original strict failure and derived corrected qualification are
retained and hash-bound in the [technical-stop configuration](go2_decision_headroom_phase1_technical_stop_2026-09-23.json).
The initial prephysics owner used 2.775 s wall and 2.12 CPU seconds. A single
prephysics continuation carried that usage and retained evidence into the
original cumulative caps; it required zero earlier source/branch attempts.

## Physical pilot results

| Quantity | Measured result |
|---|---:|
| Source attempts / fully collected sources | 3 / 2 |
| Completed source classes | Command history and reactive, layout 00 |
| Third source | Action, interrupted during worker initialisation before physics |
| Collected snapshots / planned slots | 8 / 24 |
| Source-trace restoration branches | 24 |
| Candidate branches, including repeats | 144 |
| Total attempted branches | 168 |
| Source-trace pose/yaw/contact horizon checks | 216 |
| Maximum source-restoration position / yaw discrepancy | 0 m / 0 rad |
| Source-restoration contact checks | All match |
| Pairwise candidate-repeat position / yaw discrepancies | All zero over 1296 horizon comparisons |
| Physical terminal events in branches | 0 |
| Bitwise-matching source-replay RGB images | **0 / 384** |
| Comparatively scored method selections / rankings / regrets | **0** |

Each completed source supplied all four fixed snapshots. Each state received
three full recorded-command trace replays and three repeats of each exact
six-candidate 800-ms tape, including the committed prefix. Physical agreement
does not qualify the rendered images or the complete audit. The 16 uncollected
slots are not 16 measured restoration failures; they are missing coverage.
No action-controller source or layout-02 source was completed.

For the first source's 192 source-replay images, camera transforms matched
exactly but mean absolute RGB channel error averaged 2.204/255 across images,
with a maximum image mean of 21.270/255. This uses saved source/replay images
only, with no new rendering, feature comparison or nuisance intervention.

Installed Genesis code explains a likely failure mechanism:
`Visualizer.update_visual_states()` and `RasterizerContext.update()` return
early when their cached `_t` is greater than or equal to `scene._t`.
The adapter rewinds `scene._t` but leaves these visual caches at the later
source/branch timestamp. Thus restored camera poses can coexist with stale
rendered geometry. This is a source-backed diagnosis consistent with the
measurements; the proposed correction is not yet physically verified.

The owner was stopped with SIGINT through its existing exception/closeout
path, preserving completed branches, the interrupted assignment and resource
records. It exited with status 130. The stop was technical, not disk exhaustion.
No controller/model source or sensor configuration changed.

## Measured budget and limits of extrapolation

| Resource | Measured | Original ceiling |
|---|---:|---:|
| Owner execution wall, including prephysics attempt | 459.55 s | 14400 s |
| Owner and child CPU | 625.18 core-seconds | 230400 core-seconds |
| Source physics | 35.2 s | 105.6 s |
| Branch physics | 134.4 s | 403.2 s |
| Combined physics | 169.6 s | 508.8 s |
| Peak sampled aggregate RAM | 8.55 GiB | 48 GiB |
| Peak sampled total device VRAM | 5.06 GiB | 24 GiB |
| Retained bytes at owner closeout | 524565064 (0.489 GiB) | 8 GiB |

The CPU measure excludes separate read-only closeout/document commands. The
192-image comparison took 1.214 s wall outside the owner; its CPU was not
separately metered. Memory measurements are sampled, not OS-enforced ceilings;
VRAM includes other device users. No cap or filesystem reserve was reached.
Later report/config files add small retained bytes beyond the closeout sample.

Across 168 attempted branches, median wall time was 0.623 s (maximum 1.405 s),
including median restoration time 0.017 s. These branch timings include the
render-cache defect and must not be used as a qualified Phase 2 rendering
budget. Component timing, with outputs discarded, measured a median 0.875 s
for three-image encoding, 0.303 s for one image, 0.366 s for the six-candidate
800-ms predictor batch, and about 0.65/0.88 ms for the existing/maze readouts.
The timing sample is small and is not an end-to-end inference latency claim.

## Proposed next action: one bounded RGB restoration recheck

The [reviewable audit-only patch](go2_decision_headroom_render_cache_restore_proposal_2026-09-23.patch)
invalidates both render caches through Genesis's existing visualizer reset
after physical restoration, before restoring RNG. It is **not applied**.
The proposal changes no controller, model, camera calibration or sensing.

Request approval for exactly two existing snapshots: frame 12 from each of the
completed command-history and reactive sources. From each restored state,
execute three original source traces and three repeats of the six fixed
candidates: **42 attempts**, 33.6 branch-physics seconds. Permit at most two
scene initialisations with a combined 3 s settling, so the added physics cap
is **36.6 s**. No new source trajectory or snapshot is requested. Retain every
failed attempt and all original evidence.

The proposed limits are 15 min execution wall, 4 CPU core-hours, 16 GiB RAM,
8 GiB total device VRAM, 1 GiB retained and 2 GiB peak writes, preserving
12/4 GiB RecoveryStorage/workspace reserves. No raw depth or dense features
are retained; no comparative selections, fitting or artifact retirement.
Acceptance requires all original pose/yaw/contact tolerances and **96/96
bitwise-matching source-replay RGB images**, plus agreement between candidate
repeats. Any mismatch remains a failed observation. Stop after the fixed
attempts; no tolerance adjustment or automatic extension.

This is a new set of replay attempts on already consumed state/action/repeat
identities, so explicit approval is required before execution under the
handoff's fixed-limit rule. Cumulative accounting would be at most 210 branch
attempts and 206.2 simulated seconds, including the failed original pilot.
A passing recheck would still leave the full Phase 1 pilot incomplete and
would not authorise Phase 2.

## Checkpoint status

The [protocol draft](go2_decision_headroom_protocol_draft_2026-09-23.md) and
[precision scenarios](go2_decision_headroom_precision_scenarios_2026-09-23.json)
preserve the intended scientific question, sampling structure, shared masks
and layout-level inference. No method-regret variance was estimated.

**The full checkpoint-(a) package is not qualified or complete.** RGB fidelity,
all source/layout coverage and reliable branch costs remain missing. Final
Phase 2 layout identities/counts, sampling design, near ties and measured
budget cannot honestly be frozen as ready to execute. This report submits
the failed pilot evidence and a concrete bounded remedy for approval, not a
request to approve Phase 2 or a claim that the research goal is achieved.
