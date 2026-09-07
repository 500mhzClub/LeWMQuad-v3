# Actual RGB/body active scan: useful observations, failed physical qualification

The fixed sixteen-scene collection is **COMPLETE** and its full raw audit is
**PASS**. This is a bounded local observation result, not maze navigation:
selected RGB/body views propose bearings toward39/40 actual openings, but
0/16 scans meet every fixed physical success criterion. No criterion was relaxed.

## What was actually executed

Dead end, corner, tee and cross, each crossed with0.9/1.2 m width and
−0.15/+0.15 rad initial heading. A centered real simulated Go2 settled, then
attempted four gyro-controlled quarter-turn views and a zero-command release.
The controller and observer consumed actual RGB/body/control packets, never
the known geometry or true pose. This was a fixed scan, not RGB-driven steering
or a JEPA controller. All outcomes and the original source remain frozen.

The RGB observer uses palette-positive, bottom-connected nominal floor extension
in a1.0–1.6 m body-range annulus. Its typed proposals cannot become trusted
memory exits or traversals automatically. Negative evidence remains unknown.
The estimator's nominal ground plane remains unqualified.

## Complete observation accounting

| Observation population | Frames | Proposals | Opening-directed | Closed-side / unscorable | Unique open sides covered |
|---|---:|---:|---:|---:|---:|
| Initial view |16|16|16|0 /0|16/40|
| Initial + completed target views (primary) |66|51|51|0 /0|39/40|
| All available control frames (secondary) |3,040|2,402|2,402|0 /0|40/40|

The one missed primary opening is the north branch of the0.9 m corner at
−0.15 rad heading: contact stopped the scan before its first target view.
It was visible during motion in the secondary population. That does not convert
the failed scan to success.15/16 specimens cover all their open sides at selected
views; duplicate proposals do not increase unique-side counts.

Evaluation transforms each proposal center with that frame's actual body pose
and asks which central-cell boundary its ray reaches. This verifies an
opening-directed bearing, **not** the physical width of the opening, a safe
finite-body path, arrival, place identity or a discovered beacon. The fixed
palette and flat-junction geometry are strong priors; prior appearance controls
already showed the floor rule fails grayscale/channel changes. These data do
not establish general RGB semantics, reliable negative evidence or transfer.

## Physical outcomes and why they matter

| Population | Contact stops | Completed four target views | Full physical success |
|---|---:|---:|---:|
|0.9 m dead ends and corners |4/4|0/4|0/4|
|0.9 m tees and crosses |0/4|4/4|0/4|
|All1.2 m specimens |0/8|8/8|0/8|
|All planned specimens |4/16|12/16|0/16|

No sensor faults occurred. All twelve completed scans passed release motion,
terminal stability and maximum0.15 m base-drift criteria; each missed the fixed
0.12 rad final true-heading tolerance. Final errors are0.123601 rad at the
negative initial heading and0.133321 rad at the positive heading. These are
small misses, not evidence that turning is impossible, but they remain failures.
Completed-scan maximum base drifts are0.082488/0.083761 m; maximum gyro-heading
errors are0.048811/0.045008 rad. Ideal gyro inputs do not imply exact integrated
orientation or calibrated hardware performance.

The four contact stops occur after4.482 or6.140 s of scanning, with only
0.054522/0.063985 m maximum base drift. Native contacts identify RR_calf or
RL_calf against the closed south wall. A small base displacement and correct
forward-camera opening proposal therefore do not protect the legs' swept
volume. The corresponding tees/crosses lack that wall and complete the motion;
this geometry-specific result must not be reduced to a universal width rule.

Physical traces repeat exactly across several different rendered geometries
until a wall interaction. The sixteen cases are a deterministic factorial
assay, not sixteen independent stochastic physical successes/failures. Do not
attach binomial generalization confidence intervals to these counts.

## Verification and preserved evidence

All157 bound source/test/protocol paths, predecessor inputs and fixed gait were
verified. The audit reconstructs16,632 body-sensor samples from166,322 native
physical samples, validates3,104 actual RGB packets and their camera mounts,
checks actuator gains, command slew and immediate native-contact termination,
and exactly reproduces all3,040 live scan/ground/proposal decisions. It then
recomputes physical and opening-directed outcomes from the saved evidence.
It does not rerender images or grant safe-traversal qualification.

Exact root: `.generated/go2_active_exit_scan_development_v1_attempt_001`.

- Launch SHA-256: `2c484b07ee94d1366c6c744aa725a6b5da503ca1558f84b9339b1b60bd85a70d`.
- Result SHA-256: `c77c12ac1a690fe984840b342e0dbf5b6b57bfd84320efbbc120f0dba65af46e`.
- Raw-audit SHA-256: `6f4cd54ed3bfc2189ff468f39fae2503817b5c20750c8eb03e6e1d2549e33462`.

## Next scientific decisions

1. Diagnose orientation integration/sampling separately from physical contact.
   The [posthoc numerical diagnostic](go2_active_scan_orientation_diagnostic_development_v2_2026-09-05.md)
   preserves its V1 accounting failure and corrects only grouping; it does not
   revise the above scan outcomes or controller.
2. Model articulated swept volume and uncertainty before assuming in-place
   scanning is feasible beside unseen walls. Compare a locomotion repair,
   positioning in a demonstrably larger observation region, and wider camera
   coverage as distinct future interventions. Do not choose using oracle walls.
3. Build a minimal observation-to-traversal-to-arrival integration experiment:
   choose a current actual proposal, execute a bounded local command sequence,
   and keep arrival/place association provisional until supported by observations.
   Measure false associations and failures using evaluation-only geometry.
   A fully qualified perception stack is not required to run an honestly scoped
   development prototype, but its hypotheses must not be presented as truth.
4. Advance that prototype to fresh small-maze discovery/return before another
   large model sweep. Compare matched direct, supervised and JEPA controllers
   with progress/stall/task outcomes; previous offline scores do not choose a
   scientific winner. Hardware and final generalization remain open.
