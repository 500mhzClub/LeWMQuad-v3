# Shared nominal observation and supplied-bound floor coverage

The previous goal turn made progress on exact paired-observer acceleration. This
turn exposes that observer's full nominal depth evidence to both ray and floor
consumers, and adds region coverage beyond small perturbation samples. No frozen
experiment, physical controller, pose radius or ground tolerance was changed.

## Single nominal depth observation

The lean paired observer now retains a copy-isolated full nominal depth record
and exports it only for the current successful observation. The shared floor
wrapper supplies this record alongside its joint sensor sensitivities. Nominal
wall/surface evidence is preserved; pair-only metadata is never exported as the
nominal navigation observation. Stale or failed histories cannot export an old
record. A test counts exactly one nominal depth-observer invocation per packet,
even when the ray consumer and multiple snapshot readers use the result.

The existing ray consumer still performs its own inexpensive moment-fusion
update and checks its budget. Its fusion outputs equal the shared observer's
outputs; expensive nominal depth registration is not run a second time. Under
synthetic prolonged weak geometry, the exported depth record stays rank 2 with
no full displacement/cumulative observed position, while the separate fusion
output identifies prediction. The ray consumer still latches its required
budget stop (tick 15 in that fixture); no rank or usable-state flag is fabricated.

## Coverage of an explicitly supplied region

`lewm/floor_footprint_bounds_development.py` builds an immutable summed-invalid-
cell index over the measured adjacent-pixel two-triangle floor mesh. Each cell
must retain all four valid returns, the existing ground-height condition,
normal/up alignment and 3-mm planarity requirements. Interior invalid cells
cannot be skipped merely because boundary or small perturbation samples pass.

For a supplied body-point box Q, signed-height interval H within the existing
±6-cm band and fixed observed up u, it encloses the projection of every q − h*u
for q in Q and h in H. The 16 vertices provide projection extrema when optical
depth stays positive; out-of-camera regions fail. A numerical outward allowance
includes cells across exact pixel-grid boundaries. The whole enclosing rectangle
is checked by its invalid-cell sum, including missing interior pixels.

This establishes coverage only of the supplied projected rectangle on the nominal
interpolated mesh. It does not establish the error bounds, uncertainty in up,
uncertain surface geometry, the height/contact constraint itself, or a physical
continuous-scene certificate. All approval/calibration flags remain false.
A synthetic missing-cell test explicitly has both endpoint footprints observed
while the full region correctly fails.

## Actual-packet integration

Completed probe 66014 processed the original north prefix of 181 packets. Every
serialized nominal depth record exactly matched its saved predecessor, and the
ray/floor consumer fusion records agreed at every step. The first probe 84383
stopped at its comparison because live identities are tuples whereas saved JSON
identities are lists. Diagnostic 63280 confirmed identical serialized values;
the comparison was corrected to exact JSON-record equality, not a numerical
tolerance, before probe 66014. No physical result was rerun or rescored.

The shared observer used the preceding three declared error-source hypotheses.
For this separate coverage check, point boxes used the unchanged historical
transport-radius proxy; height intervals used the complete existing ±6-cm band.
Up and the measured surface were fixed. These are not newly calibrated bounds.

| Tick | Old unknown ground samples | Covered supplied rectangles | Region-check time | Measured pipeline subset |
| --- | ---: | ---: | ---: | ---: |
| 80 | 266 | 154 | 7.10 ms | 138.70 ms |
| 140 | 272 | 263 | 14.13 ms | 155.91 ms |
| 180 | 256 | 252 | 14.34 ms | 157.40 ms |

At tick 180, the index represents 12,479,377 queried cells across 30 views; this
is a summed-area query count, not a claim that those cells were individually
rescanned at query time. Median shared-observer/ray-update/index-construction
times were 54.36/22.29/29.62 ms. The pipeline subset additionally includes nominal
turn geometry and the original ray query, but excludes camera acquisition, gait
execution and the remaining controller. The regression suite overlapped the
probe. The 100-ms full-loop deadline is still not met.

## Four remaining regions

Read-only diagnostic 39682 reproduced the four uncovered tick-180 ground samples:
217, 218, 226 and 227. Each has 13 or 14 camera-complete views, so these are not
simply outside all camera views. Their least-invalid candidate rectangles contain
2, 2, 12 and 78 non-ground cells respectively, with zero missing-depth cells.
The corresponding radii are about 20.95 mm in that retained first view.

Direct first-view diagnostic 1109 found that every one of those cells fails at
least one triangle's 0.97 normal-alignment requirement. Some also fail the 3-mm
planarity requirement; maximum plane discrepancies are about 5.0–9.1 mm. This
is evidence of steep/discontinuous returns within the enclosing rectangles,
not permission to bridge them and not proof of a physical collision.

The independent Q×H construction is conservative: it combines point and height
extremes that a coupled floor relation may rule out. A next construction may use
that coupling only if its enclosure is justified and checked against all observed
cells. It must not crop away the offending cells solely to make these four
samples pass, nor call a tighter unvalidated covariance a physical bound.

## Verification and next actions

Focused session 16605 passed 19 tests, including exact shared observer/ray/floor
outputs, snapshot isolation, partial-depth semantics, budget stops, interior
missing cells, projection containment, monotonic box enlargement and invalid
bounds. An initial bounds-shape typo caused 15 focused failures and was corrected;
a misplaced test block caused one further fixture failure and was corrected
before final verification. Full session 82076 passed 1,377 tests across 122
explicit files in 91.51 s with no concurrent source edits.

The successful integration and uncovered-region probes checked all 333 predecessor
source bindings and their bound inputs/artifacts before and after processing.
No experiment directory was created and original results remain unchanged.
All test and diagnostic handles are terminal.

Next actions:

1. Construct and validate coupled point/floor footprint enclosures, including
   explicit up and surface-error bounds. Use the paired reference as a diagnostic,
   not as proof that small perturbations cover a finite uncertainty region.
2. Address the four non-ground-region negatives by a justified geometric
   enclosure or additional observation/repositioning. Do not weaken cell
   eligibility, erase negative returns or assume an unobserved flat floor.
3. Reduce duplicate per-frame geometry/index construction and finish complete
   timing. The summed-region query is fast, but the measured pipeline subset is
   still 139–157 ms before acquisition and remaining control.
4. Integrate supported floor handling and explicit fused speed into a fresh
   dynamics-aware observation/navigation successor, then execute complete
   discovery/return missions and matched supervised/JEPA action, memory and
   genuine multistep-rollout comparisons on independent layouts.

Whole-task success remains 0/2. No learned-navigation, JEPA contribution,
independent-maze or hardware result was produced; the full goal remains active.
