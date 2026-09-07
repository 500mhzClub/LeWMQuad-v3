# Exact shared-work acceleration for joint sensor-error propagation

The previous goal turn was progress on observed floor footprints and coupled
pose/floor errors. This turn reduces the cost of the raw paired uncertainty
reference while preserving its outputs. No frozen experiment, navigation
controller, error-source scale or clearance decision changed.

## What is reused—and what is still recomputed

`lewm/certified_registration_reuse_development.py` provides a traced registration
implementation, a shared-observation paired wrapper and a lean paired wrapper.
The registration retains the original nearest-distance/normal gates, Huber
weights, eigenspace selection, least-squares iterations, stopping rule, rank and
acceptance criteria. It does not freeze translation, weights or weak directions.

For each nominal query, the trace stores immutable target/query coordinates,
explicit pixel-lineage IDs and the nearest competing target's distance. If the
query moves by at most dq and any target moves by at most dt, every competing
distance is at least d_second − dq − dt. The nominal nearest target can therefore
be reused when its actual perturbed distance is strictly below this lower bound,
with an additional coordinate-scaled numerical guard. Ambiguous queries still
use the ordinary tree search. Ties and the strict 0.2-m search boundary are
tested. Changed point lineage triggers a full search, not an assumption that
equally sized point arrays represent the same pixels. Extra iterations also
perform the normal search when no nominal iteration trace exists.

Identical depth observations can share their extracted surface cloud. Each
perturbation branch still validates and advances its own sensor history. Cached
data are copied/immutable, content-bound and scoped to the current observation.
Pixel IDs already in increasing order avoid repeated hash-based uniqueness
checks; reordered IDs still receive a full uniqueness check, including unsigned
duplicate-ID negatives.

The lean wrapper omits body-height wall/portal reports only for perturbation
branches: the moment integrator does not consume those features. It still runs
the complete dense surface-cloud registration, gyro, gravity and fusion updates
on their perturbed packets. Identity, clock, image binding, depth validity and
history checks remain. The full nominal surface report is preserved for future
navigation consumers; the pair-only metadata object must not be substituted for
a nominal wall/portal observation.

## Recorded comparison

Read-only probe 54122 compared the initial shared-work implementation against
the original raw paired observer. It completed 181 original north packets and
181 packets under the previously declared narrow-depth mask at ticks 80–139.
Every public paired output and sampled retained/current point-covariance output
matched exactly. Nominal median observer time fell from 99.2 to 77.7 ms; the
narrow-depth medians were 99.5 and 77.2 ms.

Profile 96538 processed the original 181-packet history and profiled its final
observation. It identified repeated wall-report construction, dictionary/list
copying and ID uniqueness checking as substantial costs. The profiled 118-ms
total includes profiling overhead and is not a latency measurement comparable
to the unprofiled times. This evidence motivated the lean wrapper and sorted-ID
fast path, not removal of causal or registration checks.

Final probe 66171 repeated the exact comparison with the lean wrapper. Both
conditions use the same three explicit source hypotheses as the preceding floor
diagnostic: 0.001-rad/s roll-gyro bias, 0.02-m/s² body-y accelerometer bias from
tick 80, and shared 0.002 relative range scale, at difference step 0.01.

| Recorded condition | Frames | Original median / max | Lean median / max |
| --- | ---: | ---: | ---: |
| Original north prefix | 181 | 102.44 / 133.46 ms | 52.95 / 97.14 ms |
| Narrow depth at ticks 80–139 | 181 | 101.21 / 151.38 ms | 52.02 / 102.25 ms |

Public outputs match exactly at every frame; retained/current joint point
moments also match at ticks 79, 93, 140 and 180. No observer fault was introduced.
The original weak-case proxy-budget failure at tick 93 remains unchanged; all
later outputs are diagnostics, not authorized continued motion.

For the original prefix, 8,783,348 ordinary nearest queries were certified for
reuse and 1,474,515 were searched. Collecting the references additionally required
1,465,409 nominal two-neighbour queries. The narrow-depth counts were 6,088,404,
1,023,897 and 1,016,043 respectively. The extra reference searches are real work
and must not be omitted when interpreting the speedup. No lineage fallback was
needed in these actual cases; synthetic changed-lineage tests exercise that path.

The comparison alternated implementation order. The regression suite overlapped
part of the diagnostic. These are two conditions on one reused development
trajectory, not independent maze trials, a timing guarantee or sensor calibration.

## Verification and remaining work

Focused session 95592 passed 33 tests: nearest-query perturbations, ties, boundary
values, stale cache/tree and ID errors, full/weak registration, changed surfaces,
Huber reweighting, exact pixel-cloud selection, raw-pair output equivalence and
causal sensor faults. Earlier focused session 53142 passed 28 tests before the
lean extension. Full final session 43529 passed 1,358 tests across 120 explicit
files in 90.30 s with no concurrent source edits. The earlier full session 90661
passed 1,353 tests before the final extension.

Both comparison probes and the profile checked all 333 predecessor source
bindings and bound inputs/artifacts before and after processing. They did not
create experiment outputs, modify physical scores, inspect sealed material or
change any completed source binding. All handles are terminal.

Next actions:

1. Reuse the lean paired observer and its full nominal observation in the floor/
   navigation integration so the nominal depth estimator is not unnecessarily
   executed twice. Keep its paired metadata separate from the full nominal
   surface evidence.
2. Accelerate the coupled floor-footprint query and establish coverage of the
   entire uncertain footprint, not just the small differentiation pairs. Keep
   missing-pixel, thin-wall, step/hole and out-of-view negatives, and keep lateral
   projection error distinct from vertical floor-height error.
3. Validate source magnitudes, missing error modes and finite-error behavior on
   separately declared development observations. Exact computational reuse does
   not establish calibrated covariance or repair the existing bias limitations.
4. Complete full-loop timing and integrate explicit fused speed and dynamics-
   aware observation actions into a fresh complete-mission simulation successor.
   Then compare matched supervised/JEPA action predictors, memory and genuine
   multistep rollout contributions on independent layouts.

Even this observer's measured worst case reaches 102.25 ms before acquisition,
floor queries, gait execution and remaining control work. The 100-ms full-loop
deadline is not established. Whole-task success remains 0/2; learned-navigation,
JEPA benefit, independent-maze and hardware requirements remain unproved. The
full scientific goal remains active and unachieved.
