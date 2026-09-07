# Observed floor footprints and joint pose/floor error

The previous goal turn was progress: it established that gyro error must be
propagated through depth registration, not only orientation integration. This
turn connects those shared sensor sources to observed floor patches and exposes
a separate footprint weakness in the existing ground predicate. No navigation
consumer, frozen experiment, clearance radius or acceptance tolerance changed.

## New diagnostic geometry

`lewm/correlated_floor_evidence_development.py` measures the signed height of a
query point along observed up relative to two adjacent-pixel surface triangles.
It requires the point's projected footprint to lie inside an observed triangle.
The four pixels must be valid, below the ground-role height, near horizontal
(normal/up dot at least 0.97 in magnitude) and locally planar within 3 mm.
Locating the footprint follows at most four local projections and retains the
existing 6-cm height band. Missing observations, walls, steps, degenerate patches
and out-of-view queries are not replaced by an infinite fitted floor plane.

A synthetic missing-data test demonstrates why this distinction matters. For a
ground-role query 3 cm below a flat floor, the old ray/height predicate still
accepts nearby floor returns after the four pixels under the query's vertical
footprint are removed. The new relation rejects that footprint. This demonstrates
missing evidence, not the physical existence of a hole. The old predicate and
its completed navigation outcomes remain unchanged; the counterexample is a
required negative for any successor.

The paired wrapper retains the actual raw depth, declared range-error loadings,
and each paired estimator's pose and gravity-derived up. It propagates the same
source through registration, pose transport, floor range and normal geometry.
Only tapped patch pixels are materialized for a query. A height sensitivity is
reported only if the nominal and every small perturbation pair retain observed
footprints. Finite pairs do not cover a continuous uncertainty region, so every
result explicitly keeps `whole_error_envelope_covered` and
`ground_support_approved` false.

An analytic shared-height fixture verifies cancellation between point and floor
errors; falsely treating those factors as independent reports nonzero variance.
Conversely, unsupported pose-only/floor-only comparisons return no variance,
even when the jointly perturbed patch and point remain mutually observed.

## Recorded arrival evidence

Read-only probes used the first 181 original north depth-floor-hold observations,
without masking or changing the nominal packets. Three explicit hypotheses were
used: a persistent 0.001-rad/s body-roll gyro-bias source, a 0.02-m/s² body-y
accelerometer-bias source beginning at tick 80, and a shared 0.002 relative depth
range-scale source. These are assumed source scales, not sensor calibration.
Both difference steps, 0.01 and 0.005, recomputed full raw registration and fusion.
All 181 nominal fusion records matched the unchanged ray-memory estimator.

The original memory selected the retained views; the most recent view with a
valid local paired footprint was used for each reported height. No view was
selected to minimize its variance. Results from completed strict probe 79982:

| Tick | Old unknown ground samples | With measured nominal and paired footprints | Assumed joint height standard deviation, median / max |
| --- | ---: | ---: | ---: |
| 80 | 266 | 183 | 1.145 / 3.452 mm |
| 140 | 272 | 272 | 1.975 / 4.569 mm |
| 180 | 256 | 256 | 2.598 / 6.015 mm |

At tick 180 the observed nominal heights range from −36.99 to +40.70 mm.
The three-source vertical response is much smaller than the old isotropic
transport allowance. This identifies a concrete route for an anisotropic
pose/floor model, but does not establish that the omitted errors are small or
that an entire uncertain footprint is observed. At tick 80, 83 of the blocked
ground-role samples still lack a located footprint and remain unresolved.

The initial exploratory probe 2835 mixed different most-recent valid views in
some step-size comparisons and reported split variances even when their separate
footprints were unsupported. Before the strict probe, comparisons were changed
to require the same view, and unsupported split variances were masked. The strict
probe excludes 7 and 10 changed-view selections at ticks 140 and 180. Maximum
same-view factor disagreement is about 3.66 micrometres per declared source;
the initial mixed-view maximum of 0.117 mm is not a derivative-stability result.
Four and two unsupported independent comparisons are omitted at those ticks.

On this trajectory the independent and joint height sensitivities are often
similar. The data do not show that large floor/pose cancellation is the dominant
benefit; the distinction between directional height error and an isotropic
position allowance is important in its own right. The analytic cancellation
test establishes a possible correlated case, not its prevalence here.

## Cost and scope

The strict pair of raw observer diagnostics (two step sizes, seven observers
each) took median 210.6 ms and maximum 322.8 ms per observation. Querying both
paired floor histories took approximately 344/512/523 ms at the three checkpoints.
The regression suite was running concurrently. Acquisition and normal controller
work are excluded. This is an offline reference, not a real-time implementation.

Both probes verified all 333 predecessor source bindings and the bound inputs and
artifacts before and after processing. They printed diagnostic results without
creating an experiment output directory or changing any original physical score.
Focused session 45004 passed 23 tests, including footprint, missing-data, shared
error, quantized raw-depth, fault-latch and unsupported-comparison negatives.
Full regression session 43308 passed 1,325 tests across 119 explicitly selected
files in 88.15 s with no concurrent source edits. All diagnostic and test handles
are terminal; no navigation, training or experiment process remains running.

## Next implementation steps

1. Derive an efficient joint update, using the raw paired reference to test
   registration, gyro, depth-scale, gravity, velocity and bias derivatives.
   Keep rank/matching changes explicit; do not fall back to the already disproved
   fixed-registration model. Profile the complete control path rather than
   presenting offline paired-query timings as deployable performance.
2. Establish the validity and magnitude of the sensor/estimator error model on
   separately declared development inputs. These three source columns do not
   cover every bias axis, independent range noise, residual registration bias,
   motion distortion or unknown sensor calibration. Small finite-difference
   agreement does not prove coverage at one or three source standard deviations.
3. Combine directional height uncertainty with observed coverage of the entire
   projected footprint region. Preserve missing-pixel, thin-wall, step, hole,
   out-of-view and large-bias negatives. Do not replace the old isotropic radius
   by the reported height standard deviation everywhere: lateral ray coverage
   and floor-height error are different constraints.
4. Integrate the supported successor into the explicit fused-speed and
   dynamics-aware observation controller, then run fresh complete discovery and
   return missions. Follow with matched supervised/JEPA action predictors,
   memory ablation, genuine multistep online rollouts and independent layouts.

Whole-task success remains 0/2. No learned-navigation, JEPA-contribution,
independent-maze or hardware result was produced. The full scientific goal
remains active and unachieved.
