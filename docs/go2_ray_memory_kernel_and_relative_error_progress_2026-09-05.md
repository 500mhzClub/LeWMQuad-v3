# Exact ray-query acceleration and explicit relative-error propagation

The previous goal turn was progress: it implemented uncertainty-dependent ray
memory and exposed latency and ground-support limits. This turn improves query
latency without changing clearance decisions, and establishes the mathematical
interface needed to handle shared pose errors. These remain development
prototypes, not newly executed navigation or calibrated uncertainty results.

## Exact query acceleration

The original vectorized-projection/NumPy rectangle reference remains available.
A CPU-compiled reducer uses immutable per-frame 8×8 block summaries: valid-depth
minimum/maximum, complete validity, ground-role eligibility and height extrema.
Fully covered blocks can be reduced exactly. Partial blocks are inspected pixel
by pixel. A depth interval that straddles the near-surface range is ambiguous:
the reducer inspects the underlying pixels rather than inventing an intermediate
surface. Once a non-ground conflict is proved, no later pixel can restore approval.

The cache is not a coarsened map or a relaxed footprint. Projected rectangles,
unknown-pixel treatment, depth/ground predicates and margins are unchanged.
`projected_window_pixels` records the full rectangle area; `examined_pixels`
records leaf pixels visited during the query, excluding per-frame summary
construction. The distinction avoids claiming early-rejected rectangles were
fully scanned pixel by pixel.

An initial cache implementation failed four focused tests: three fixtures changed
raw evidence after cached summaries were built, and one test assumed a particular
early-exit pixel count. The evidence arrays and mapping are now read-only and
isolated from caller input, preventing accidental stale-cache mutation. Tests
construct altered scenes before building summaries and explicitly verify cache
immutability and accurate visited-versus-covered counts. No frozen experiment
source or output was changed.

The implementation uses installed Numba 0.66.0, llvmlite 0.48.0 and NumPy 2.4.6.
Compilation uses CPU only, bounds checking, no fast-math, no parallel kernel and
no persistent JIT cache. Constructor warm-up was about 0.710 s in the tiled probe;
it must happen before the control episode, not within a timed observation step.

## Actual-packet checks and latency

Read-only probe 91257 processed all 457 original north depth-floor-hold packets
and compared compiled and reference results at seven fixed checkpoints. It
checked predecessor bindings before and after. That first compiled version was
faster but still took 234 ms with 64 retained views.

Probe 38897 repeated the complete recorded packet history with the tiled reducer.
All sampled free/support/conflict/unknown decisions and projected-window sizes
matched the reference exactly at all seven checkpoints, including all ground
rejections. Representative query timings are:

| Tick | Retained views | Reference query | Tiled query | Unsupported samples |
| --- | ---: | ---: | ---: | ---: |
| 80 | 20 | 77.4 ms | 15.3 ms | 309 |
| 140 | 28 | 156.8 ms | 23.9 ms | 272 |
| 180 | 29 | 169.8 ms | 25.7 ms | 256 |
| 325 | 53 | 183.3 ms | 33.7 ms | 309 |
| 450 | 64 | 392.0 ms | 55.8 ms | 466 |
| 456 | 64 | 427.7 ms | 60.3 ms | 462 |

These are diagnostic measurements on one reused development trajectory, not a
latency guarantee. Block construction raises memory-observation time to roughly
21–23 ms. A faster query alone does not establish the control deadline.

Probe 68456 additionally recomputed every depth/gyro observer record exactly
from recorded RGBD and fast-gyro packets. The measured sum of depth observation,
fusion/memory update, nominal-volume construction and query was approximately
63.0/74.2/81.6/104.5/107.0 ms at ticks 80/180/325/450/456. The regression suite
was running concurrently. File I/O, camera acquisition, gait execution and the
remaining controller were excluded. End-to-end 100-ms operation remains unproven;
the measured subset alone still exceeds it at two checkpoints.

## What shared pose error actually cancels

`lewm/relative_pose_uncertainty_development.py` implements first-order point
transport with a complete 12×12 joint covariance of current/stored position and
left-rotation errors in their shared reference. It returns the transport Jacobian
and propagated point covariance, without asserting input calibration or navigation
qualification. It does not infer this joint covariance from existing scalar
proxies and is not connected to clearance approval.

Tests establish exact invariance to a shared rigid reference change, cancellation
of its complete joint covariance including position/rotation cross terms, and
agreement of the Jacobian with independent finite differences. The equal-marginal
counterexample matters: with endpoint translation variance σ², the relative
variance is 0 for perfectly shared errors, 2σ² for independent errors, and 4σ² for
opposite errors. Equal endpoint scales therefore cannot justify subtracting
scalar variances or zeroing the transport radius. Missing, marginal-only,
nonsymmetric, indefinite and nonfinite covariance inputs are rejected.

The consumer's absolute-envelope radii are unchanged. No joint sensor/estimator
error model has yet been established, and the earlier bias/jerk limitations are
not repaired by a correct Jacobian.

## Ground-contact semantics checked, not bypassed

The original native contact topology permits ground contact on support links
26–29, the four calf links containing the feet. That is distinct from proof of
ground coverage or association with a retained floor observation. It does not
authorize ignoring an unknown wall, gap, changed surface or unsupported pose.
The 256 rejected ground-role samples at tick 180 remain rejected. Neither the
ground tolerance nor any pose radius was relaxed in this turn.

## Verification and next actions

Focused session 37344 passed 29 ray-memory/kernel tests, including randomized
scenes and queries, exact margin neighbors, ambiguous block extrema and immutable
cache behavior. The relative-error module passed 11 tests in session 46a93e.
Full session 45810 passed 1,274 tests across 117 files in 86.38 s, without source
edits during the run. All diagnostic/test handles are terminal; no experiment or
navigation process is left running.

1. Supply an explicit joint error model through the estimator's actual observed
   and predicted updates, retaining common-reference, velocity, gyro and bias
   correlations. Test it on correlated-error fixtures and separately declared
   development observations before using it to reduce historical uncertainty.
   The original uncalibrated marginal proxies are not that model.
2. Establish supported current/retained floor constraints and their contact-role
   semantics without inventing unseen floor coverage. Keep step/hole, wall,
   missing-ray and bias negatives. Recheck actual blocked samples; do not simply
   relax the 6-cm ground tolerance to make the recorded arrival pass.
3. Complete full-loop latency accounting, including acquisition and controller
   work. If scheduling must change, represent actual acquisition/availability and
   command clocks explicitly; never report stale packets as current.
4. Integrate an explicit fused-speed interface and information-seeking,
   dynamics-aware observation actions into a fresh navigation successor. Preserve
   native stops and task metrics. Couple turning data and action-choice work to
   matched geometric/empirical, supervised and JEPA predictors; then execute
   complete missions and the required memory/multistep/independent-layout tests.

Whole-task success remains 0/2. No learned-navigation, JEPA-benefit, independent-
maze or hardware result was produced here. The full scientific goal remains
active and unachieved; software and simulation work remain available.
