# Observed exploration integration and measured translation baseline

This step adds a real runtime connection between observed symbolic exits and
the existing local controller, plus measured evidence about translation. It does
not supply qualified RGB place/exit/beacon detection or demonstrate full-maze
navigation. Earlier JEPA negatives and the bounded positive successive-control
result remain unchanged.

## Runtime work implemented

`lewm/memory/observed_exploration_development.py` maintains observed places,
untried exits, discovery identities and complete traversal-attempt history.
It uses `DirectedTraversalGraph` only for executed, reached, stable, viable,
association-confirmed directed traversals. A selection creates an outstanding
attempt, not an edge. An ambiguous arrival requests localization and cannot be
retroactively promoted by a later fix. A conflicting destination quarantines
the exit and disables the old route; a failed latest traversal is retained and
not automatically retried. Re-observing the same exit does not erase its failure.

Local observed frontiers are explored first, then reachable remote frontiers
using measured directed-route costs. Return uses only verified directions. If
an outward edge has no executed reverse, the controller searches remaining
observed frontiers for a return path; it never invents an inverse edge. Repeated
beacon detections do not multiply the discovery count. Unknown association,
unstable arrival, stale observations and stale body-frame exit bearings produce
explicit non-traversal decisions. Internal mission completion requires the
requested distinct observed beacon count and a current stable home association;
that is conditional on the upstream detector's reports, not scientific proof
that its identities are correct.

`lewm/exploration_local_bridge_development.py` connects a fresh observed exit
bearing to the existing `OnlineTemporalChoice` direct/supervised/JEPA heads.
It retains actual past packets continuously, initializes each traversal's fresh
relative direction frame from the last four actual packets, and requests the
unchanged five-candidate half-second local decision. Bearings must belong to the
actual current packet. It never constructs a past image from a later sensor
buffer, installs an oracle route or qualifies an arrival from model prediction.
Errors latch and require explicit executor stopping. Ending a failed attempt
does not clear a sensor fault. The cue is still a direction, not a metric exit
position; this bridge does not yet use the translation baseline below.

The25 memory tests cover discovery/return, directed asymmetry, remote frontiers,
uncertain and conflicting revisits, failures and chronology. Ten bridge tests
exercise actual adapter code, including all three synthetic model members for
direct, supervised and JEPA paths. Synthetic perception/arrival events test the
integration contract; they are not evaluated RGB detectors or physical routes.

## Translation: evidence rather than assuming commands equal motion

`lewm/causal_command_odometry_development.py` integrates the actually applied
velocity command with the existing causal relative-gyro estimate. The command
sample at t describes the interval ending at t; using the preceding sample would
add a one-tick switch/braking lag. Rotation uses an endpoint trapezoid over each
100-ms interval. There are nine causal clock/identity/history/fault tests.

The estimator reads no world pose, actual velocity, map or future command. But
the command is not measured body velocity. Its output explicitly remains
`metric_translation_qualified: false`; no fitted scale, bias or covariance is
claimed. A relative-point operation subtracts accumulated estimated translation
before rotating into the current body frame, ready for a separately evaluated
fixed-point controller. It is not yet used to associate places.

### Completed short replay

The [fixed144-stream replay](go2_causal_command_odometry_development_v1_2026-09-05.md)
completed in session89352, exit0:5,928 post-initial packet updates, with45 native
contact terminal images explicitly excluded. The old raw-audited physical bytes
were verified; no new physics, checkpoint fitting or policy selection occurred.
True displacement was used only by the separate quaternion-vector evaluator.

| Source controller | Mean last-precontact control xy error | Worst such error | Observed fixed4-s endpoints /24 | Mean fixed4-s xy error |
|---|---:|---:|---:|---:|
| Stop | 0.45 cm | 0.57 cm | 24 | 0.45 cm |
| Direct | 2.86 cm | 7.71 cm | 17 | 2.89 cm |
| Supervised direct | 3.25 cm | 6.44 cm | 16 | 3.20 cm |
| Supervised latent | 3.51 cm | 7.31 cm | 14 | 2.74 cm |
| JEPA direct | 3.14 cm | 5.85 cm | 11 | 2.35 cm |
| JEPA latent | 3.65 cm | 6.00 cm | 21 | 3.52 cm |

Last-precontact durations differ (means3.45–3.91 s for learned controllers);
the fixed4-s column has explicit censoring. A release contact can leave a valid
4-s endpoint, so these counts differ from complete-control-plus-release counts.
Methods visit different states: these are not causal method comparisons or
independent144-maze evidence. The zero-translation comparison's mean endpoint
errors are80.9–90.0 cm for learned-controller streams, so retaining short-range
command odometry as a baseline is useful. Low short-horizon error does not prove
long-term place localization.

Result SHA-256:
`947efe6dcfb2143386d1aba8dec496b93b8de915fe96d36a7c8d0e4db632a64b`.
Launch SHA-256:
`fcffc4323df1811ba7099d7a3a7613ddec38a0e8acadefe57e26a9988d01c38b`.
Root: `.generated/go2_causal_command_odometry_development_v1_attempt_001`.
The estimator, its test, runner and fixed protocol are now bound: do not edit or
rerun them as the same experiment.

### Completed longer-route and turn replay, without refitting

The [separate26-stream extension](go2_route_turn_command_odometry_development_v1_2026-09-05.md)
completed in session43057, exit0. The same unchanged estimator consumed2,826
updates across all eight old multi-junction routes and18 old turn trials. No
reset occurred between route edges. All original physical task failures remain.

| Route | Width | Observed duration | Endpoint xy error |
|---|---:|---:|---:|
| Left-right | .9 m | 16.3 s | 3.68 cm |
| Left-right | 1.2 m | 19.0 s | 7.43 cm |
| Right-left | .9 m | 16.5 s | 11.13 cm |
| Right-left | 1.2 m | 19.0 s | 13.14 cm |
| Hairpin | .9 m | 15.8 s | 5.21 cm |
| Hairpin | 1.2 m | 19.1 s | 5.47 cm |
| Dead-end return | .9 m | 19.9 s | 5.67 cm |
| Dead-end return | 1.2 m | 23.4 s | 7.25 cm |

Across routes, mean endpoint error is7.37 cm and maximum observed error anywhere
is13.80 cm. Across18 nominal zero-translation turns, endpoint errors range4.59
to9.48 cm (mean6.70 cm). That physical drift is absent from a command-only
translation signal. All26 streams were on-clock and contact-free, as the source
audits stated; none was omitted. These descriptive development maxima are not
calibrated confidence bounds and must not become post-hoc place-association
thresholds. The eight routes are four motifs at two confounded width/spawn
settings, not eight independent random maze draws.

Result SHA-256:
`46a25db62091b935df95d4f94ac11232a6354b9d672cdf83f0b9c004b0d79b3c`.
Launch SHA-256:
`5e0347daecd6607ac66e425f2d473c3861c6e02f6eada944ba227a6b26eca93e`.
Root: `.generated/go2_route_turn_command_odometry_development_v1_attempt_001`.
Its new runner/protocol and predecessor identities are now frozen by execution.

## Next implementation decisions

1. Continue the already live384-trial moving-prefix collection (session1894),
   then full raw audit and actual composite-tensor validation. Do not rerun or
   change its source. It reached191 persisted trial artifacts at this checkpoint.
2. Exercise the new bridge with fitted checkpoints on actual recorded packets,
   comparing input/prediction identity to the original adapter. This checks
   integration only; symbolic exit labels in replay are not RGB observations.
3. Implement and evaluate RGB exit, place and beacon observations, including
   explicit ambiguous/repeated-looking junctions and missed/wrong revisits.
   A viewed old route image has extremely low texture and repeated neutral walls;
   neither visual uniqueness nor metric monocular scale may be assumed.
   Keep supervised geometry labels evaluation/training-only, never runtime input.
4. Use the measured command odometry as the baseline in short fixed-point
   control; compare it with visual/inertial or joint-kinematic correction on
   continuous turns and revisits. Do not gate place merges on uncalibrated
   commanded-motion point estimates. Do not silently combine a new control/loss
   change with the action-coverage training intervention.
5. Integrate physically observed arrival and directed return, then evaluate
   actual hidden-beacon exploration with/without persistent memory. Freeze the
   methods only after development, then test independent layouts/shifts and
   supervised real-Go2 transfer. None of this final evidence exists yet.
