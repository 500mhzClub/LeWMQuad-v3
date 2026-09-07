# Previous-measured-pose initialization: fixed development replay V1

The frozen rotation-only correspondence experiment failed early in all six
recorded trajectories:471/7,433 available poses versus7,400 for the original.
Do not modify or relabel it. This distinct method tests whether a translation
initialization derived from the last accepted observation improves image-track
capture range while retaining all current-measurement acceptance checks.

## Fixed method

For each retained reference, use its observed rotation/position R_ref,p_ref and
the last accepted position p_last to form t_seed=R_ref.T*(p_last-p_ref).
Use current measured relative gyro rotation G to project reference points as
(a-t_seed)@G into the current body, including the existing calibrated camera
lever arm. The previous accepted pose must be the immediately preceding100ms
observation; future, stale or missing seed provenance is not allowed. Retain
the seed's source time and reference-frame vector in per-attempt diagnostics.

This is a numerical initialization, not a current-position output, velocity
extrapolation, command-derived prior, rigid-fit constraint or additional sensor.
Current RGB-D observations must support the returned position. Keep reference
features, bounded half-pixel deduplication, bidirectional LK/0.5px round-trip,
depth validity/discontinuity checks, robust registration/reprojection, inlier
fraction, six-cell support, increment gates, primary-first recent references,
promotion and terminal latching unchanged from the frozen flow method. No
alternate method search, pose reset, reference replacement by native truth,
threshold tuning, post-failure extrapolation or physical-control integration.

The original descriptor-based association remains a distinct baseline. Both flow
methods replace mutual descriptor association and descriptor-endpoint proximity;
retaining final rigid gates does not make their association reliability equal.
Periodic-image ambiguity, depth bias and incorrect previous poses remain risks.
Neither a warm start nor low residuals establish calibrated uncertainty.

## Synthetic evidence required before recorded execution

Multi-frame images/depth must generate the seed from the previous *estimated*
observation, never from fixture truth. Cover the keyframe translation range,
positive/negative translation, depths0.7/2/3.5m, an occluding nearer panel and
combined rotation/translation. Preserve full trajectories and failures for
original, frozen rotation-only and measured-pose initialization. Include invalid,
wrong and stale priors, unknown/blank imagery, clock/identity/privacy-contract
faults, zero-seed equivalence, seed correction by new observations and retained
periodic-ambiguity counterexamples. These are synthetic, not physical trials.

The fixed seven13-frame comparisons have already completed: original91/91,
rotation-only59/91, measured-pose91/91; maximum measured-pose error0.215mm.
The current21 matcher/runtime tests pass. These results justify testing the
recorded population; they do not justify adoption or establish sensor realism.

## Full recorded comparison and frozen baseline

Run original and measured-pose observers on all three inner-arrival and all three
persistent-intent recordings:7,433 frames,14,866 frame/arm observations. Verify
the complete original registration replay, raw/native/source bindings and the
terminal rotation-only replay. Inherit the latter's source closure and bind new
sources explicitly. Every original pose, selection and failure must exactly
match the original registration witness, including all drain and terminal no-op
frames. Never truncate a trajectory when the candidate loses tracking.

Do not rerun the failed rotation-only method. Preserve its exact result and all
output identities as the third-method baseline. At completion, verify that its
whole original trajectory populations/first failures match this run and retain
its existing reports/evaluations in a separately identified comparison artifact.
Do not treat old rotation-only timings as contemporaneous measurements.

Persist and bind all six paired sensor streams before parsing native coordinate
arrays. Evaluate exact samples749+50*frame with verified acquisition timestamps,
position and orientation errors in the initial body frame. Report full
both/original-only/measured-pose-only/neither availability, first failures,
available-arm errors and paired common-frame errors. Native coordinates and
evaluation metrics never enter either observer, seed, reference decision or
output pose. No thresholds or method selections depend on these results.

This is reused development data, not independent validation or closed-loop
recovery. Changed control would change future observations. Timing is untraced
observer-only, excludes packet loading and control, uses original-first sequential
calls and may overlap collection. Terminal no-ops are separate from available
observations. No real-time, hardware or learned-navigation benefit is claimed.

## Bounds and next decision

One exclusive output:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_pose_correspondence_replay_v1_attempt_001`.
Keep the2GiB output allowance,40GiB free-space reserve, bounded metadata/rows,
exclusive durable writes and complete source/input/output checks. Retain failure
and partial artifacts on any fault; no retry, resume or replacement attempt.
Use the existing development venv with PYTHONDONTWRITEBYTECODE=1,
PYTHONHASHSEED=0, PYTHONPATH=.:lewm_genesis:lewm_worlds, OMP/MKL/OPENBLAS threads1
and OpenCV threads1. No GPU, new physics, model fitting or hardware operation.
Preserve the running771-source collector, reviewed786-source matched-study
definition and completed691-source failed flow experiment.

After complete verified results, decide whether to proceed to independent scene,
sensor-error and end-to-end latency challenges. Even success here is not permission
for physical adoption. If this fails, retain it and reconsider complementary
sensing or explicit handling of temporary visual unavailability, without silently
resetting coordinates or selecting weaker gates. Continue collecting all12 layout
receipts for the planned36-fit JEPA/supervised/ablation comparison. Reliable local
execution, online rollouts, memory/backtracking, unfamiliar-maze missions,
deployment-valid sensors, real-time operation and bounded hardware evidence
remain the full unfinished goal.
