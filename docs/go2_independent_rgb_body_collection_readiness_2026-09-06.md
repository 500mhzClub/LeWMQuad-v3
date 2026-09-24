# Next: independent RGB/body prediction data, separate from depth navigation

## Scientific decision and evidence

Do not make full depth-based navigation qualification a prerequisite for testing
RGB/body prediction. These remain different claims under the same final goal.
The earlier acquisition plan already distinguishes their eligibility; the new
terminal-event contract and native edge assay now make that distinction concrete.

Source inspection confirms `lewm/rgb_body_tensor_interface_development.py`
constructs only RGB, gyro/specific-force/joint history, validity/age and applied-
command tensors. `lewm/pulse_timed_observation_pairing_development.py` accesses
the policy element of each replay packet for both history and future targets;
depth and the separate fast-gyro packet do not enter these learned tensors.
`lewm/pulse_timed_dataset_development.py` joins native motion/contact labels
target-side only. The fixed inventory collector's commands depend on prescribed
action/history and causal packet clocks, not on the shadow depth tracker or
native poses. Thus a foreground-boundary centre-ray depth residual is not, by
itself, evidence of corrupt RGB/body learning inputs or native labels.

This is an input-dependency inference from the source, not a new learned-model
result. The new `test_rgb_body_depth_separation_development.py` now verifies it
through the actual dataset materializer:policy packets and native target labels
are fixed, depth changes from valid toNaN/invalid and shadow tracking changes
from available tofailed. All learner input and target tensors remain exactly
equal, including positive contact labels. The replay tuple test double raises
if the materializer accesses any non-policy element. Combined12311 passes47
tests in2.33s (this new test plus31 coverage and15 edge-assay tests). This proves
the tested dependency boundary, not data or sensor validity. Preserve actual raw
packet and native label audits; do not bypass them on this argument.

Requiring every strict centre-ray depth score to pass would select against
foreground silhouettes. Those scenes are precisely relevant to obstacle/branch
prediction. Conversely, ignoring all sensor errors would be unjustified. Freeze
the following modality-specific rules before collecting fresh data.

## Implement the fresh independent-layout collector and auditor

1. Use distinct paths/output roots, the fixed existing 12-layout inventory and
   its6/3/3 train/selection/development-evaluation roles. Collect120 cases per
   layout:all6 action-duration cells,5 contexts,2 histories and2 supports. Do
   not train on or relabel the completed sensor pilot or old failed batch.
2. Reuse the verified union scene, floor-first rendering and core-profile query.
   Retain the physical setup, native guards, exact commands, sensor recording
   and contact stops. Integrate full raw reconstruction plus terminal-event
   coverage at every committed case and terminal-audit each bounded layout.
3. Define RGB/body prediction acquisition eligibility from faithful complete
   schedule or physical-terminal observation, actual action departure, complete
   causal history, exact packet reconstruction and valid target censoring.
   Reject infrastructure/corrupt cases without treating them as safe negatives.
   Keep every planned case and reason in the public development denominator.
4. Retain all strict depth scores, finite-pixel partitions, actual native order
   and precision witnesses. Stable-interior errors and below-near/false-valid
   occlusion failures remain hard measurement failures to investigate. Do not
   turn boundary values into corrected depths or policy free-space evidence.
   Strict centre-ray boundary-only failures may coexist with RGB/body prediction
   eligibility under this new prospective contract; they remain failures in their
   own metric. No post-hoc predecessor eligibility change is permitted.
5. Record all same-context action-prefix comparisons, including unequal and
   unavailable pairs. These determine paired counterfactual evidence separately
   from individual supervised sample coverage. Do not select favorable episodes
   or geometries using tracking/mission/contact outcomes.
6. Freeze the complete source/target/schedule/eligibility contract and resource
   budget before any fresh physics. Begin the first120-case training-layout
   batch, audit it completely, then proceed through the fixed remaining layout
   population as storage and valid acquisition permit. No fits until declared
   train/selection/evaluation data coverage is known and the study is frozen.

## Continue toward the unchanged final goal

After independent collection, run matched direct/supervised-rollout/JEPA arms
with3 paired seeds, shared exposure and task labels, action/time and zero-motion
baselines, RGB/history/action ablations and independent-layout effects. Existing
same-room heads still lose the empirical baseline; no utility is assumed.

In the navigation track, boundary uncertainty and near-range invalidity must
remain conservative. Analytic thin-obstacle/near-plane counterexamples, runtime
free-space semantics, articulated stopping/turning, online memory/backtracking
and real-time execution remain required before any maze-navigation qualification.
The present camera hides the robot and the body sensors are ideal:hardware
self-occlusion, noise, latency, calibration and bounded supervised real-platform
evidence remain deployment work. Advancing the prediction experiment does not
shrink or discharge these final-goal requirements.
