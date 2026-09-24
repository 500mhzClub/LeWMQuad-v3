# Matched action/time baseline: posthoc development diagnostic V1

Fit one deterministic empirical predictor to the same 72 scheduled draws used
by all nine completed pulse-timed neural fits. Repeated draws keep their weights;
no additional examples, optimization, neural resume, physics or navigation.
Average XY and circular sin/cos yaw separately in each exact action-index and
target-offset-nanosecond cell. Average observed contact labels, clipped to
[1e-6, 1-1e-6] solely for a finite logit. No tuning, interpolation, cross-action
fallback, future-image inference or evaluation-target fallback.

Compute query times from the known command plan, including partial 2.2 s blocks,
not observed outcomes. Fit only train-role rows and observed target masks. Score
all 185 eligible train-role windows using the frozen reducer and unchanged
917 observed motion/contact targets. Report all six actions, actual offsets and
three recorded conditions as diagnostics, not independent statistical units.
If any queried motion/contact cell is unavailable, publish missing-cell indices
and withhold the full empirical comparison; do not silently change the denominator.

Bind the completed dataset/training launch and result identities, inherited
source/input closure and this source/test/protocol before fitting. Write once to
`.generated/go2_pulse_action_time_baseline_v1_attempt_001`; do not overwrite or
retry that attempt. Verify input/source identities again after scoring. Publish
the fitted table, complete predictions, matched schedule and scores. Preserve
10 GiB free storage. This is one-room training-role resubstitution, designed after
seeing the neural negative result. No independent generalization, risk calibration,
JEPA utility, learned navigation policy or hardware success can follow from it.

Interpretation fixed before fitting: beating zero would show learnable action/time
structure that this short neural pilot did not capture, not sensor utility or a
validated controller. Failing to beat zero would motivate examining action/state
coverage and irreducible variation before more neural fitting. In either case,
diagnose optimization scale and collect independent layouts and supported states;
do not select a new same-room winner as a deployment solution.
