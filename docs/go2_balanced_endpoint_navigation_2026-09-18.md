# Dense predictor to maze navigation: endpoint interface and measured limits

The 500-ms endpoint adapter is implemented and verified against 36 retained
native branches. It reproduces the original parent predictor's decoded
motion, but the supplemented predictor does **not** improve the existing
physical waypoint interface. No full-maze trial has been launched with it.

| Forecast / diagnostic input | Transfer XY RMSE (mm) | Transfer yaw RMSE (degrees) | Point-goal action regret (mm) |
|---|---:|---:|---:|
| Original-data continuation, action | 8.812 | 1.812 | 0.927 |
| Supplemented continuation, action | 9.067 | 1.646 | 1.055 |
| Original-data continuation, action blind | 9.863 | 3.474 | 0.873 |
| Supplemented continuation, action blind | 11.978 | 4.567 | 0.873 |
| Original parent, action | 8.189 | 1.581 | 1.048 |
| Command history | 5.113 | 0.588 | 0.153 |
| Visual persistence through readout | 6.338 | 2.197 | 0.873 |
| Zero motion | 9.447 | 1.961 | 0.873 |
| Actual future features, oracle diagnostic | 10.064 | 1.052 | 1.273 |

Transfer scores use 18 branches in six shared-history groups across two
previously exposed geometries. Point-goal regret uses the same established
0.25-m targets at -45/0/+45 degrees, selecting among the three matched branch
actions per history and averaging true outcomes over exact ties. This is an
offline decision diagnostic, not new navigation. Command history, persistence,
zero motion and oracle predictions are retained references with identical
trial order and physical labels, not new independent trials.

The head is the existing frozen `DenseVisualMotionReadout`, fitted to actual
current/future features. No head, encoder or predictor is trained here. Poor
transfer of the oracle head is still visible, so these scores cannot establish
that the latent features lack motion information. Likewise, the successful
visual-goal control comparison (2/4 versus 0/4) is not invalidated: it uses a
different frozen goal metric and asks a different downstream question.

The actual maze runtime mismatch is now traced to its consumers:

- `PacedMultirateController._select_action` requires six complete eight-step
  motion forecasts at 100-ms offsets. It scores a dispatch prefix plus an
  action interval, then invokes clearance and recovery consumers.
- `ContinuousCommitmentRuntime` uses a three-tick committed prefix and four
  ticks of command commitment. The main scoring endpoint is 700 ms.
- `TerminalTranslationPulseRuntime` shortens near-goal translation execution
  to 100 ms, but retains the 700-ms scoring endpoint and settling tail.
- The current dense predictor takes three visual frames spaced 500 ms apart
  and five future applied commands, and predicts only the +500-ms image.

The new `DenseEndpointMotion` in
`lewm/dense_endpoint_navigation_development.py` accepts normalized dense
context tokens, their exact times, 15 causal applied-command history rows and
candidate five-tick applied tapes. It retains the existing normalizer and
raw future-action convention, then decodes the predicted tokens through the
same motion head. Blind forecasts are decoded once and expanded afterward,
so numerical batch effects cannot invent action preferences. Outputs explicitly
declare the 500-ms endpoint and the absence of intermediate motion or contact
predictions. `endpoint_waypoint_scores` provides endpoint utility only;
observed-geometry feasibility remains a separate requirement.

The native-data evaluation checked temporal alignment, matched actual applied
suffixes, blind equality and reproduction of the original parent predictions
(absolute tolerance 1e-6 in decoded motion). Predictions completed before
physical labels were consumed. Session 46795 exited 0. The result and source
identities are in `go2_balanced_endpoint_navigation_2026-09-18.json`.

This interface is not a replacement for the existing maze runtime: it does
not supply its required horizon or intermediate path, and the measured
physical scoring does not improve on the command baseline. Do not fill
missing times by interpolation and call them learned forecasts. The next
integration must either train and evaluate the required temporal outputs or
use a controller whose command timing and scoring genuinely consume 500-ms
visual endpoints. Keep the positive visual-goal result and this negative
motion-decoding result distinct when choosing that experiment.

No additional navigation, real-time qualification or hardware validation is
claimed. Existing exploration, routing memory and physical backtracking remain
verified only for the earlier controller stack, not for this dense-model adapter.
