# Fresh nominal predictive navigation paired with the learned measured-plane run

Execute one fresh maze-02 episode only after the exact learned native parent
PID 2916106, creation time 1789162140.42, ends with a complete, fully audited
result. Its launch is
`93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`.
Admit a complete scientific negative as well as a success; preserve any
execution/audit failure without automatic bypass. Do not start another scene
while the original is live or restart a failed attempt.

The runner is `scripts/run_go2_nominal_measured_plane_maze02_v1.py`, with root
`go2_nominal_measured_plane_maze02_v1_attempt_001`. The controller is the fixed
nominal mode of `MeasuredPlaneForecastSourceController`. Its subclass only
fixes that constructor argument; decision identity is unchanged from the
completed forecast-source prefix.

Match the learned episode's scene, physics and appearance seeds, robot,
initial state, public sensors, measured-plane perception, floor map, mission,
candidate actions, score rules, predictive feasibility, observed residual
updates, command limits, 4,000-tick navigation budget, renderer environment
and full physical/sensor audit. Load the same corrected no-RGB direct model
identity for interface/assignment admission, but prohibit all learned model
forward calls in collection and audit. No weights or corrections are fit.

The nominal branch predicts perfect requested-velocity tracking and uses a
constant contact reference. Both arms remain predictive; this comparison
does not isolate planning on/off, JEPA training, or memory. Record zero actual
model calls only after enforcement. The full audit must be labeled controller
and nominal-forecast replay, with learned-model replay false. Complete raw
sensor, command, renderer, physical and outcome checks remain mandatory.

The prospective intervention is bound by forecast-source result
`0d080057b6fb4802e103623496e92e0bbb7be77474cfdbf9a9e6d08c500b8478`.
It contains four shared observations, one learned forecast, zero nominal model
calls, and a first changed request at frame 3: learned left arc `[.16,0,.45]`
versus nominal forward `[.2,0,0]`, with all six actions feasible.

Require exact equality of all first 900 physical samples and all first four
public observations between the fresh learned and fresh nominal episodes.
Require complete reproduction of both controller decisions and actual
completion of the different frame-3 commands, including their 50 physical
steps. Following physical outcomes belong to their respective fresh episodes;
do not compare them to a counterfactual replay. Preserve an early negative
before the boundary without an intervention or success claim.

Freeze source before execution. Authenticate the complete learned artifact
roster and all worker/audit/readout identities, the recorded prefix and model
identity. Use the exact deterministic CPU/renderer environment, at least
32 GiB available RAM and 55 GiB artifact space, and one fresh spawned scene
worker. Wait before output creation on occupied slots, retain terminal
failures and do not retry, resume or overwrite. Source preflight runs no scene
or model and consumes no learned outcome while its owner is live.

Evaluate each full fresh episode with the unchanged physical round-trip,
settling, strict visibility and hard-measurement criteria. Physics still
pauses during computation. This is a reused development maze, not independent
layout evidence; it grants no real-time, hardware or deployment qualification.
