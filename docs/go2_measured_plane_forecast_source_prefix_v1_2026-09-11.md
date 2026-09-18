# Actual learned-versus-nominal forecast-source prefix

Use the two prepared `MeasuredPlaneForecastSourceController` modes with the
same measured-plane perception, observed map, mission, candidate actions,
scoring, predictive feasibility and observed residual update rules. The
assigned corrected no-RGB direct model remains
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
Load it twice independently with disjoint tensor storage. The nominal arm
must make zero model forward calls. Preserve its requested-velocity tracking
assumption and constant contact reference explicitly.

This compares learned and nominal prediction sources. Both arms remain
predictive. It does not isolate planning on/off, JEPA training or persistent
memory. In the current policy, waypoint pose utility already uses 100 ms;
later forecasts affect contact risk and path feasibility. Do not describe
this experiment as a fully nonpredictive or fully memoryless baseline.

Consume only the already verified measured-plane public prefix, at most
frames 0 through 122. Reproduce every complete baseline decision after only
removing its added source provenance/root metadata. Require identical full
visual, floor, map and mission receipts in the nominal arm while history is
shared. Residual values may differ because their forecast source differs;
do not erase or normalize them. Record complete decisions from both arms.

Stop at the first different requested command or terminal outcome. Never
consume the next observation after that boundary. If commands remain equal,
stop at frame 122, before the observation following the already known
measured-plane physical intervention. Do not infer a counterfactual physical
outcome, navigation improvement or model advantage from this replay.

The runner is `scripts/replay_go2_measured_plane_forecast_source_prefix_v1.py`.
Its exclusive root is `go2_measured_plane_forecast_source_prefix_v1_attempt_001`
on the existing navigation artifact volume. Require the original deterministic
single-thread CPU environment, 64 GiB available RAM and 43 GiB artifact free
space, ended preceding full CPU replay/checker and ended combined-controller
prefix replay. One separate fresh native scene may continue concurrently.

Authenticate the original recorded prefix, assigned model and full original
worker artifact roster before and after. Check all model states and absent
gradients. Count actual model forwards with non-mutating hooks; remove them
at the end. Reconstruct the entire saved output and report against the fixed
reference. Preserve every failure without retry, resume or overwrite. No
native, real-time, hardware, independent-layout or navigation claim is made.
