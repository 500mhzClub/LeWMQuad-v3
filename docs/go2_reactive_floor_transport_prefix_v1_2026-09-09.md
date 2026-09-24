# Reactive comparator with the current perception and settling contract

The earlier reactive maze pilots used older motion, map/contact and arrival
rules. Implement a separate ReactiveFloorTransportController sharing the current
DualCameraVisualMotion, MeasuredFloorTransportRegistration, MeasuredFloorTransportMap
and MeasuredFloorTransportMission with the learned controller. Retain the
existing ReactiveConnectorRouteSelector and its current-geometry action rule.
No learned model, forecast residual, predicted candidate feasibility or learned
score is used. This is a method-level non-predictive comparator with shared
sensing/mapping/mission, not an isolated prediction-ranking ablation. Predictive
and current-geometry gates cannot be described as identical.

Use scripts/replay_go2_reactive_floor_transport_prefix_v1.py --preflight-only,
then execute the same script without the flag after input/source/hardware
admission. Exclusive output:go2_reactive_floor_transport_prefix_v1_attempt_001.
One CPU replay, one numerical/OpenCV thread, at most64observations. It may run
beside the existing single CPU native scene after assessing hardware; no native
scene or learned model is loaded by this replay. Require8GiBavailableRAM and
the original40GiBreserve plus256MiB. Preserve failed outputs without retry.

Bind the completed eleventh native result
44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c
and every recorded artifact. Also bind current native launch
8dbd36ce4c300bef2b42b31cb6c04d5633624163684f1a5e88de5fcbb8002369
only as source-identity evidence: no new observation from its running trajectory
is used. All frozen predecessor bindings must remain unchanged.

Start a fresh reactive controller from original observation0. Reconstruct
primary/body/fast-gyro and paired auxiliary RGB/depth packets from the completed
native captures. At each observation, require exactly the same raw and admitted
pose, map receipt, auxiliary partition receipt, observed distance and settling
mission as the original policy up to its current pre-command state. Permit only
the validated settling motion-source wording change associated with explicit
floor transport. Compare requests with the actual completed old command tape.

Stop at the first requested-command or terminal-policy disagreement. Do not
read the following recorded observation as a reactive outcome. If no difference
occurs, stop at the fixed64observation limit. Preserve complete reactive
decisions and the exact shared-state checks. An internal sensor/controller
failure or mismatched shared state fails the replay rather than being treated
as a successful baseline intervention.

This replay establishes only causal integration over the observed common prefix
and identifies the first proposed policy change. It does not establish the
outcome of that new command, reactive navigation performance, floor-transport
use on an executed reactive trajectory, a matched native comparison, a memory
or JEPA advantage, real-time operation or deployment qualification. Those
require fresh prospective native episodes with their own full raw audits.
Keep the current learned native attempt and independent-layout cohort unchanged.
