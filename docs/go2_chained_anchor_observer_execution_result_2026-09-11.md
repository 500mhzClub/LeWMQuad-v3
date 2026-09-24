# Chained retained-anchor observer integration and registered replay

Implemented `ChainedAnchorDualCameraPose` and its public `ChainedAnchorVisualMotion`
adapter in separate source files. The original observer runs first. A successful
original anchor measurement returns unchanged; a measured bridge or missing pose
can trigger a chained retained-anchor search. New qualified conflicts remain
terminal. If the search adds no anchor, the original evidence and bridge counters
are restored, so the allowance is spent once. Cache storage is limited to 33
owned grayscale/depth frame pairs. Existing pose, gyro, reference and promotion
rules remain in the inherited observer.

Validation completed:

- 12 association tests passed in the preceding component/probe work.
- 15 observer integration tests passed, including an actual synthetic image-chain
  rigid fit, original-anchor bypass, unchanged bridge accounting, exhausted-budget
  rejection, original and new conflict vetoes, cache ownership/bounds and public
  sensor-failure latching.
- 18 replay-runner tests passed, covering exact original evidence reproduction,
  changed-evidence and terminal stopping, clocks, frame limits and CPU ordering.
- Source/resource preflight passed with 2,258 source bindings. It observed 6.9%
  CPU use, about 67.6 GiB available RAM, 564 GiB artifact free space and 19.8 GiB
  workspace free space. This was a capacity check, not a resource reservation.

The completed tracking worker was admitted through its original worker checks
and 5,283 artifact/log/terminal/launch bindings. Its raw audit has finished:
five maze-edge crossings, no arrival, no round trip, strict visibility pass,
and no hard measurement-failure frames. Its original intervention command at
859 was physically completed and its preintervention physical/public prefix
matched. The original parent completion was still pending at this inspection.

Registered one exclusive observer replay:

- Root: `go2_chained_anchor_observer_prefix_v1_attempt_001` under the development
  artifact volume.
- PID/create time: `2838943` / `1789126094.17`.
- Tool session: `46363`.
- Launch SHA-256:
  `c1ee58ed93cd71dae49198747f8c3be281cd24850051e1379e29c1b1a11edf18`.
- Worker terminal SHA-256:
  `d79f413371e4ee030156b60270cfc7077c50640b416bdca29a6d18aacc0fef4c`.
- Preparation receipt SHA-256:
  `99b7102afa9927ec7e51c6aef4f4e08368d2145796a0f3357214f0324569cec1`.
- Execution receipt SHA-256:
  `9068a3273ec6421d91029b32ec579e3dd7a8a6de210103f000bc286f54d493b6`.

The execution receipt confirmed the exact PID, create time, argv, boot and
frozen source table. The process was waiting for visibility replay PID 2834244
to end. The runner will then replay at most 864 observations, stopping at the
first changed evidence or terminal outcome. It will not read a recorded future
after the first changed observer evidence. No full observer result was available
when the execution receipt was written.

The component, observer, runner, tests and protocol are now source-bound to that
registered attempt. Preserve them. Follow-up changes require separate source
and attempt identities. A full controller replay and fresh prospective physics
remain necessary; no navigation recovery, timing qualification or hardware
result is claimed here.
