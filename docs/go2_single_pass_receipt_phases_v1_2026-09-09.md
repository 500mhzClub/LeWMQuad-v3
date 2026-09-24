# Remaining costs of the combined controller: phase diagnosis

Replay the full 514-observation original learned maze-2 episode using the
verified SinglePassReceiptCopiedController, with explicit phase timers only.
Require the completed combined benchmark result
`d688f2ed9d30177d2e55fb98e9c9f25d2b035f2258d86449ac8e86615cd13c72`,
its output bindings, the original complete raw-audited episode and the original
verifiers of both optimization predecessors. Preserve every frozen source.

Instantiate a fresh assigned corrected JEPA model and controller with their
original state. Time controller observe/advance/result, dual-camera motion,
floor registration, map integration/waypoint, memory insertion/classification,
auxiliary confirmation, contact queries, selector and model forward. Timers
delegate original methods; all eight map indices retain the verified single-pass
implementation. No changed physics, policy, forecasts, feasibility, geometry,
memory retention, scoring or copy semantics. Remove all temporary model hooks.

At every observation require exact equality of the complete saved original
decision and unchanged public arrays; stop immediately on any discrepancy.
Authenticate actual completed requests, observation endpoints, the complete
514-observation population, unchanged model state and absent gradients. Reject
truncation. No following observation is consumed after a mismatch. Exclusive
phase times must sum exactly to the root controller duration at every frame.
Report active-observation timing separately from warmup and terminal frames.

This is a bottleneck diagnosis. Instrumentation overhead is included; it is not
a controlled speed comparison, native execution, acquisition/receipt-I/O
measurement, or evidence of real-time operation or navigation success.

One CPU replay, single numerical threads, 8 GiB RAM admission, 128 MiB output
allowance above the existing 40 GiB reserve. Measure topology, affinity, CPU,
RAM, GPU and both storage volumes and competing jobs before launch. It may
overlap the single native scene and the independently frozen hold replay only
with measured headroom; refresh resources at admission and every 64 frames.
No model training, GPU execution, real-robot motion, source export or protected
benchmark access. Retain exclusive output and any failure at
`go2_single_pass_receipt_phases_v1_attempt_001`.
