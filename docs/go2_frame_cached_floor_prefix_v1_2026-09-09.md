# Exact common-floor controller prefix with frame cache V1

The completed floor-input diagnosis found nine index calls but exactly two
byte-identical input groups at each of frames20,60,100. Result:
`49de94b98890885d0c47bf78313d2383e62c0d3fc2583795bd05cc1a81291c60`.
Its complete101 decisions matched the recorded predecessor. This motivates
explicit per-observation reuse of the existing pure floor-index calculation.

New implementation: FrameCachedJointFloorRoundTripController. It inherits the
common-floor estimator/controller algorithm and replaces only its map with
explicit frame-scoped floor geometry providers. Thirteen source derivatives
are individually checked against the exact predecessor sources with only the
declared provider/method routing changes. The cache holds at most eight exact
depth/validity/up byte keys and immutable results. Excess distinct calls fall
back to the original calculation. Changed input bytes miss; no rounded pose,
hash-only equality, validation bypass, cross-observation cache or global function
replacement. Every camera packet still passes the original validation. Identical
pure geometric arguments may share a result; camera identities and acquisition
receipts remain separate. Close and discard the cache even when observation
fails. Original historical patch arrays, classifiers, uncertainty statements,
model, scores, action policy and receipt copies remain intact.

The cache is not installed in the running native attempt. The controller's
algorithm label remains unchanged so complete decision equality can be tested;
this distinct implementation is identified by source/launch bindings and class.

Exclusive root go2_frame_cached_floor_prefix_v1_attempt_001. Reconstruct all960
observations of the completed common-floor prefix from its bound public
predecessor packets. Require every complete new decision to equal the saved
common-floor candidate decision, including its original visual witnesses, maps,
mission, forecasts, surface/nominal checks and requested commands. Stop after
959 without reading the unexecuted changed command outcome. Record timings,
exact decision hashes and cache hit/miss counts separately. Model state must
remain unchanged. Verify prefix artifacts, failed-predecessor bindings, model
admission, original native/readout inputs and complete source bindings before
and after. Preserve all failures; no retry or source change after launch.

One CPU replay process, one numerical thread, minimum8GiB available RAM and
128MiB additional artifact allowance above the40GiB reserve. It may run beside
the existing single native scene using independent immutable inputs and its own
output root. Inspect current topology/affinity/utilization, RAM, GPU, competition
and storage before launch. No model training, native execution or new navigation
outcome. Timings are descriptive; this unpaired replay cannot establish a
controlled full-loop speedup or real-time qualification.
