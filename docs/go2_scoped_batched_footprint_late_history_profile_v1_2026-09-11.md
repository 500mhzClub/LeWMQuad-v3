# Profile the combined controller after completed paired replay

Require the original combined replay to end and publish its complete result.
Authenticate its launch, sources, all 1,428 ordered paired decisions, timing
windows, unchanged model, seven retained-state hashes against the completed
scoped reference, and unchanged negative sensing scope. A full row stream is
not sufficient. The original owner must release the one-full-replay slot.

Rebuild the same full causal history using ScopedBatchedFootprintController.
Reuse the original late-history profiler body through an isolated global
dictionary changing only controller construction, decision normalization,
output destination and progress text. Profile exactly frames 3–12, 395–404
and 1418–1427. No packet 1428 or new physical command is consumed. Require all
1,428 profiled candidate/public/original decision hashes to match the completed
combined replay. Profiling overhead remains present; do not infer a speedup
from these profile times.

Before and after profiling, rehash every original raw worker artifact bound
by the completed paired launch, validate its completed case/model state and
negative sensing scope, and call the original native input checker with
full=False to verify bound model/correction inputs. The completed paired result
already records the original full ancestry checks before and after replay.
This new timing diagnostic reuses that completed provenance; it does not
claim to rerun the full training ancestry. It verifies the actual raw inputs
and model bindings used by its inference. Any changed binding is terminal.

Use a fresh process with one Torch/OpenCV/BLAS thread, hash seed zero, and
OPENCV_OPENCL_RUNTIME=disabled before import. Require the existing profiling
resource envelope: 48 GiB available RAM, 41 GiB artifact space and at least
four physical CPUs. A live native worker may continue; no additional native
scene is created. No second full controller replay runs concurrently.

Keep the original visibility failure at frame 1173 and unsuccessful round
trip. Preserve partial profiles and a terminal failure receipt on failure;
there is no automatic retry or resume. State-size snapshots are diagnostic
counts, not a new retained-state equality proof. This work changes no policy,
training, independent-layout assignments or native queue ownership and claims
no navigation, real-time or hardware qualification.
