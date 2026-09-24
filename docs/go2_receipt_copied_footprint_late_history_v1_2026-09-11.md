# Receipt-copy footprint comparison

Compare the completed packed/fused controller with a separately named successor
that uses the existing `copy_receipt` implementation in the later-floor footprint
function and the confirmed auxiliary-contact helper. Their original code,
closures, defaults and checks remain; only private function globals change.
A mirrored method-resolution order routes the two functions through a temporary
query view. The view shares the original memory fields during synchronous
selection and prohibits field assignment/deletion through itself. It introduces
no persistent state or memory-class replacement. This is not a security boundary
against explicit base-class mutation. The original scope lifetime, 18-entry
capacity, fresh/stale/failure checks, geometry guards and owned result cloning
remain. Unsupported selector inputs use the original fallback.

The verified packed/fused profile identifies evidence copying as a material
cost. This comparison measures its incremental effect; it does not change
planning, sensing, observations, model, collision rules, success rules, or the
already frozen native queue.

The original paired replay body reconstructs all 1,428 observations, compares
1,425 forecasts and complete normalized decisions, and authenticates seven
retained-state checkpoints against the completed packed replay. Two fresh models
and controllers alternate execution order. Original raw/model bindings and the
completed packed reference are authenticated before and after. Only the new
controller identifier/flag are normalized away in addition to the predecessor's
declared metadata. No further state-type normalization is introduced.

Use one full CPU replay slot, at least 64 GiB available RAM, at least 41 GiB
artifact storage and four physical CPUs. The original profile must have ended.
Fixed hash seed and single threads, disabled OpenCL, no profiling. Source-only
preflight loads no model and creates no output. The fresh output is
`go2_receipt_copied_footprint_late_history_v1_attempt_001`; failures and partial
evidence are retained, with no automatic retry or resume. There is no native
execution, new data collection, model training, or hardware command.

Preserve the original failed visibility at frame 1173 and failed round trip.
The replay is a controller-only shared-host timing comparison, not real-time,
navigation, independent-layout, memory-benefit or JEPA-benefit qualification.
Adoption requires completion and verification of the actual replay, not just
synthetic or public-packet component tests.
