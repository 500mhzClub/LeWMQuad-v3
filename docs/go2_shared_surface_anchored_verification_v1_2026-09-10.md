# Independent check of the completed receipt-sharing replay

`scripts/verify_go2_shared_surface_anchored_prefix_v1.py` accepts one explicit
completed replay result SHA-256 at the fixed original output root and binds the
observed launch `3b2c3d3650ebefc2e314dab4de107877cc2dc805bed43bf5b190422bc6f4d97d`.
Reject failure, absent completion, changed source/output identities, altered
intervention or expanded scientific claims. Require the exact completed
batched-query, receipt-copy and profile predecessors.

Reconstruct the canonical hashes of all 405 saved original decisions and their
metadata-only candidate equivalents. Verify the original compressed decision
stream and command tape against the paired replay's admitted bindings before
and after reading. Check all 405 command endpoints, complete timing population
and fixed alternating order. Compare every public packet hash and original
decision hash with both completed profile and receipt-copy records. Recompute
all three timing windows and require exactly 402 forecast-bearing decisions.

Require the four reported retained-state hashes to match the completed
receipt-copy comparison exactly, with no state-type normalization in the new
launch. This check does not reconstruct hidden state, reload raw sensor packets
or rerun neural inference: those executions belong to the completed paired
replay. It verifies recorded evidence and reports that limitation explicitly.
It does not replace the original before/after full input admission.

Only after all checks pass, write the exclusive repository verification JSON,
including result/launch/output identities, the original stream/tape bindings,
recomputed timings and this checker's own recursively verified source closure.
Do not produce a completed verification record from partial replay output.
The check adds no native episode, independent-layout result, real-time or
hardware qualification.
