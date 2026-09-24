# Prospective reached-frontier controller replay on completed maze 03

Use the completed original RecentQualifiedDirectFlowController native episode,
result `330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723` and
launch `25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7`.
Create two fresh identical original assigned JEPA models, state
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
The candidate is ReachedFrontierRecentQualifiedController. Only observation-based
frontier arrival/retirement/rediscovery changes; keep all sensor processing,
observed pose, floor/obstacle evidence, contact state, mission and model unchanged.
The expanded-model adapter is not part of this separate intervention.

Consume only the original raw public packets. Reconstruct every complete original
decision and compare its actual completed command. Compare complete candidate
decisions before any reached frontier after removing only intervention metadata.
Require identical observed-state receipts, accumulated cells, retained contact
state and executed residual history; compare all raw forecast banks whenever
both controllers compute them. Preserve every complete candidate decision and
observed frontier transition receipt. Stop at the first changed request or
terminal, or the unchanged original terminal. Never consume a recorded packet
following a changed request. No native execution or counterfactual outcome claim.

Bind and verify original source/artifact identities and original completed audit
and physical-prefix receipts. Reexecute original input verification before and
after this replay, using the original verifier and its existing scoped digest
procedure. One inference thread, 32 GiB available RAM, 40 GiB disk reserve plus
one GiB replay allowance. Existing adapter simulations may run separately if the
resource envelope remains available; never start a second native scene.

Exclusive output `go2_reached_frontier_maze03_prefix_v1_attempt_001`. No automatic
retry/resume. A changed request is intervention evidence only. Fresh physics,
successful maze navigation and independent-layout comparisons remain necessary.
