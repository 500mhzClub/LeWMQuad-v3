# Completed hold replay and veto diagnosis

The full V2 hold replay exited 0 (session 13252). Result SHA-256:
47f8ab41c12d91d9f46a1def1308218a5eead9ca6c18651683051477001db89a.
Root: go2_residual_hold_prefix_v2_attempt_001 in the navigation development
artifact volume. All 3,004 decisions through the original terminal at frame
3003 preserve the original commands, mission/residual state and selections.
All 3,000 forecast comparisons pass, public arrays and model state are unchanged.
There are zero interventions; terminal remains MISSION_TICK_BUDGET_EXHAUSTED.
This is a completed negative prospective replay, with no new physical execution.
The earlier V1 output-headroom failure remains intact.

The separate hold-veto readout exited 0 (session 3476). Result SHA-256:
14a6b2cc7bcc889823426ab9961444754a849cde6c3d8a60514480db82ee68c8.
Root: go2_residual_hold_veto_readout_v1_attempt_001. All 3,014 original decisions
were processed, including all 2,643 selected holds. Original hold utilities
were reconstructed from the raw predictions and strictly causal residuals.
24 holds had no strictly better allowed movement. At the other 2,619 holds,
every better alternative had a veto unaffected by correcting prediction 1.
No better alternative was blocked solely by the first two path segments.
There were no original surface-veto occurrences among these alternatives;
later nominal path segments account for the invariant vetoes. Segment counts
overlap and must not be added as counts of independent decisions.

Independent check 40923 exited 0: all 1,687 replay source bindings and both
replay output bindings match; all 1,725 readout source bindings and its output
binding match. Both result identities above were checked. The earlier check
4325 output was unavailable, so this report relies on the recovered new check.

Next hypothesis: retain the existing first-point recovery, then consider a
separate nominal policy that anchors the model's later relative XY displacements
at its causally corrected first point. This changes later predicted positions
and has no calibrated later-horizon error bound. It requires component tests,
a prospective replay stopping before any following old observation, and a fresh
native run before claims about navigation. No physical alternative outcome is
inferred from this diagnosis. The goal remains unachieved.
