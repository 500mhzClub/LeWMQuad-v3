# Completed-trajectory hold-veto readout V1

Read the original completed residual-feasibility maze 2 episode, native result
55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466,
and require the completed paired native readout
aa1abf8ce0110dca271bc5f93fabf51a62bb41315fa5971afe16f87c25c59888.
Use all 3,014 original ordered decisions and actual commands, including exactly
2,643 selected holds and first terminal MISSION_TICK_BUDGET_EXHAUSTED at 3,003. Do not read
the incomplete hold-policy replay as a completed result or infer its outcome.

For every hold, reconstruct all six saved utilities and components exactly from
the current raw forecast, current goal and strictly causal observed residual.
Retain the original phase allowance, full-plan contact score, 45 cm nominal
radius, all eight ordered path segments and original surface vetoes. Reject
inconsistent summaries, altered scores, malformed forecasts or command mismatch.
No model, tracker, map or corrected geometry is reexecuted.

Classify whether there is no strictly better allowed nonhold, whether every
better alternative has an original surface veto or a blocked segment numbered
2–7, or whether some better alternative is blocked only in segments 0–1.
The existing first-point correction changes predicted point 1 only, and thus
only segments 0 and 1; endpoints of segments 2–7 stay unchanged under the same
map/pose. Tests verify this with the original constrain/plan functions. This is
a structural limit of that intervention, not a recomputed feasibility verdict
for another policy or evidence about unexecuted physical outcomes.

Retain first examples and aggregate veto counts without selecting a policy
parameter from a successful physical outcome. This is diagnosis on a reused
development trajectory. A new policy still requires explicit implementation,
prospective first-change replay and fresh physical execution with full audit.

Authenticate original native/readout metadata and all input/source bindings.
Use the frozen scoped verifier only after admitting its completed benchmark
137867773bfe6c6eb05a125a288012ff6017aa3134f4687d7bccdc7f99c02071.
All original native verifier conditions execute, every cached file is freshly
hashed at the end and no cache survives. Repeat verification after diagnosis.

One CPU reader, 8 GiB memory admission, 128 MiB output allowance and standing
40 GiB artifact reserve. Refresh hardware before substantial work and check the
complete active allowances. No model load, training, GPU or native scene. Output
go2_residual_hold_veto_readout_v1_attempt_001 is exclusive; preserve errors and
do not retry in place. No deletion, sealed access, source export, physical
clearance certification, navigation qualification or deployment follows.
