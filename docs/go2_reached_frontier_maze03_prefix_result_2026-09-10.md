# Completed reached-frontier controller replay and queued physical test

The separate frontier candidate first changes a command at observation 134,
after recognizing that the robot occupies its selected observed frontier cell
`[9, -3]`. The original proposal contains one route cell; the new proposal contains
two, retaining the reached cell as traversable while choosing another frontier.
The original right turn `[0, 0, -0.45]` becomes a left turn `[0, 0, 0.45]`.
Both decisions remain nonterminal. No physical consequence of that new command
is inferred from the old trajectory.

The complete replay contains 135 observations (0–134). All 134 complete original
decisions before the intervention match after removing only candidate metadata.
All 132 raw forecast banks (observations 3–134) match exactly. Original complete
decisions reproduce the recorded stream; observed evidence, retained contact
state, accumulated floor/occupied cells and executed residual history are
unchanged. The same original corrected JEPA state is used throughout:
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
The first reached frontier, first normalized decision difference and first
request difference all occur at 134. Observation 135 was not consumed.

Result: `00579b00e70179bd1687a44ade0ed7282d8f6d3a742bbb7727a1a18733d14d25`.
Launch: `28380c18617abc59c7019a90786204ae6b3281842b70ca9dcc0f8de0b93b755f`.
Complete decision stream:
`c679f7332519faaf3fd4c13c1a729e6978016409cb25bed46bf3359ded60bf16`.
Owner output: `go2_reached_frontier_maze03_prefix_v1_attempt_001` under the
external navigation artifact root. Tool session 57528 exited zero. The result
binds 1,873 sources and two outputs. Reported wall time after launch was
1,064.858 seconds, including final original input verification. Complete original
input verifiers ran before and after replay, freshly rehashing their inputs.

Independent check: tool session 71625 exited zero. It verified the result/source/
output hashes, reconstructed all saved comparisons and summary counters from
the original and prospective streams, and regenerated all 135 public-packet
fingerprints from original raw files. That independent check did not rerun
neural inference or reconstruct contact state anew; those computations were
performed in the completed original replay.

The fresh physical test is implemented in
`scripts/run_go2_reached_frontier_maze03_pilot_v1.py`, with separate unchanged-
calculation collector/audit sources and an exact physical-prefix comparator.
For this admitted boundary it must match 7,450 physical samples, 134 prior
completed commands, all 135 raw public packets and every prospective candidate
decision, then actually complete the changed command through sample 7,499.
Subsequent progress, contact, arrival, return and settling are evaluated from
the new trajectory under the original strict criteria.

Validation: 42 native input/worker/prefix/calculation tests passed in 2.54 seconds
(session 79347). Native source/resource preflight passed (session 72497), binding
1,926 source paths with sufficient RAM and storage and creating no native output.
Six queue/process-identity/failure/no-retry tests passed in 2.15 seconds (session
61414).

The automatic handoff is active: PID 2663938, created 1789032169.81, tool session
23814; queue output `go2_reached_frontier_maze03_native_wait_v1_attempt_001`.
Queue launch:
`68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74`.
All 1,929 queued source bindings were independently verified unchanged. It has
observed the completed frontier replay and is waiting on the exact original
adapter batch PID 2659758. It will admit all six completed adapter outcomes,
recheck resources and native idleness, and launch the frontier pilot once.
No frontier native output existed at the 09:23 UTC check. Do not separately
launch another copy, alter queued sources or restart a quiet original process.

The active adapter batch and this frontier replay are different interventions
using different preassigned models on different existing development mazes.
Neither comparison is an independently unseen-layout success claim. Completed
development navigation episodes remain 37, with zero verified round trips;
all active adapter outcomes and the future frontier physics/audits remain pending.
