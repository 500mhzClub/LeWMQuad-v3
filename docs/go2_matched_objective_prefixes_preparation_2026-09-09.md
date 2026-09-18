# Paired JEPA/supervised-rollout prefixes implemented and submitted

The actual-model comparison supplements the previously verified pure metadata
admission. It reloads both exact full-RGB first-seed fits through the existing
evaluation-only loader, checks their base states and each own training-only
correction buffers, records corrected state hashes, and requires unchanged
weights/no gradients. Both unchanged observed controllers use rollout_outcomes.
No direct-head substitution, training or new fitted maze correction is allowed.

The fixed layouts1,2,3are replayed from episode start with a fresh model pair
and fresh controller/memory per case. The JEPA arm must reproduce every original
decision exactly. Both arms must share raw/registered observation evidence,
mapping and mission receipts, goal and floor-partition state. Warmup is exact
except the declared model condition. Each selection must retain the same
100–800ms full-RGB rollout interface and its own authenticated intercepts.
Online residuals remain each model's own causal prediction errors.

Stop at the first changed requested command or terminal, or either original
terminal. Prediction differences alone do not stop a still-common command
history. Never consume the next recorded observation after a changed command.
Maximum case prefixes215/504/265observations include the known original first
terminal boundaries. Retain all fixed cases and all failures.

21tests passed in1.94s. They check matched warmup, forecast differences without
command changes, all nine observed-state fields, wrong heads/bias/variant/
clocks/shapes/nonfinite forecasts, original replay/tape mutation, and terminal
boundaries. A complete runner test fails if the reader requests the next
observation after its first changed command; both model arms consume exactly
the four expected observations in that fixture. These are integration checks,
not a completed actual-model comparison or JEPA advantage result.

| Source | SHA-256 |
| --- | --- |
| `lewm/matched_objective_prefix_development.py` | `1388d483470a6353460d5a691016ea479499924f9a0952215be22839e724877c` |
| `scripts/replay_go2_matched_objective_prefixes_v1.py` | `6712a5c4b0c3a1da7c1b54f7c9d7941972e968c1b039b0f1d085637afc050d64` |
| `lewm/tests/test_matched_objective_prefix_development.py` | `5fffb66e3632991b911aa3b886d9f4f30a4019cf79a88904abd7fff52a413558` |
| `docs/go2_matched_objective_prefixes_v1_2026-09-09.md` | `4eb9be9d8f8b809d549a48f1bc16618ec494f14735a99fc689165cc5b66b7643` |

Submitted session58952,PID2485387; initial verified running37.0CPU seconds,
RSS1,350,909,952bytes, input authentication pending. Exclusive output
`go2_matched_objective_prefixes_v1_attempt_001`. Preserve this attempt and all
frozen source identities. No checkpoint loading or result completion is claimed
by this submission record. Full execution protocol is the source listed above.

Submission hardware:82,783,170,560availableRAM bytes,88,063,770,624artifactfree
bytes,21,359,616,000workspacefree bytes,CPU0.3%,bothGPUs0%,all32CPUaffinity.
The completed native pilot had exited. Independently submitted residual-native
78591/PID2485335, memory-readout49332/PID2485368 and this CPU replay may overlap
their immutable-input verification. Capacity allowances are32+8+12GiB against
82.78GBavailable, with one native scene and two independent CPU analyses. These
are capacity checks, not OS quotas; each launcher rechecks current resources.
The readout and paired model comparison do not alter the native definition.

The matched-model replay is not physical evidence for changed trajectories.
Fresh paired native execution/raw audits remain required, and one optimization
seed does not establish broad JEPA advantage or reliability. Keep the residual
then tracking native queue unchanged.
