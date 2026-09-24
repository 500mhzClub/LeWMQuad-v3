# No-RGB supervised case completed: stationary, contact-free failure

The fifth case of the fixed six-model comparison completed its original raw
audit and parent acceptance. It exhausted the mission budget with no observed
arrival, no traversed maze edge and no verified round trip. Strict primary and
auxiliary visibility passed, with no hard measurement failure frames and no
physics contact samples. This confirms a navigation failure despite valid
measurements; it does not establish successful control.

- 3,014 observations, 3,013 command intervals and 151,400 physics samples.
- Observation/control median 2,671.93 ms; all 3,014 observations exceeded the
  100 ms command interval.
- Complete worker wall time: 16,553.72868090705 seconds, including collection,
  original raw audit and its input checks; this is not an isolated audit time.
- Original unchanged model state:
  `7d9a53c6477884548f687cb9a26184e11621304b927cb924838795badb226035`.

Completion authentication checked 1,908 frozen sources and 18,124 artifact
bindings, the exact worker and parent receipts, original case contract, and
the readout reconstructed from raw physics-contact samples. Original bound
input verification was repeated. The raw model/sensor auditor and full
training ancestry were not rerun by this authentication.

Worker SHA-256:
`620f7a48421edca49145553207a75aba7f2fe5ae39d1c2cfb2e0717e2172d674`.
Parent completion SHA-256:
`f6b14c28d65b1266b46fb6d86a6237fec221dddf7036a8d1d1ecf1b449cfa773`.
Verification record:
`docs/go2_all_phase_adapter_no_rgb_supervised_maze02_verification_2026-09-11.json`,
SHA-256 `c80eb47d9790ecf56939193ab19cbd74935b568eb156dd3a5e8d2beedb041771`,
session 32497, exit 0.

The original worker PID 2743870, creation time 1789071424.56, ended. The
original parent PID 2659758, creation time 1789030196.29, then started the sixth
case, `all_phase_no_rgb_direct_residual_maze_02`, in fresh worker PID 2775051,
creation time 1789088004.42. At 00:59:14 UTC the sixth worker and its collection
directory existed, with no complete collection or terminal receipt yet.
The queue order and existing source bindings remain unchanged.

The running total is 42 audited development episodes and zero verified round
trips; the current batch is five of six audited cases. The independent study
has not started. Its existing metadata correctly describes the direct outcome
model as predictive, the reactive comparison as a whole-method comparison,
and the current-pair arm as a planning-map ablation rather than a fully
memoryless controller. The direct head uses eight action-conditioned future
outcomes without using latent rollout for those selected outcome estimates;
the shared model forward still computes both output branches. No retrospective
relabeling or changes to the frozen assignments were made during review.

Next, finish the sixth case and the queued navigation-fix experiments, use
their results to decide the prospective controller and comparison scope, and
then complete the independent-study policy/input/queue admission. The CPU
monitor and overlap verifier are prepared, but their preparation is not a
navigation result or final population launch.
