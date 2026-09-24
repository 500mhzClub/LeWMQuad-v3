# First corrected adapter JEPA case: audited navigation failure

The first case of the six-model adapter batch completed its full raw audit.
There are now **38 completed audited development episodes and zero verified
round trips**. This is one case of a still-running matched batch, not a complete
model comparison or an independent-layout result.

Case: `all_phase_full_jepa_residual_maze_02`, expanded full-JEPA state
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
Original native output root:
`go2_all_phase_adapter_maze02_matched_native_v1_attempt_001` under the external
navigation artifact root. Important SHA-256 identities:

- Worker terminal: `617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e`
- Raw audit: `4d50d207b1eb99f82a74ff1869f097834578abe4787538c1b922ab1c8e0acf46`
- Readout: `21893cad78e7741eef805b6a0cbc15042be5433f1d70d3a3e42fc7dca5b46652`
- Actual adapter startup: `ef7cbc70cbb6c8841e858dc7e2de74615f751501385460e661d47848e93f81e0`
- Independent verification record:
  `0635ea5663b0d3e97f87defc232bcccacb5ff5c59bc7791d0d63cf4652f8453b`
  (`go2_all_phase_adapter_full_jepa_maze02_verification_2026-09-10.json`).

The run recorded 1,529 observations, 1,528 completed commands and 77,150 physics
samples, including ten terminal zero commands. It stopped with
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`. There was no
physical/acquisition stop. Worker processing took 5,773.76 seconds including
collection, persistence, verification and audit.

Native evaluation found two crossings of declared open edges: `[-1,0] → [0,0]`
at sample 5,901, then `[0,0] → [0,1]` at sample 11,974. There were no invalid
crossings and no nonzero native contact flags in all 77,150 samples. The run
made no observed/native arrival, did not return or retrace an outbound route,
and failed the terminal native quiet criterion. Traversing two edges does not
establish useful exploration, goal-reaching or backtracking.

Raw sensor reconstruction, full model/controller command replay, actual command
audit and unchanged model state all passed. The complete four-observation
adapter startup and executed first planned command matched the prospective
adapter replay. Independent sessions 74246 and 97263 checked 1,908 source
bindings, all 9,210 worker artifacts plus its log/terminal identities, the exact
case contract, reconstructed actual startup, and recomputed the complete
readout from the audited collection and native contact trace. Those independent
checks did not rerun neural inference; the completed worker audit did.

Strict physical visibility failed at **auxiliary observation 1173**. The front
camera passed at every frame. At the auxiliary failure, stable-interior metric
checks passed with zero bad interior rays and a maximum error of about 9.6 µm,
but one boundary ray failed. The original strict failure is retained; no pixel
is reclassified away or granted qualification. The hard-failure list is `[1173]`.

All observations **0–405** passed both camera visibility checks and the hard
measurement gate. The planned hold-reorientation intervention at 405 therefore
precedes the visibility failure and does not necessarily inherit it. Whether
the new trajectory escapes stagnation or avoids later measurement failures
requires the separately queued physical test; later original observations are
not candidate outcomes.

The automatic handoffs advanced after the original worker ended. The batch now
runs the full supervised-rollout case in worker PID 2672443 (created
1789037034.85); 262 observations were recorded at the latest inspection. The raw
hold-reorientation replay child is PID 2672447 (created 1789037036.8), launched
by its original waiter with the exact terminal hash above. It is performing
full input admission; its exclusive replay output has not yet been created.
Quiet admission is expected and does not authorize a restart. The original
frontier waiter and subsequent hold-native waiter retain their specified order.

A separate fixed-snapshot timing diagnosis found median acquisition-plus-control
latency of 1.809 seconds and p95 2.358 seconds against a 100-ms command interval;
all 997 post-warmup observations exceeded that interval. See
`go2_adapter_first_1000_observed_timing_2026-09-10.md`. These are paused-physics
workstation timings, not real-time or hardware qualification. Storage remains
available. Reliable navigation, the full matched comparison, independent maze
evidence, continuously advancing timing validation and bounded hardware evidence
remain unfinished.
