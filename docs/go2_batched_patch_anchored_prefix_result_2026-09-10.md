# Completed batched floor-query controller replay

The separately named batched patch-store controller preserved all 405 original
normalized decisions and 402 fresh-model forecasts over observations 0–404.
The original replay completed its final input verification; its process ended.
No native episode was executed and observation 405 was not consumed.

| Fixed window | Calls | Original median | Candidate median | Original total | Candidate total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-warmup, 3–404 | 402 | 0.800440 s | 0.756689 s | 344.834359 s | 327.759241 s |
| Early navigation, 3–12 | 10 | 0.690189 s | 0.703894 s | 7.129359 s | 7.338524 s |
| Repeated hold, 395–404 | 10 | 1.644346 s | 1.359760 s | 16.181577 s | 13.233535 s |

Total post-warmup controller time decreased by 4.95%.
Repeated-hold median time decreased by 17.31%.
Early navigation was slightly slower. Every one of the 402 post-warmup calls
exceeded 100 ms for both controllers. Timings cover controller observe only,
exclude sensor acquisition, alternate execution order and were measured on a
shared workstation alongside the original native audit and brief diagnostics.
This is a single replay measurement, not an isolated benchmark or an estimated
population performance advantage. End-to-end replay wall time, including
input verification, was 1738.276030 seconds.

The intervention only replaces the two fresh retained-floor patch stores with
the batched query implementation. The memory, mapper, selector and mission
implementations remain the originals. Receipt-copy optimization is not
combined with this change. Complete retained observed state matches at frames
3, 12, 395 and 404 after normalizing only the two declared patch-store type
tags. Complete decisions normalize only the declared controller metadata.
Public input arrays and both model states remained unchanged; the assigned
JEPA model SHA-256 is `35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.

The independent recorded-evidence check reconstructed all 405 original saved
decision hashes and all 405 expected candidate hashes, checked public input
and the four reported state hashes against the completed receipt-copy replay,
recomputed every timing window, and verified 1,982 source bindings and both
output bindings. This second check did not rerun neural inference, reload raw
sensor packets or independently reconstruct hidden state; those executions
belong to the original paired replay. The 66 component/integration/replay tests
passed before execution, as recorded in the execution note.

Artifact root: `go2_batched_patch_anchored_prefix_v1_attempt_001`.

- Result: `650fe137b41c91f8860ce59987c876d1c2591329ee2fc63ad056e63059e47419`.
- Launch: `38a474e183ca8d0972d2e10b1f3afb18efb191aca0e6fd0113e3928964477c30`.
- Comparison: `de638cad9fd23f1dbe6f567ecfcaf307ce386bbc5c30db326ea6d0ae0da2b4f7`.
- Independent verification: `b1a7b903d874dfb7d833d939984c4f2dcc65a9678f8a68f7b2c3e14a408bb7ea`,
  `docs/go2_batched_patch_anchored_prefix_verification_2026-09-10.json`.

This is completed controller equivalence and timing evidence on a fixed
original development prefix. It establishes no new navigation outcome,
independent-layout reliability, real-time sensing/control or hardware readiness.
The running native batch and queued policy trials retain their frozen sources.
