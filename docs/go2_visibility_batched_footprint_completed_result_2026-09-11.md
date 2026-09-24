# Visibility batching: completed and verified paired controller replay

The full paired replay and its completion verification are complete. Skipping
empty retained-patch visibility enumerations reduced total timed controller
work by **2.175638%** on the fixed original full-JEPA recorded history:

| Metric | Receipt-copy baseline | Visibility-batched candidate |
| --- | ---: | ---: |
| Total controller time, 1,425 navigation observations | 1034.109 s | 1011.611 s |
| Median controller time | 679.621 ms | 663.808 ms |
| Observations exceeding 100 ms | 1,425 | 1,425 |

All 1,428 observation rows were verified against the completed predecessor's
public-input, original-decision and candidate/baseline identities. The complete
report reconstructed, including 1,425 raw-model forecast comparisons, all seven
retained-state checkpoints, the fixed alternating execution order, all timing
windows and the original negative sensing scope. The previous receipt-copy
completion check was reauthenticated and the actual raw sensor/model bindings
were rehashed. No new raw sensor/model replay or full training-ancestry study
was executed by this verification.

The 18 focused completion-verifier tests passed in 2.30 seconds. The completed
verification checked 2,220 source bindings, including the 2,217 original run
bindings, and the original process was confirmed ended. Verifier session 89671
completed with exit code zero.

- Replay result SHA-256:
  `3aae2867faf3a127b3a42acf13c69514607ce15272d5a98ad4a87da8d32fddb4`.
- Completion verification:
  `docs/go2_visibility_batched_footprint_completion_verification_2026-09-11.json`,
  SHA-256 `8ed19a7e3de6b0542f912f35f6b3f3de38e812a51a7dcc7748ee091ada4f44ed`.
- Verifier source:
  `scripts/verify_go2_visibility_batched_footprint_completion_v1.py`.

These are paired controller-observe timings on a shared host. Sensor acquisition
was not timed and no physical commands were executed by this replay. The
unchanged recorded failure remains part of the evidence. No real-time,
navigation-success or hardware qualification follows. This result is specific
to the full-JEPA timing case; it is not a measured speedup for the separate
no-RGB chained-anchor controller. The optimization has not been adopted by that
tracking experiment. Do not combine its percentage with earlier optimization
percentages from different paired runs.
