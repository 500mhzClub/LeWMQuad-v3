# Packed persistent indices preserve decisions and reduce total time

The original packed/fused replay completed and exited 0 in session 74358.
All 1,428 observations, 1,425 raw model forecasts and seven retained-state
checkpoints match the completed fused-controller reference. Model weights,
public arrays and normalized complete decisions remain unchanged.

| Window | Total controller-time reduction | Fused median | Packed/fused median |
| --- | ---: | ---: | ---: |
| All 1,425 planning observations | 9.38% | 0.737 s | 0.665 s |
| Early, 3–12 | 20.94% | 0.586 s | 0.486 s |
| Repeated hold, 395–404 | 4.22% | 0.739 s | 0.708 s |
| Late, 1418–1427 | 1.01% | 0.916 s | 1.042 s |

The overall total-time reduction is 9.38%, with a median of about 665 ms.
Every one of the 1,425 candidate planning calls still exceeds 100 ms. The last
ten frames' median is worse even though their total is slightly lower. This
is an alternating shared-host, controller-only comparison, not an isolated
benchmark or sensor-acquisition timing result. Do not add or multiply its
percentage with measurements from earlier runs.

The completed result supports retaining this composition as an optimization
candidate. It does not justify a real-time claim or replacement of controllers
in the frozen native queue. The original visibility failure at frame 1173 and
failed round trip remain; no new physical navigation or hardware evidence was
generated. Further timing work must address the remaining per-frame costs.

Completion verification reauthenticated the completed fused reference, checked
all 2,183 original source bindings and both original output bindings, and
reconstructed the complete row population, timing summaries and seven state
identities. The checker/test union has 2,185 sources. Its 23 synthetic tamper
tests passed in 2.24 seconds (session 19506, exit 0). Actual completion
verification session 72485 exited 0. No full raw model replay or training
ancestry admission was rerun.

Result SHA-256:
`ddcb9719bd60b55d44865078773c7098357fc7db316f0c0987dd5690ae52d72f`.
Launch SHA-256:
`c23a894e770903e05cbe55e91e82b21c30dda29326309ebd3fcb0596cb1c6b88`.
Verification record:
`docs/go2_packed_fused_scoped_late_history_completion_verification_2026-09-11.json`.
SHA-256: `13facf5948079eb6d26571b8d8977dacda3da77e9ab72b628575ad84a213411c`.

The original packed owner PID 2786620 ended, freeing the full CPU replay slot.
The no-RGB direct sixth-case native audit continues under its original owner.
