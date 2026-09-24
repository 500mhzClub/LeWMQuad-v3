# Fused receipt construction completed with exact decisions and modest speedup

The original fused receipt replay completed and exited 0 in session 36692.
All 1,428 ordered observations, 1,425 forecasts and seven retained-state
checkpoints match the completed combined-controller reference. Model weights,
public arrays and complete normalized decisions remain unchanged. Completion
verification checked all 2,164 original source bindings, both output bindings,
the preceding completed reference, all rows, states and reconstructed timings
using the new checker's 2,183-source union. Session 41595 exited 0.

Result SHA-256:
`021b52bbc269bfd4f92a493f7adf90f9b8c714b8c4dac673e9fd1f3ccb97a2ba`.
Launch SHA-256:
`45bab28d786c146a5d2f66bf7d8d1d429f8a30e7b2007dfde0425be6ce47e6da`.
Verification record:
`docs/go2_fused_scoped_batched_late_history_completion_verification_2026-09-11.json`,
SHA-256 `3452f05e5eb4e68d4b1d3cd7c0940256d7945fd0de9ef2d6a59ba26eb83f17f8`.

| Window | Total controller-time reduction | Combined median | Fused median |
| --- | ---: | ---: | ---: |
| All 1,425 planning observations | 5.61% | 0.777 s | 0.736 s |
| Early, 3–12 | 6.13% | 0.586 s | 0.585 s |
| Repeated hold, 395–404 | 11.08% | 0.841 s | 0.728 s |
| Late, 1418–1427 | 7.76% | 0.966 s | 0.934 s |

The original owner PID 2780519, creation time 1789090638.22, ended. This
completed result frees the sole full CPU replay slot. All 1,425 candidate
planning calls still exceeded 100 ms. Alternating order limits one scheduling
confound, but this remains a shared-host controller-only comparison with no
sensor acquisition timing or isolated benchmark. The reduction must not be
added to earlier runs. The original visibility failure at frame 1173 and failed
round trip remain. No native command or navigation qualification follows.

The next prepared comparison substitutes the existing packed-owned insertion
into eight empty persistent indices. Its earlier resource preflight failed
while this predecessor was live. After this process ended, preflight session
12841 passed with the same 64 GiB RAM threshold and 2,183 frozen source paths.
The failed preflight and its corrected record remain preserved.
