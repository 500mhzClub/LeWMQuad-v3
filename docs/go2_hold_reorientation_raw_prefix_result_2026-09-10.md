# Completed raw hold-reorientation prefix

The complete paired raw replay passed through observation 405. Original and
candidate controllers used fresh copies of the same expanded JEPA model and
the same recorded public observations. The first and only changed request was
at observation 405: original hold `[0, 0, 0]` became left turn `[0, 0, 0.45]`.

- Raw result: `f7e054e78879fd7827761cf36bb3c698c22fe394cd084ed17f165a0f643611e6`.
- Owning waiter result: `62980125c12a4455f355f6c311ccf56d74d4f94bbc6ee0507b7908d53d6a135d`.
- Original worker terminal: `617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e`.
- Candidate decision stream: `98147f2c0604ec6a8379e032a3381d679e79e9d937b764073b52cd4802aac1ea`.
- Model state: `35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.

There were 406 observations and 403 raw forecast comparisons. Every complete
original decision was reconstructed. All candidate selections matched the
previously fixed saved-selection expectations. Forecasts, scores, gates,
observed map/contact memory and executed residual state remained exact, with
only the declared first request change and its receipt at the boundary. Model
state and public input arrays remained unchanged. No observation after the
changed request was consumed. Full original input verification passed before
and after the replay; both the child and its owning waiter exited successfully.

Independent verification rechecked all 1,943 raw source bindings, both raw
outputs, all 1,946 waiter source bindings and five waiter outputs. Sessions
84671 and 78707 reconstructed all saved comparisons and public packet
fingerprints, including completed original command endpoints and the exact
first changed request. A separate waiter-chain check reconstructed its
completion receipt. These independent checks did not rerun neural inference or
rebuild hidden contact memory; those checks were performed by the original
completed raw runner. Machine-readable verification:
`docs/go2_hold_reorientation_raw_prefix_verification_2026-09-10.json`.

This establishes the prospective intervention, not its physical outcome. The
candidate command has not yet been executed in a new scene. It does not prove
escape from the hold, goal-reaching, return navigation, memory advantage or
real-time operation. The registered hold-native waiter now has its raw-prefix
dependency complete and continues waiting for the scheduled frontier trial.
The frontier trial follows the complete six-case adapter comparison batch.
No queue order, policy source or failure was replaced.
