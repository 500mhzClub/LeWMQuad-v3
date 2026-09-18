# Expanded-model planner adapter: completed startup replay

All six expanded-data models now reach the unchanged planner in recorded raw
controller replay. The complete first three warmup decisions are unchanged. At
observation 3 the original model wrapper raises `training-only translation
wrapper in evaluation mode required`; the adapter produces a normal selection.
All saved model state entries and all expanded forward-output tensors are exact.
The planner's selected forecast bank equals the original expanded wrapper's
direct forward computation on the same causal inputs. The failed original
controller had no saved forecast at that boundary.

Full/no-RGB JEPA and direct models request the left arc; both supervised-rollout
models request the left turn. These replay requests were not executed in physics.
No observation after the first changed request was consumed.

V2 result: `0d006a18920ba733f46f79fc27444c72038de8da3a4df08ff5af343c8f965daa`.
It binds 1,902 sources and seven outputs. Independent receipt reconstruction
checked all 24 original public packets, original requests, replayed decisions,
unchanged observed evidence and contact state, and boundary commands; that check
did not rerun neural inference. Verification document:
`go2_all_phase_planner_adapter_startup_verification_2026-09-10.json`, SHA-256
`dd344be02f04905d16ccd170e990028ca588124a44f946c231052519bdadb9f4`.
Adapter tests: 11 passed; observed-state comparison tests: eight passed.

The failed V1 diagnostic remains preserved. It incorrectly required the old
exception-path top-level distance display (`None`) to equal the candidate's
valid mission distance. V2 checks unchanged mission and observed-state receipts
and checks each display against its actual terminal behavior.

The original six-case native result and its waiter have since completed and all
their bound source/artifact hashes were verified: native result
`a08496e1d62ec6e00ffae3d85729cc7cf069c2fddcb60d421910c83d30556e80`, waiter
`68b965638210893d5b3cd446eac70d432b7b9d9afd130c2ee64e4d9966b70da2`.
All six stopped at the wrapper failure with 14 observations and 13 completed
zero commands including terminal drain. All raw audits passed; zero round trips.
These are integration failures, not meaningful learned navigation comparisons.
The cumulative completed development episode count is 37, with zero verified
round trips. No independent-layout or real-platform claim is established.
