# Deferred atomic memo lookups: full paired controller replay V1

Use the complete 1,428-frame history and two fresh original trained models from
the verified single-pass body-projected replay. Keep alternating execution
order, full public packets and decisions, 1,425 model forecasts, seven retained
state comparisons and the original sensing-failure scope. No frame after 1,427
is consumed, and the original frame-1,173 sensing failure remains a limitation
of this timing diagnostic.

Baseline is SinglePassBodyProjectedController. Candidate is
DeferredMemoSinglePassController, changing only the receipt copier in the two
existing pure footprint evidence-building paths. Until the private copying memo
is passed to standard deepcopy, it cannot contain atomic-value overrides and
those lookups are skipped. After any fallback, the original atomic memo lookups
remain enabled for the whole operation, including keys copied after custom
values. Preserve aliases, cycles, custom-copy behavior and independent outputs.
The actual geometry, memory, indices, floor classification, observation,
controller action selection and existing scope/cache guards remain unchanged.

The component experiment used two fixed serialized 3.2 MB decisions, seven
alternating rounds of ten copies per arm. Median baseline times were
12.4046385 and 12.3296161 ms; candidate times were 7.7647524 and 7.8190487 ms.
Its output SHA-256 is
`a54320b92c7272a90b6c2fe406098dfc12b9e958a0083a2024e84f439e5ae405`.
These timings exclude controller execution, and serialization does not preserve
the original runtime alias graph. They are motivation for this prospective
full-controller comparison, not an end-to-end speed claim.

Admit the exact single-pass completion
`f9833329610096f9d5776ccc208d0de33988d8ebd4e6cdff4895004aa5190f1e`
and result `6144645f568d99ccf89b31a76e586277a315f840e2da6db835c06d0519150c44`.
Reauthenticate its original raw/model inputs and complete comparison via the
unchanged prior admission/check functions. Require the original replay owner
and the latest measured-plane controller replay owner to be ended on the
recorded boot before occupying the full CPU replay slot.

Use the established deterministic one-thread CPU environment. Record actual
CPU, RAM, GPU/VRAM, volume and competitor state before launch. Keep the existing
native queue unchanged. Bind all source/tests/protocol and input identities;
verify them before and after replay. Only declared root controller labels and
the new flag are removed when comparing complete decisions. Retain the existing
ten state-type normalization paths; introduce no additional state exclusions.

Output root is `go2_deferred_memo_single_pass_late_history_v1_attempt_001` under
the existing navigation development artifact volume. Preserve failures and
partial output, with no retry, resume or automatic replacement. No profiler is
attached. The result remains a paired timing diagnostic with sensor acquisition
outside its timed region, no native adoption, no navigation qualification and
no real-time or hardware claim.
