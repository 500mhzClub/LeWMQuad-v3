# Budget-only execution comparison prepared

The comparator now joins actual physics, command tapes, public sensor packets
and complete decisions through the original deadline. Only two exact integer
budget fields may differ before that deadline. Earlier divergence is returned
as a negative result, not removed from the comparison. The comparison covers
3,004 observations, 3,003 preceding command intervals and 150,900 physical
samples. Outcomes after the original boundary are not equated.

All 25 focused tests passed in 7.61 seconds (session 82698, exit 0). They cover
both allowed fields, rejected missing/mistyped budgets, earlier prediction,
pose, command, terminal and unrelated-budget drift, incomplete command tapes,
last-prefix physics/public-packet differences, missing raw input bindings and
changed input hashes after comparison. An observation-3004 sentinel confirms
the decision readers stop at the original boundary. Synthetic physical samples
after the boundary may differ without invalidating the earlier prefix.

Preparation session 7395 exited 0 after verifying the previous 1,913-source
preparation and a 1,923-source union with the new comparator, tests, protocol and
explicit imported dependencies. It also read the original first four decision
rows twice, confirmed their canonical fingerprints match and checked both
budget fields. That bounded readout is not a complete-stream hash or model
replay. No actual candidate execution exists to compare yet.

Preparation record:
`docs/go2_extended_budget_anchored_prefix_preparation_2026-09-11.json`

SHA-256: `13fea1e1ff7b30d562b0e48401e0613fc4f235b9808e88accb6a4aa13b7ba367`.

The next integration is an exclusive launcher that authenticates the original
assigned no-RGB direct model, inputs, source closure and completed experiment
queue, and preserves both the full raw audit and this prefix result. This
component does not perform that admission or start a new episode. Original
collection failures and the native queue remain unchanged.
