# Completed thin-obstacle and near-plane counterexamples

The synthetic diagnostic completed in session 28211. Output:
`go2_depth_boundary_counterexamples_v1_attempt_001`.
Launch SHA-256: `fe7038843d217e4ec56beb6c62595e9d62714a6596d2b8be7b7c291d2b3f65a4`.
Result SHA-256: `d928e84da2e947076eb40e6c770d26e3e788766511d6d20cdbf5ddd82b634189`.
All 257 recursive source bindings were checked before and after computation.
Three tests passed in 0.65 s. No native runtime input, scene, model or policy
packet was used or modified.

| Synthetic case | True depth at [244,324] | Reported depth | Bad foreground rays | Original strict result |
| --- | ---: | ---: | ---: | --- |
| Missing 1 mm post | 0.96 m | 1.960000038 m | 60 | Pass |
| Fabricated empty-gap return | 0.96 m | 1.5 m | 60 | Pass |
| Occluder inside native near plane | 0.004 m | 1.960000038 m | 1 | Fail |

In the two post cases, all 60 foreground hits are excluded by the original
2 cm face-interior condition. The strict comparison uses 4,661 other rays and
reports only 5.68e-8 m maximum error. The footprint diagnostic also reports zero
stable-interior bad rays. Its boundary-ambiguous count is zero because that
count is restricted to the original compared domain: excluded thin-post rays
do not enter either reported comparison population. Thus neither interior metric
alone establishes correct boundary/thin-obstacle sensing.

The near-plane case independently reports one clipped opaque ray and one falsely
public-valid ray. Both original checks correctly reject it despite the thin
surface being excluded from the interior metric. Preserve these checks in any
prospective treatment.

These cases are broader than the current development maze's 8 cm walls. They do
not prove an existing run encountered a missing 1 mm post, invalidate a recorded
arrival, or change any saved result. They establish necessary negative controls
for a future boundary-aware measurement model. Qualification remains false.
