# Completed packed/fused controller profile

The original one-model profile completed all 1,428 observations and 1,425
forecast comparisons. Its owner ended. Completion verification passed with
2,189 original source bindings, all eight output bindings, the preceding
packed replay reauthenticated, actual raw/model inputs rehashed, and all
three saved pstats files exactly reconstructing their full JSON summaries.
The verifier source union contains 2,191 paths. No neural replay or full
training-ancestry audit was repeated for this completion check.

- Original result: `5c8b9d3ff1cfd711cf6478dc6ee950e6c7fb86809a4daf67bfd0df0de8f32c45`.
- [Completion verification](go2_packed_fused_profile_completion_verification_2026-09-11.json):
  `f1f68c1bafea843b9a53029f455c171c0ff4ccd2186a78cef59532d44d152244`.
- Verifier: `scripts/verify_go2_packed_fused_profile_completion_v1.py`;
  tool session 91211 exited zero.
- [Caller diagnosis](go2_packed_fused_profile_bottleneck_diagnosis_2026-09-11.json):
  `e675c49e756c8d067442ed6bb691787ddf8c929f6e57c5c7fb94ea4b780dfca2`;
  tool session 33101 exited zero.

| Fixed window | Frames | Total exclusive profiled seconds |
| --- | --- | ---: |
| Early navigation | 3–12 | 7.480755156 |
| Repeated hold | 395–404 | 11.770334960 |
| Late navigation | 1418–1427 | 13.873421589 |

In the late window, fused/scoped footprint evaluation takes 7.945 seconds
cumulative, about 57.3% of the window. Python `deepcopy` has 4,493,101 calls,
1.844 seconds self time and 3.432 seconds cumulative time. Its two largest
non-recursive callers are `LaterResolvedFloorMemory.footprint` (1.213 seconds)
and `confirmed_contact_check` (1.032 seconds). Source inspection shows both
retain deep copies of predecessor contact evidence inside their new receipts.
All measured-bound queries together take about 0.638 seconds cumulative;
further query-enumeration optimization is a secondary candidate.

The next timing investigation should target ordinary nested evidence copying
in those two receipt-construction paths. Any candidate must preserve complete
decision values, independent ownership, alias topology, and failure checks;
it must be compared against the completed packed implementation before
adoption. No such new implementation or speedup is established by this report.
Existing frozen native runs remain unchanged.

Cumulative times overlap and must not be added. Profiling overhead is present,
the host is shared, and these are controller-only windows. They establish no
real-time or navigation qualification. The prior unprofiled paired replay's
9.4% total-time reduction remains a separate result; these profiles do not
establish an additional speedup. State-size snapshots were retained but not
independently reconstructed. The original sensing failure at frame 1173 and
failed round trip remain part of the evidence.
