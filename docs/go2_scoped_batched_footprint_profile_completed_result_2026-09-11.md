# Combined-controller profile completed and authenticated

The original profile exited 0 in session 79227. All 1,428 input/decision rows
match the completed combined replay, with 1,425 forecasts and unchanged model
identity. Completion verification checked 2,157 frozen sources, all eight
output bindings, the completed paired reference and every saved comparison.
All three JSON profile summaries were reconstructed from their bound `.prof`
files. The original process PID 2776882, creation time 1789088841.43, ended.

Result SHA-256:
`c1890e862509753457c1df2fca03555064dbfa1e678f93433572b0ba8f08363e`.
Verification record:
`docs/go2_scoped_batched_footprint_profile_completion_verification_2026-09-11.json`,
SHA-256 `100c32d448a47eeb6860af7a404325c1443cbef76cfc9fae5ef43743bed233df`.
The verification process exited 0 in session 5735; it did not rerun controller
inference or full training ancestry.

| Fixed ten-observation window | Total exclusive profiled time | Scoped footprint cumulative time |
| --- | ---: | ---: |
| Early navigation, 3–12 | 9.419 s | 2.557 s |
| Repeated hold, 395–404 | 14.217 s | 7.761 s |
| Late navigation, 1418–1427 | 16.968 s | 9.972 s |

Late footprint queries account for about 58.8% of total profiled time. Within
that overlapping call tree, `freeze_footprint` takes 2.859 s cumulatively.
The cache-hit path detaches and re-freezes an already validated receipt;
new receipt freezing separately validates and traverses the same graph.
This motivates testing single-pass construction and direct cloning of cached
frozen containers while preserving independent public ownership.

Cumulative times overlap and must not be summed. Profiling overhead is
included; these figures are not additional speedup measurements or isolated
hardware benchmarks. The preceding unprofiled paired replay established the
27.92% incremental total controller-time reduction. All its planning calls
still exceeded 100 ms. Original visibility failure at frame 1173 and the failed
round trip remain. This profile executed no new native command and establishes
neither real-time control nor navigation success.
